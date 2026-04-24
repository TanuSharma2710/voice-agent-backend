"""
Voice Agent Tools

Tools called by the Deepgram Voice Agent via FunctionCallRequest.
Each tool returns a human-readable string that Deepgram's agent incorporates
into its spoken response.

Tools:
  1. schema_discovery   — list user's registered databases
  2. knowledge_retrieval — get detailed schema for a specific database
  3. sql_agent           — execute read-only SQL, store results in memory
  4. get_memory          — retrieve stored SQL results
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, time, date
from decimal import Decimal
from typing import Any

from sqlalchemy import create_engine, text

from config import SQL_RESULT_MAX_ROWS, SQL_QUERY_TIMEOUT_SECONDS
from services.database_registry import get_database_url_by_sub_id
from services.state import (
    get_memory,
    peek_memory,
    store_memory,
    get_user_databases,
    get_database_by_nickname,
)
from vector_store.retrieval import retrieve_schema_context, summarize_tables_from_chunks

logger = logging.getLogger(__name__)


# =========================================================================
# Deepgram tool definitions (exposed for frontend "get_available_tools")
# =========================================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "schema_discovery",
            "description": (
                "Discover what databases and tables are available to the user. "
                "Use this first when the user asks about their data structure."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "knowledge_retrieval",
            "description": (
                "Retrieve detailed schema information (tables, columns, "
                "relationships) for a specific database. Use before running queries."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "db_nickname": {
                        "type": "string",
                        "description": "The nickname of the database.",
                    },
                    "user_query": {
                        "type": "string",
                        "description": "What the user is looking for.",
                    },
                },
                "required": ["db_nickname"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sql_agent",
            "description": (
                "Execute a read-only SQL SELECT query against the user's "
                "database. Results are stored in memory for later retrieval."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "db_nickname": {
                        "type": "string",
                        "description": "The nickname of the database to query.",
                    },
                    "query": {
                        "type": "string",
                        "description": "Natural-language description of what to fetch.",
                    },
                    "sql_query": {
                        "type": "string",
                        "description": "Read-only SQL SELECT query (must have LIMIT 100).",
                    },
                },
                "required": ["db_nickname", "query", "sql_query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_memory",
            "description": (
                "Retrieve the most recent SQL query results stored in memory. "
                "Use when user asks for their data or results."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


# =========================================================================
# SQL safety
# =========================================================================

FORBIDDEN_SQL_KEYWORDS = {
    "insert", "update", "delete", "drop", "alter", "create", "truncate",
    "grant", "revoke", "merge", "replace", "upsert", "call", "execute",
    "vacuum", "analyze", "refresh", "copy",
}

# SELECT INTO new_table is a PostgreSQL write operation that starts with SELECT
# and contains no other blocked keyword — catch it explicitly.
_SELECT_INTO_RE = re.compile(r"\bselect\b.+\binto\b\s+\w", re.IGNORECASE | re.DOTALL)


def _sanitize_sql(sql_query: str) -> str:
    cleaned = sql_query.strip()
    cleaned = re.sub(r"^```(?:sql)?", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    cleaned = cleaned.rstrip(";").strip()
    return cleaned


def _ensure_read_only_sql(sql_query: str) -> str:
    cleaned = _sanitize_sql(sql_query)
    if not cleaned:
        raise ValueError("SQL query was empty.")
    if ";" in cleaned:
        raise ValueError("Only single-statement SQL allowed.")

    # Strip comments before all keyword checks
    lowered = re.sub(r"--.*?$", "", cleaned, flags=re.MULTILINE)
    lowered = re.sub(r"/\*.*?\*/", "", lowered, flags=re.DOTALL).lower().strip()

    if not re.match(r"^(select|with)\b", lowered):
        raise ValueError("Only SELECT / WITH read-only queries are allowed.")

    # Catch SELECT INTO new_table (PostgreSQL table-creation write operation)
    if _SELECT_INTO_RE.search(lowered):
        raise ValueError("SELECT INTO is not allowed — use SELECT without INTO.")

    for keyword in FORBIDDEN_SQL_KEYWORDS:
        if re.search(rf"\b{keyword}\b", lowered):
            raise ValueError(f"SQL contains forbidden keyword: {keyword}")

    return cleaned


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


# =========================================================================
# Tool implementations (return dicts)
# =========================================================================

def schema_discovery(user_id: str) -> dict[str, Any]:
    databases = get_user_databases(user_id)
    if not databases:
        return {
            "status": "no_databases",
            "message": "You haven't registered any databases yet.",
            "databases": [],
        }
    return {
        "status": "success",
        "message": f"You have {len(databases)} registered database(s).",
        "databases": [
            {
                "database_id": db.database_id,
                "nickname": db.nickname,
                "sub_database_id": db.sub_database_id,
            }
            for db in databases
        ],
    }


def knowledge_retrieval(
    user_id: str,
    db_nickname: str,
    user_query: str = "",
) -> dict[str, Any]:
    db = get_database_by_nickname(user_id, db_nickname)
    if not db:
        return {
            "status": "error",
            "message": f"Database '{db_nickname}' not found. Check your registered databases.",
        }

    query_text = user_query or db_nickname
    schema_lookup = retrieve_schema_context(
        query_text=query_text,
        sub_database_ids=[db.sub_database_id],
        limit=10,
    )

    if schema_lookup.get("error") == "qdrant_unreachable":
        return {
            "status": "error",
            "message": (
                "The schema database is currently unreachable. "
                "This usually means the Qdrant Cloud cluster is paused. "
                "Please resume it from the Qdrant Cloud dashboard and try again."
            ),
        }

    chunks = schema_lookup.get("chunks", [])
    context_text = schema_lookup.get("context_text", "")

    if not chunks:
        return {
            "status": "no_schema",
            "message": f"No schema found for '{db_nickname}'. It may not be processed yet.",
        }

    tables = summarize_tables_from_chunks(chunks, max_tables=20)
    return {
        "status": "success",
        "database": db_nickname,
        "tables": tables,
        "schema_details": context_text[:3000],
        "chunks_count": len(chunks),
    }


def _attempt_column_fix(sql: str, actual_cols: dict[str, list[str]]) -> str | None:
    """Try to auto-correct quoted column names by normalized comparison.

    Handles camelCase ↔ snake_case (userId ↔ user_id) and simple casing
    differences.  Returns corrected SQL, or None if no change was needed.
    """
    # Build norm → actual map (first match wins to avoid ambiguity)
    norm_to_actual: dict[str, str] = {}
    for cols in actual_cols.values():
        for col in cols:
            norm = col.lower().replace("_", "")
            if norm not in norm_to_actual:
                norm_to_actual[norm] = col

    fixed = sql
    changed = False
    for match in re.finditer(r'"(\w+)"', sql):
        used = match.group(1)
        norm = used.lower().replace("_", "")
        if norm in norm_to_actual:
            actual = norm_to_actual[norm]
            if actual != used:
                fixed = fixed.replace(f'"{used}"', f'"{actual}"')
                changed = True
    return fixed if changed else None


def _run_sql_inner(engine: Any, sql: str, user_id: str, query: str) -> dict[str, Any]:
    """Execute one validated read-only SQL statement and store results.
    Raises on any database error (caller decides how to handle)."""
    with engine.begin() as connection:
        connection.execute(text("SET TRANSACTION READ ONLY"))
        try:
            connection.execution_options(timeout=SQL_QUERY_TIMEOUT_SECONDS)
        except Exception:
            pass
        result = connection.execute(text(sql))
        columns = list(result.keys())
        fetched = result.fetchmany(SQL_RESULT_MAX_ROWS + 1)
        truncated = len(fetched) > SQL_RESULT_MAX_ROWS
        if truncated:
            fetched = fetched[:SQL_RESULT_MAX_ROWS]
        rows: list[dict[str, Any]] = []
        for row in fetched:
            row_dict: dict[str, Any] = {}
            for idx, col in enumerate(columns):
                row_dict[col] = _json_safe(row[idx])
            rows.append(row_dict)
    memory = store_memory(user_id, rows, query)
    return {
        "status": "success",
        "message": f"Query executed. {len(rows)} rows returned.",
        "row_count": len(rows),
        "truncated": truncated,
        "memory_id": memory.memory_id,
    }


def _fetch_actual_columns(engine: Any, sql_query: str) -> dict[str, list[str]]:
    """Query information_schema to return the real column names for every table
    referenced in *sql_query*.  Returns a dict keyed by '"schema"."table"'."""
    table_pattern = re.compile(
        r'(?:FROM|JOIN)\s+(?:"?(\w+)"?\.)?"?(\w+)"?',
        re.IGNORECASE,
    )
    matches = table_pattern.findall(sql_query)
    columns_by_table: dict[str, list[str]] = {}
    try:
        with engine.connect() as conn:
            for schema_name, table_name in matches:
                schema_name = schema_name or "public"
                rows = conn.execute(
                    text(
                        "SELECT column_name FROM information_schema.columns "
                        "WHERE table_schema = :schema AND table_name = :table "
                        "ORDER BY ordinal_position"
                    ),
                    {"schema": schema_name, "table": table_name},
                ).fetchall()
                if rows:
                    key = f'"{schema_name}"."{table_name}"'
                    columns_by_table[key] = [r[0] for r in rows]
    except Exception as exc:
        logger.debug("Could not fetch actual columns from information_schema: %s", exc)
    return columns_by_table


def sql_agent(
    user_id: str,
    db_nickname: str,
    query: str,
    sql_query: str,
) -> dict[str, Any]:
    """Execute read-only SQL and store results in memory."""
    db = get_database_by_nickname(user_id, db_nickname)
    if not db:
        return {"status": "error", "message": f"Database '{db_nickname}' not found."}

    # Get the actual connection URL from the encrypted registry
    db_url = get_database_url_by_sub_id(user_id, db.sub_database_id)
    if not db_url:
        return {
            "status": "error",
            "message": f"Connection URL not found for '{db_nickname}'. Please re-register the database.",
        }

    try:
        validated_sql = _ensure_read_only_sql(sql_query)
    except ValueError as exc:
        return {"status": "error", "message": str(exc)}

    engine = create_engine(db_url)
    try:
        return _run_sql_inner(engine, validated_sql, user_id, query)
    except Exception as exc:
        orig = getattr(exc, "orig", None)
        pgcode = getattr(orig, "pgcode", "") if orig else ""
        if pgcode == "42703":
            # Undefined column — look up actual names and attempt auto-correction
            actual_cols = _fetch_actual_columns(engine, validated_sql)
            fixed_sql = _attempt_column_fix(validated_sql, actual_cols)
            if fixed_sql:
                logger.info("Auto-correcting column names and retrying SQL")
                try:
                    return _run_sql_inner(create_engine(db_url), fixed_sql, user_id, query)
                except Exception:
                    logger.exception("SQL retry after auto-correction also failed")
            return {
                "status": "error",
                "message": "SQL column not found. The query could not be completed.",
            }
        logger.exception("SQL execution failed")
        return {"status": "error", "message": "SQL execution failed."}


def get_memory_impl(user_id: str) -> dict[str, Any]:
    memory = peek_memory(user_id)
    if not memory:
        return {
            "status": "no_memory",
            "message": "No stored results. Please run a query first.",
        }
    return {
        "status": "success",
        "query": memory.query,
        "row_count": len(memory.results),
        "results": memory.results,
    }


# =========================================================================
# Deepgram-facing dispatcher — returns a *string*
# =========================================================================

def _format_for_deepgram(result: dict[str, Any], tool_name: str) -> str:
    """Convert a tool result dict into a human-readable string for the agent."""
    status = result.get("status", "")

    if status == "no_databases":
        return "You haven't registered any databases yet. Please register one through the web interface."

    if status == "error":
        return result.get("message", "An error occurred.")

    if tool_name == "schema_discovery":
        databases = result.get("databases", [])
        if databases:
            names = ", ".join(d.get("nickname", d.get("database_id", "")) for d in databases)
            return f"You have {len(databases)} registered database(s): {names}."
        return "No databases found."

    if tool_name == "knowledge_retrieval":
        tables = result.get("tables", "")
        db = result.get("database", "")
        details = result.get("schema_details", "")
        if tables:
            return (
                f"[INTERNAL SCHEMA CONTEXT for '{db}' — DO NOT narrate or describe this to the user]\n\n"
                f"Tables: {tables}\n\n"
                f"Full schema:\n{details}\n\n"
                f"[CRITICAL INSTRUCTIONS:\n"
                f"1. DO NOT summarise or explain this schema to the user.\n"
                f"2. Use it silently to confirm you can answer the user's request.\n"
                f"3. When writing SQL, copy column names CHARACTER-FOR-CHARACTER from the schema above.\n"
                f"   NEVER convert camelCase to snake_case (e.g. if schema shows 'userId', write \"userId\", NOT user_id).\n"
                f"4. Wrap EVERY column and table name in double quotes in PostgreSQL SQL.\n"
                f"5. After calling this tool, respond with ONE sentence confirming the task, e.g.:\n"
                f"   'Got it — I'll find X from your Y database. Should I go ahead?']"
            )
        return result.get("message", "No schema information found.")

    if tool_name == "get_memory":
        if status == "no_memory":
            return "No stored results. Please run a query first."
        rows = result.get("results", [])
        row_count = result.get("row_count", len(rows))
        query = result.get("query", "")
        if not rows:
            return "The query returned no results."
        cols = list(rows[0].keys())
        summary = f"Query: {query}\nResults: {row_count} rows. Columns: {', '.join(cols)}.\n\n"
        for i, row in enumerate(rows[:10]):
            parts = ", ".join(f"{k}={v}" for k, v in row.items())
            summary += f"Row {i + 1}: {parts}\n"
        if row_count > 10:
            summary += f"\n…and {row_count - 10} more rows."
        return summary

    return result.get("message", "Done.")


def format_sql_results_for_agent(results: list[tuple[str, dict]]) -> str:
    """Generate a short natural spoken announcement for InjectAgentMessage.

    IMPORTANT: InjectAgentMessage is spoken VERBATIM by the agent — this must
    be plain natural language only.  No brackets, no instructions, no SQL.
    """
    successes = [(d, r) for d, r in results if r.get("status") == "success"]
    failures  = [(d, r) for d, r in results if r.get("status") != "success"]

    if not failures:
        if len(successes) == 1:
            n = successes[0][1].get("row_count", 0)
            plural = "s" if n != 1 else ""
            return (
                f"Your data is ready! I found {n} result{plural}. "
                "Would you like me to share the details?"
            )
        parts = " and ".join(
            f"{r.get('row_count', 0)} result{'s' if r.get('row_count', 0) != 1 else ''}"
            for _, r in successes
        )
        return (
            f"All your data is ready — {parts} across {len(successes)} queries. "
            "Would you like me to share the details?"
        )

    if not successes:
        return (
            "The query had an issue and couldn't be completed. "
            "Would you like to try again?"
        )

    # Mixed: some succeeded, some failed
    total_rows = sum(r.get("row_count", 0) for _, r in successes)
    n_fail = len(failures)
    plural_rows = "s" if total_rows != 1 else ""
    plural_fail = "queries" if n_fail > 1 else "query"
    return (
        f"I got some of your data — {total_rows} result{plural_rows} fetched. "
        f"However, {n_fail} {plural_fail} had an issue and couldn't be completed. "
        "Would you like me to retry, or shall I share what I have?"
    )

    return "\n".join(lines)


def execute_tool_for_deepgram(
    user_id: str,
    tool_name: str,
    arguments: dict[str, Any],
) -> str:
    """Execute a tool and return a string for Deepgram FunctionCallResponse.

    Any exception raised by a tool is caught here so the relay WebSocket is
    never killed by a tool failure.
    """
    try:
        if tool_name == "schema_discovery":
            result = schema_discovery(user_id)
        elif tool_name == "knowledge_retrieval":
            result = knowledge_retrieval(
                user_id,
                arguments.get("db_nickname", ""),
                arguments.get("user_query", ""),
            )
        elif tool_name == "get_memory":
            result = get_memory_impl(user_id)
        else:
            result = {"status": "error", "message": f"Unknown tool: {tool_name}"}

        return _format_for_deepgram(result, tool_name)
    except Exception as exc:
        logger.exception("Tool %s failed unexpectedly", tool_name)
        return (
            f"Sorry, I ran into an error while executing {tool_name}. "
            f"Details: {exc}"
        )


def sql_agent_for_deepgram(
    user_id: str,
    db_nickname: str,
    query: str,
    sql_query: str,
) -> str:
    """Run sql_agent synchronously and return a notification string."""
    result = sql_agent(user_id, db_nickname, query, sql_query)
    if result.get("status") == "success":
        row_count = result.get("row_count", 0)
        return (
            f"Your data is ready! I found {row_count} result(s) from your query. "
            "Would you like me to share the details?"
        )
    return f"There was an issue with your query: {result.get('message', 'Unknown error')}."
