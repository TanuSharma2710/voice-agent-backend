from __future__ import annotations

import logging
import os
import time
from typing import Any

from sqlalchemy import create_engine, inspect
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from services.ai import generate_table_description
from vector_store.ingest import create_chunks, embed_and_store

logger = logging.getLogger(__name__)

DEFAULT_EXCLUDED_SCHEMAS = {
    "information_schema",
    "pg_catalog",
    "pg_toast",
    "mysql",
    "performance_schema",
    "sys",
}


def _clean_schema_name(schema_name: str | None) -> str | None:
    if schema_name is None:
        return None
    schema_name = schema_name.strip()
    if not schema_name or schema_name.lower() == "default":
        return None
    return schema_name


def _sub_database_id(engine: Engine, schema_name: str | None) -> str:
    backend = engine.url.get_backend_name()
    normalized = schema_name or "default"
    return f"{backend}:{normalized}"    
    #will it return unique db id


def _normalize_requested_schemas(include_schemas: list[str] | None) -> list[str]:
    if not include_schemas:
        return []
    normalized: list[str] = []
    for schema_name in include_schemas:
        cleaned = (schema_name or "").strip()
        if cleaned:
            normalized.append(cleaned)
    return normalized


def _get_schema_names(engine: Engine, include_schemas: list[str] | None = None) -> list[str | None]:
    inspector = inspect(engine)
    backend = engine.url.get_backend_name().lower()
    requested_schemas = _normalize_requested_schemas(include_schemas)

    if backend == "sqlite":
        return [None]

    try:
        schema_names = inspector.get_schema_names()
    except Exception:
        return [None]

    if backend in {"postgresql", "postgres"}:
        if requested_schemas:
            requested_existing = [schema for schema in requested_schemas if schema in schema_names]
            return requested_existing or ["public"]

        # Safe default for hosted Postgres-like DBs (Supabase, Neon, etc.).
        return ["public"] if "public" in schema_names else [None]

    cleaned: list[str | None] = []
    for schema_name in schema_names:
        normalized = _clean_schema_name(schema_name)
        if normalized is None:
            continue
        if requested_schemas and normalized not in requested_schemas:
            continue
        if normalized.lower() in DEFAULT_EXCLUDED_SCHEMAS:
            continue
        cleaned.append(normalized)

    return cleaned or [None]


def _safe_table_comment(inspector, table_name: str, schema_name: str | None) -> str | None:
    try:
        comment = inspector.get_table_comment(table_name, schema=schema_name)
    except Exception:
        return None

    if isinstance(comment, dict):
        return comment.get("text")
    return None


def _extract_tables(
    engine: Engine,
    include_schemas: list[str] | None = None,
    max_tables: int | None = 100,
) -> list[dict[str, Any]]:
    inspector = inspect(engine)
    tables: list[dict[str, Any]] = []

    for schema_name in _get_schema_names(engine, include_schemas):
        table_names = inspector.get_table_names(schema=schema_name)
        for table_name in table_names:
            if max_tables is not None and len(tables) >= max_tables:
                return tables

            columns = []
            for column in inspector.get_columns(table_name, schema=schema_name):
                columns.append(
                    {
                        "name": column.get("name"),
                        "type": str(column.get("type")) if column.get("type") is not None else None,
                        "nullable": column.get("nullable"),
                        "default": column.get("default"),
                        "primary_key": bool(column.get("primary_key")),
                        "autoincrement": column.get("autoincrement"),
                    }
                )

            pk_constraint = inspector.get_pk_constraint(table_name, schema=schema_name) or {}
            foreign_keys = []
            try:
                raw_foreign_keys = inspector.get_foreign_keys(table_name, schema=schema_name) or []
            except Exception:
                raw_foreign_keys = []

            for fk in raw_foreign_keys:
                foreign_keys.append(
                    {
                        "constrained_columns": fk.get("constrained_columns", []),
                        "referred_schema": fk.get("referred_schema"),
                        "referred_table": fk.get("referred_table"),
                        "referred_columns": fk.get("referred_columns", []),
                    }
                )

            table_schema = {
                "columns": columns,
                "primary_key": pk_constraint.get("constrained_columns", []),
                "foreign_keys": foreign_keys,
                "table_comment": _safe_table_comment(inspector, table_name, schema_name),
            }

            tables.append(
                {
                    "sub_database_id": _sub_database_id(engine, schema_name),
                    "schema_name": schema_name,
                    "table_name": table_name,
                    "schema": table_schema,
                }
            )

    return tables


def _apply_groq_rate_limit(
    requests_in_window: int,
    window_started_at: float,
    requests_per_window: int,
    window_seconds: int,
) -> tuple[int, float]:
    """
    Fixed-window limiter:
    - allow up to `requests_per_window` requests
    - if exceeded before window end, sleep until window resets
    """
    now = time.monotonic()
    elapsed = now - window_started_at

    if elapsed >= window_seconds:
        return 0, now

    if requests_in_window >= requests_per_window:
        sleep_seconds = window_seconds - elapsed
        if sleep_seconds > 0:
            logger.info(
                "Groq rate limit reached (%s requests/%ss). Sleeping %.2f seconds.",
                requests_per_window,
                window_seconds,
                sleep_seconds,
            )
            time.sleep(sleep_seconds)
        return 0, time.monotonic()

    return requests_in_window, window_started_at


def urlprocessor(
    db_url: str,
    user_table_description: str | None = None,
    include_schemas: list[str] | None = None,
    max_tables: int | None = 100,
) -> dict[str, Any]:
    """
    Extract table metadata from the database URL, enrich it with AI descriptions,
    chunk it, and store it in the vector database.
    """

    try:
        engine = create_engine(db_url)
    except Exception as exc:
        return {
            "status": "error",
            "message": f"Unable to create a database engine for the provided URL: {exc}",
            "tables_processed": 0,
        }

    try:
        extracted_tables = _extract_tables(
            engine=engine,
            include_schemas=include_schemas,
            max_tables=max_tables,
        )
    except SQLAlchemyError as exc:
        return {
            "status": "error",
            "message": f"Database metadata extraction failed: {exc}",
            "tables_processed": 0,
        }

    enriched_tables: list[dict[str, Any]] = []
    requests_per_window = int(os.getenv("GROQ_REQUESTS_PER_MINUTE", "20"))
    window_seconds = int(os.getenv("GROQ_RATE_LIMIT_WINDOW_SECONDS", "60"))
    requests_in_window = 0
    window_started_at = time.monotonic()

    try:
        for table in extracted_tables:
            requests_in_window, window_started_at = _apply_groq_rate_limit(
                requests_in_window=requests_in_window,
                window_started_at=window_started_at,
                requests_per_window=requests_per_window,
                window_seconds=window_seconds,
            )
            table_description = generate_table_description(
                table_name=table["table_name"],
                schema=table["schema"],
                user_table_description=user_table_description,
            )
            enriched_tables.append({**table, "table_description": table_description})
            requests_in_window += 1
    except Exception as exc:
        return {
            "status": "error",
            "message": f"Table enrichment failed: {exc}",
            "tables_processed": 0,
        }

    chunks = create_chunks(enriched_tables)
    try:
        store_result = embed_and_store(chunks)
    except Exception as exc:
        logger.exception("Vector storage failed.")
        error_type = type(exc).__name__
        return {
            "status": "error",
            "message": f"Vector storage failed [{error_type}]: {exc}",
            "tables_processed": 0,
        }

    return {
        "status": "success",
        "tables_processed": len(enriched_tables),
        "message": (
            f"Processed {len(enriched_tables)} tables and stored {store_result.get('stored_chunks', 0)} chunks."
        ),
    }
