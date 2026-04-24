"""
Voice Agent WebSocket — Deepgram Agent Proxy with Tool Execution

Architecture:
  Frontend  <->  Backend WS  <->  Deepgram Agent WS

1. Frontend connects, authenticates via Supabase token.
2. Frontend sends `start_conversation` -> backend connects to Deepgram Agent API
   with tool definitions and system prompt tailored to user's databases.
3. Binary audio frames are relayed bidirectionally (mic -> Deepgram, TTS -> frontend).
4. Deepgram text events (transcripts, agent thinking, etc.) are forwarded to frontend.
5. FunctionCallRequest from Deepgram is intercepted:
   - schema_discovery, knowledge_retrieval, get_memory -> executed synchronously,
     result sent back as FunctionCallResponse.
   - sql_agent -> FunctionCallResponse returned immediately ("fetching data..."),
     SQL runs in background, InjectAgentMessage sent when results are ready.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import requests
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed

from config import (
    SUPABASE_URL,
    SUPABASE_PUBLISHABLE_KEY,
    DEEPGRAM_API_KEY,
    DEEPGRAM_AGENT_WS_URL,
    DEEPGRAM_PROXY_CONNECT_TIMEOUT_SECONDS,
    GROQ_API_KEY,
    GROQ_MODEL,
)
from services.state import get_user_databases

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/voice", tags=["voice"])

DEEPGRAM_SAMPLE_RATE = 16000


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def _verify_token_sync(token: str) -> str | None:
    if not SUPABASE_URL or not SUPABASE_PUBLISHABLE_KEY:
        return None
    try:
        resp = requests.get(
            f"{SUPABASE_URL}/auth/v1/user",
            headers={
                "Authorization": f"Bearer {token}",
                "apikey": SUPABASE_PUBLISHABLE_KEY,
            },
            timeout=10,
        )
        if resp.status_code == 200:
            return resp.json().get("id")
    except Exception as exc:
        logger.error("Token verification failed: %s", exc)
    return None


# ---------------------------------------------------------------------------
# Helper: receive a JSON text message, silently discarding binary frames
# ---------------------------------------------------------------------------

async def _receive_json(websocket: WebSocket) -> dict[str, Any] | None:
    """Read messages from the frontend WebSocket until we get a valid JSON
    text frame.  Binary frames (stray audio) are silently discarded.
    Returns None if the socket disconnects."""
    while True:
        msg = await websocket.receive()
        msg_type = msg.get("type")

        if msg_type == "websocket.disconnect":
            return None

        # Text frame -> try to parse JSON
        text = msg.get("text")
        if text:
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                continue

        # Binary frame -> discard (audio arriving before relay phase)
        # Just loop and wait for the next message


# ---------------------------------------------------------------------------
# Deepgram settings builder
# ---------------------------------------------------------------------------

def _build_deepgram_settings(user_id: str) -> dict[str, Any]:
    databases = get_user_databases(user_id)
    db_list = (
        ", ".join(f"'{db.nickname}' (id: {db.sub_database_id})" for db in databases)
        if databases
        else "No databases registered yet"
    )

    instructions = (
        "You are a voice-based SQL database assistant. You help users query "
        "their PostgreSQL databases using natural language.\n\n"
        f"The user has these registered databases: {db_list}\n\n"
        "## When to use tools\n"
        "ONLY call a tool when the user explicitly asks about databases, "
        "tables, data, or wants to run a query. Do NOT call any tool in "
        "response to greetings, small talk, or questions unrelated to "
        "databases (e.g. 'hello', 'how are you', 'what can you do').\n\n"
        "## Workflow\n"
        "1. When the user asks which databases they have or what tables exist, "
        "   call `schema_discovery`.\n"
        "2. When the user asks to fetch or query data:\n"
        "   a. Call `knowledge_retrieval` ONCE and SILENTLY — do NOT describe or explain "
        "      its output to the user. Use it as internal context only.\n"
        "   b. In ONE short sentence confirm what you are about to do, e.g.: "
        "      'Got it — I'll find the total users and API keys per user from "
        "      your gaze-core database. Should I go ahead?'\n"
        "   c. Once the user confirms (yes / go ahead / sure), call `sql_agent` "
        "      immediately. Do NOT ask for confirmation again.\n"
        "   d. Do NOT call `knowledge_retrieval` a second time for the same request.\n"
        "3. When the SQL query finishes, tell the user results are ready and "
        "   ask if they want them.\n"
        "4. When the user says yes, call `get_memory` and summarise the "
        "   findings in 2-3 plain-language sentences.\n\n"
        "## SQL rules — CRITICAL\n"
        "- Copy column and table names CHARACTER-FOR-CHARACTER from the schema. "
        "  NEVER convert camelCase to snake_case or change capitalisation.\n"
        "- Wrap EVERY column name and table name in double quotes to handle "
        "  mixed-case identifiers in PostgreSQL, e.g.:\n"
        '  SELECT "userId", "email" FROM public."user" LIMIT 100\n'
        '  SELECT "userId", COUNT(*) FROM public."apikey" GROUP BY "userId" LIMIT 100\n'
        "- Only write SELECT queries. Always include LIMIT 100.\n\n"
        "## Conversation rules\n"
        "- Keep all responses short and natural — this is a voice conversation.\n"
        "- NEVER explain or narrate database schema details unless the user "
        "  explicitly asks 'what tables do you have?' or similar.\n"
        "- Never call any tool for greetings, pleasantries, or off-topic chat.\n"
        "- When `get_memory` returns results, summarise key findings in plain "
        "  language. Do not read raw column names or every row.\n"
        "- A new query replaces previous results in memory.\n"
        "- Only query databases the user has registered.\n"
    )

    functions = [
        {
            "name": "schema_discovery",
            "description": (
                "List all databases and tables available to the user. "
                "Only call this when the user explicitly asks which databases "
                "they have, what data is available, or what tables exist — "
                "NOT for greetings or general questions."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
        {
            "name": "knowledge_retrieval",
            "description": (
                "SILENT internal lookup. Retrieves exact table names and "
                "column names for a specific database. Call this ONCE before "
                "sql_agent to learn the exact schema. After receiving the "
                "result, immediately confirm the task to the user and wait "
                "for their yes — do NOT call this tool again for the same "
                "request. DO NOT share or describe its output to the user."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "db_nickname": {
                        "type": "string",
                        "description": "Nickname of the database to inspect.",
                    },
                    "user_query": {
                        "type": "string",
                        "description": "What the user wants to know about.",
                    },
                },
                "required": ["db_nickname"],
            },
        },
        {
            "name": "sql_agent",
            "description": (
                "Execute a read-only SQL SELECT query against the user's "
                "database. Data is fetched in the background and stored in "
                "memory. Only call after user confirmation. Always include "
                "LIMIT 100."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "db_nickname": {
                        "type": "string",
                        "description": "Nickname of the database to query.",
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
        {
            "name": "get_memory",
            "description": (
                "Retrieve the most recent SQL query results stored in memory. "
                "Call this when the user wants to hear their query results."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    ]

    think_config: dict[str, Any] = {
        "prompt": instructions,
        "functions": functions,
    }

    # Use Deepgram's native OpenAI integration — billed to Deepgram credits, no key needed.
    # Groq key (if present) is only used for direct SQL agent calls, not for the voice think model.
    think_config["provider"] = {
        "type": "open_ai",
        "model": "gpt-4o-mini",
    }

    return {
        "type": "Settings",
        "audio": {
            "input": {
                "encoding": "linear16",
                "sample_rate": DEEPGRAM_SAMPLE_RATE,
            },
            "output": {
                "encoding": "linear16",
                "sample_rate": DEEPGRAM_SAMPLE_RATE,
                "container": "none",
            },
        },
        "agent": {
            "listen": {
                "provider": {
                    "type": "deepgram",
                    # Flux model (v2) exposes eot_threshold for end-of-turn sensitivity.
                    # Higher value = requires more confidence before considering user done
                    # speaking, which significantly reduces false triggers from background noise.
                    "model": "flux-general-en",
                    "version": "v2",
                    "eot_threshold": 0.87,
                },
            },
            "think": think_config,
            "speak": {
                "provider": {
                    "type": "deepgram",
                    "model": "aura-asteria-en",
                },
            },
            "greeting": "Hello! I'm your database assistant. How can I help you today?",
        },
    }


# ---------------------------------------------------------------------------
# SQL batch tracker — one instance per WebSocket session
# ---------------------------------------------------------------------------

class _SqlBatch:
    """Aggregates parallel sql_agent tasks and fires ONE combined InjectAgentMessage
    only when every task in the current batch has completed.

    Duplicate detection: if the LLM sends the exact same SQL query twice (in the
    same or separate FunctionCallRequests), the second one is skipped.
    Different queries (e.g. user count + API key count) are run in parallel.
    """

    def __init__(self) -> None:
        self._pending: int = 0
        self._results: list[tuple[str, dict]] = []
        self._seen_sqls: set[str] = set()
        self._lock = asyncio.Lock()

    async def try_dispatch(self, sql_query: str) -> bool:
        """Register a task.  Returns False if this exact SQL is already queued."""
        async with self._lock:
            if sql_query in self._seen_sqls:
                return False
            self._seen_sqls.add(sql_query)
            self._pending += 1
            return True

    async def finish(self, query_desc: str, result: dict) -> str | None:
        """Mark one task done.  Returns the combined message when all are done, else None."""
        from services.tools import format_sql_results_for_agent
        async with self._lock:
            self._pending -= 1
            self._results.append((query_desc, result))
            if self._pending > 0:
                return None
            # All tasks done — snapshot, reset, return combined message
            done = list(self._results)
            self._results.clear()
            self._seen_sqls.clear()
        return format_sql_results_for_agent(done)


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------

async def _handle_function_call(
    user_id: str,
    data: dict[str, Any],
    deepgram_ws: ClientConnection,
    frontend: WebSocket,
    sql_batch: _SqlBatch,
    agent_speaking: asyncio.Event,
) -> None:
    """Intercept a Deepgram FunctionCallRequest, execute each tool, respond.

    Deepgram sends: { type: "FunctionCallRequest", functions: [{ id, name, arguments, client_side }] }
    We respond per function: { type: "FunctionCallResponse", id: <id>, name: <name>, content: <string> }
    Required fields per AsyncAPI spec: type, name, content.  id must match the request id.
    """
    from services.tools import execute_tool_for_deepgram

    functions = data.get("functions", [])
    for fn in functions:
        call_id = fn.get("id", "")
        fn_name = fn.get("name", "")
        # arguments is a JSON string
        try:
            fn_input = json.loads(fn.get("arguments", "{}"))
        except (json.JSONDecodeError, TypeError):
            fn_input = {}

        if fn_name == "sql_agent":
            sql_query_text = fn_input.get("sql_query", "")
            if not await sql_batch.try_dispatch(sql_query_text):
                # Exact same SQL already running in this session — ack and skip
                await deepgram_ws.send(json.dumps({
                    "type": "FunctionCallResponse",
                    "id": call_id,
                    "name": fn_name,
                    "content": "That query is already running.",
                }))
                continue
            # ── Async path: ack immediately, run SQL in background ──
            await deepgram_ws.send(json.dumps({
                "type": "FunctionCallResponse",
                "id": call_id,
                "name": fn_name,
                "content": "On it — fetching your data now.",
            }))
            try:
                await frontend.send_json({"type": "SqlQueryStarted"})
            except Exception:
                pass
            asyncio.create_task(
                _async_sql_agent(user_id, fn_input, deepgram_ws, frontend, sql_batch, agent_speaking),
            )
        else:
            # ── Sync path: execute and respond ──
            try:
                output = await asyncio.to_thread(
                    execute_tool_for_deepgram, user_id, fn_name, fn_input,
                )
            except Exception as exc:
                logger.exception("Tool %s raised an unhandled exception", fn_name)
                output = (
                    f"Sorry, the {fn_name} tool encountered an unexpected error. "
                    "Please try again."
                )
            await deepgram_ws.send(json.dumps({
                "type": "FunctionCallResponse",
                "id": call_id,
                "name": fn_name,
                "content": output,
            }))


async def _async_sql_agent(
    user_id: str,
    fn_input: dict[str, Any],
    deepgram_ws: ClientConnection,
    frontend: WebSocket,
    sql_batch: _SqlBatch,
    agent_speaking: asyncio.Event,
) -> None:
    """Run sql_agent in a thread; report to _SqlBatch.
    A combined InjectAgentMessage is fired only when every task in the batch completes,
    and only after the agent has finished speaking the current ack (to avoid overlap).
    """
    from services.tools import sql_agent as _sql_agent_impl

    query_desc = fn_input.get("query", "data query")
    success = False
    try:
        result = await asyncio.to_thread(
            _sql_agent_impl,
            user_id,
            fn_input.get("db_nickname", ""),
            query_desc,
            fn_input.get("sql_query", ""),
        )
        success = result.get("status") == "success"
    except Exception as exc:
        logger.exception("Async SQL agent failed")
        result = {"status": "error", "message": str(exc)}

    # Tell the frontend this specific query finished (clears the badge per query)
    try:
        await frontend.send_json({"type": "SqlQueryCompleted", "success": success})
    except Exception:
        pass

    # Report to batch — combined message returned only when ALL tasks are done
    combined = await sql_batch.finish(query_desc, result)
    if combined is not None:
        # Wait for the agent to finish speaking the ack before injecting the
        # data-ready notification.  This prevents the injection from racing
        # mid-sentence TTS, which Deepgram would interrupt or lose.
        if agent_speaking.is_set():
            deadline = asyncio.get_event_loop().time() + 8.0
            while agent_speaking.is_set():
                if asyncio.get_event_loop().time() >= deadline:
                    break
                await asyncio.sleep(0.1)
        try:
            await deepgram_ws.send(json.dumps({
                "type": "InjectAgentMessage",
                "message": combined,
            }))
        except Exception:
            logger.warning("Could not inject SQL-ready message into Deepgram.")


# ---------------------------------------------------------------------------
# Relay helpers
# ---------------------------------------------------------------------------

async def _relay_frontend_to_deepgram(
    frontend: WebSocket,
    deepgram_ws: ClientConnection,
    agent_speaking: asyncio.Event,
) -> None:
    """Forward mic audio (binary) from the browser to Deepgram.
    Drops mic frames while the agent is speaking to prevent the open mic
    from feeding the agent's own TTS back into Deepgram (echo loop).
    Also handles stop_conversation text command."""
    try:
        while True:
            msg = await frontend.receive()
            msg_type = msg.get("type")
            if msg_type == "websocket.disconnect":
                break

            data_bytes = msg.get("bytes")
            if data_bytes:
                # Suppress mic audio while agent TTS is playing
                if not agent_speaking.is_set():
                    await deepgram_ws.send(data_bytes)
                continue

            data_text = msg.get("text")
            if data_text:
                try:
                    parsed = json.loads(data_text)
                    if parsed.get("type") == "stop_conversation":
                        break
                except json.JSONDecodeError:
                    pass
    except (WebSocketDisconnect, ConnectionClosed):
        pass
    except Exception as exc:
        logger.debug("Upstream relay ended: %s", exc)


async def _relay_deepgram_to_frontend(
    frontend: WebSocket,
    deepgram_ws: ClientConnection,
    user_id: str,
    agent_speaking: asyncio.Event,
    sql_batch: _SqlBatch,
) -> None:
    """Forward TTS audio + events from Deepgram to the browser.
    Intercept FunctionCallRequest to execute tools server-side.
    Sets/clears agent_speaking so the upstream relay can gate the mic."""
    try:
        while True:
            payload = await deepgram_ws.recv()

            if isinstance(payload, bytes):
                # TTS audio -> browser; mark agent as speaking
                agent_speaking.set()
                await frontend.send_bytes(payload)
            else:
                # JSON event
                data = json.loads(payload)
                event_type = data.get("type", "")

                if event_type == "AgentAudioDone":
                    # Agent finished speaking — re-open the mic
                    agent_speaking.clear()
                    await frontend.send_text(payload)
                elif event_type == "FunctionCallRequest":
                    # Notify frontend about each function being called
                    for fn in data.get("functions", []):
                        try:
                            await frontend.send_json({
                                "type": "FunctionCalling",
                                "function_name": fn.get("name", "unknown"),
                            })
                        except Exception:
                            pass
                    await _handle_function_call(user_id, data, deepgram_ws, frontend, sql_batch, agent_speaking)
                else:
                    # Forward transcript / status events to frontend
                    await frontend.send_text(payload)
    except ConnectionClosed as exc:
        logger.info("Deepgram WS closed during relay: code=%s", exc.rcvd)
    except (WebSocketDisconnect, Exception) as exc:
        logger.debug("Downstream relay ended: %s", exc)


async def _keepalive(deepgram_ws: ClientConnection) -> None:
    """Send a KeepAlive ping to Deepgram every 8 seconds to prevent idle disconnect."""
    try:
        while True:
            await asyncio.sleep(8)
            await deepgram_ws.send(json.dumps({"type": "KeepAlive"}))
    except (ConnectionClosed, Exception):
        pass


# ---------------------------------------------------------------------------
# Main WebSocket endpoint
# ---------------------------------------------------------------------------

@router.websocket("/ws/agent")
async def voice_agent_ws(websocket: WebSocket) -> None:
    await websocket.accept()

    user_id: str | None = None
    deepgram_ws: ClientConnection | None = None

    try:
        # ── Phase 1: Authenticate ──
        # Uses _receive_json which silently discards any binary frames
        while True:
            data = await _receive_json(websocket)
            if data is None:
                # Client disconnected
                return
            if data.get("type") == "auth":
                token = data.get("token", "")
                user_id = _verify_token_sync(token)
                if not user_id:
                    await websocket.send_json({"type": "error", "message": "Authentication failed"})
                    await websocket.close(code=4401)
                    return
                await websocket.send_json({"type": "auth_success"})
                break

        # ── Phase 2: Wait for start_conversation ──
        # Uses _receive_json which silently discards any binary frames
        while True:
            data = await _receive_json(websocket)
            if data is None:
                # Client disconnected
                return
            if data.get("type") == "start_conversation":
                break
            # Allow tool list requests while waiting
            if data.get("type") == "get_available_tools":
                from services.tools import TOOLS
                await websocket.send_json({"type": "tool_definitions", "tools": TOOLS})

        # ── Phase 3: Connect to Deepgram ──
        if not DEEPGRAM_API_KEY:
            await websocket.send_json({
                "type": "error",
                "message": "Deepgram API key is not configured on the server.",
            })
            return

        settings = _build_deepgram_settings(user_id)
        logger.info("Connecting to Deepgram Agent API...")

        deepgram_ws = await asyncio.wait_for(
            connect(
                DEEPGRAM_AGENT_WS_URL,
                additional_headers={"Authorization": f"Token {DEEPGRAM_API_KEY}"},
                max_size=None,
                ping_interval=20,
                ping_timeout=20,
            ),
            timeout=DEEPGRAM_PROXY_CONNECT_TIMEOUT_SECONDS,
        )

        # Send agent settings (first message Deepgram expects)
        await deepgram_ws.send(json.dumps(settings))
        logger.info("Deepgram settings sent, waiting for SettingsApplied...")

        # Wait for Deepgram to acknowledge settings before starting relay.
        # This prevents audio from being sent before Deepgram is ready.
        try:
            ack_raw = await asyncio.wait_for(deepgram_ws.recv(), timeout=10)
            if isinstance(ack_raw, str):
                ack = json.loads(ack_raw)
                ack_type = ack.get("type", "")
                logger.info("Deepgram first response: %s", ack_type)

                # Forward the Welcome / SettingsApplied to frontend
                await websocket.send_text(ack_raw)

                # If we got Welcome, wait for SettingsApplied too
                if ack_type == "Welcome":
                    ack_raw2 = await asyncio.wait_for(deepgram_ws.recv(), timeout=10)
                    if isinstance(ack_raw2, str):
                        ack2 = json.loads(ack_raw2)
                        logger.info("Deepgram second response: %s", ack2.get("type", ""))
                        await websocket.send_text(ack_raw2)
                        if ack2.get("type") == "Error":
                            logger.error("Deepgram settings error: %s", ack2)
                            await websocket.send_json({
                                "type": "error",
                                "message": f"Deepgram error: {ack2.get('description', 'Unknown')}",
                            })
                            return

                # Check if the first message was an error
                if ack_type == "Error":
                    logger.error("Deepgram settings error: %s", ack)
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Deepgram error: {ack.get('description', 'Unknown')}",
                    })
                    return
        except asyncio.TimeoutError:
            logger.warning("Deepgram did not acknowledge settings in time, proceeding anyway")

        await websocket.send_json({"type": "conversation_started", "sample_rate": DEEPGRAM_SAMPLE_RATE})

        # ── Phase 4: Bidirectional relay ──
        # Shared event: set while agent TTS is playing so mic is gated.
        agent_speaking = asyncio.Event()
        # One batch tracker per session — aggregates parallel sql_agent results.
        sql_batch = _SqlBatch()

        upstream = asyncio.create_task(
            _relay_frontend_to_deepgram(websocket, deepgram_ws, agent_speaking),
        )
        downstream = asyncio.create_task(
            _relay_deepgram_to_frontend(websocket, deepgram_ws, user_id, agent_speaking, sql_batch),
        )
        keepalive = asyncio.create_task(
            _keepalive(deepgram_ws),
        )

        done, pending = await asyncio.wait(
            {upstream, downstream, keepalive},
            return_when=asyncio.FIRST_COMPLETED,
        )

        # Cancel remaining tasks
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

        # Consume exceptions from completed tasks to avoid
        # "Task exception was never retrieved" warnings
        for task in done:
            if task.exception() is not None:
                logger.debug("Relay task ended with: %s", task.exception())

    except WebSocketDisconnect:
        logger.info("Frontend websocket disconnected.")
    except ConnectionClosed as exc:
        logger.info("Deepgram websocket closed: %s", exc)
        try:
            await websocket.send_json({
                "type": "error",
                "message": "Voice provider connection closed unexpectedly.",
            })
        except Exception:
            pass
    except asyncio.TimeoutError:
        logger.error("Timed out connecting to Deepgram.")
        try:
            await websocket.send_json({
                "type": "error",
                "message": "Could not connect to voice provider (timeout).",
            })
        except Exception:
            pass
    except Exception as exc:
        logger.exception("Voice agent WS error: %s", exc)
        try:
            await websocket.send_json({
                "type": "error",
                "message": f"Server error: {exc}",
            })
        except Exception:
            pass
    finally:
        if deepgram_ws:
            try:
                await deepgram_ws.close()
            except Exception:
                pass
