"""
Voice Agent State Management

Handles:
- User sessions
- Registered databases (nickname → sub_database_id → URL)
- Memory blocks (temporary SQL results)
- Background jobs
- Conversation context
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DatabaseEntry:
    database_id: str
    user_id: str
    nickname: str
    sub_database_id: str


@dataclass
class MemoryBlock:
    memory_id: str
    user_id: str
    results: list[dict[str, Any]] = field(default_factory=list)
    query: str = ""
    status: str = "pending"  # pending, ready, consumed
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class BackgroundJob:
    job_id: str
    user_id: str
    query: str
    status: str = "running"  # running, completed, failed
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# In-memory storage for session-scoped data (memory blocks, jobs, context).
# Database registrations are persisted to Supabase via services/persistence.py.
_memory_blocks: dict[str, MemoryBlock] = {}
_background_jobs: dict[str, BackgroundJob] = {}
_conversation_context: dict[str, dict[str, Any]] = {}

_state_lock = threading.Lock()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# =============================================================================
# Database Registry  (backed by Supabase via services/persistence.py)
# =============================================================================

def register_db(
    user_id: str,
    nickname: str,
    sub_database_id: str,
    db_url: str = "",
) -> DatabaseEntry:
    from services import persistence
    row = persistence.insert(
        user_id=user_id,
        nickname=nickname,
        sub_database_id=sub_database_id,
        db_url=db_url,
    )
    return DatabaseEntry(
        database_id=row["database_id"],
        user_id=row["user_id"],
        nickname=row["nickname"],
        sub_database_id=row["sub_database_id"],
    )


def get_user_databases(user_id: str) -> list[DatabaseEntry]:
    from services import persistence
    try:
        rows = persistence.list_by_user(user_id)
        return [
            DatabaseEntry(
                database_id=r["database_id"],
                user_id=r["user_id"],
                nickname=r["nickname"],
                sub_database_id=r["sub_database_id"],
            )
            for r in rows
        ]
    except Exception as exc:
        logger.error("Failed to list databases from persistence: %s", exc)
        return []


def get_database_by_nickname(user_id: str, nickname: str) -> DatabaseEntry | None:
    from services import persistence
    try:
        row = persistence.get_by_nickname(user_id, nickname)
        if not row:
            return None
        return DatabaseEntry(
            database_id=row["database_id"],
            user_id=row["user_id"],
            nickname=row["nickname"],
            sub_database_id=row["sub_database_id"],
        )
    except Exception as exc:
        logger.error("Failed to get database by nickname: %s", exc)
        return None


def delete_database(user_id: str, database_id: str) -> tuple[bool, str | None]:
    from services import persistence
    try:
        row = persistence.delete_by_id(user_id, database_id)
        if row:
            return True, row["sub_database_id"]
        return False, None
    except Exception as exc:
        logger.error("Failed to delete database: %s", exc)
        return False, None


def update_database_nickname(user_id: str, database_id: str, nickname: str) -> bool:
    from services import persistence
    try:
        return persistence.update_nickname(user_id, database_id, nickname)
    except Exception as exc:
        logger.error("Failed to update nickname: %s", exc)
        return False


# =============================================================================
# Memory Blocks (SQL Results Storage)
# =============================================================================

def store_memory(
    user_id: str,
    results: list[dict[str, Any]],
    query: str = "",
) -> MemoryBlock:
    with _state_lock:
        # Clear old memory for this user
        user_memory_ids = [
            mid for mid, mem in _memory_blocks.items()
            if mem.user_id == user_id
        ]
        for mid in user_memory_ids:
            del _memory_blocks[mid]

        # Create new memory block
        memory_id = f"mem_{uuid.uuid4().hex[:12]}"
        memory = MemoryBlock(
            memory_id=memory_id,
            user_id=user_id,
            results=results,
            query=query,
            status="ready",
        )

        _memory_blocks[memory_id] = memory
        return memory


def get_memory(user_id: str) -> MemoryBlock | None:
    with _state_lock:
        for memory in _memory_blocks.values():
            if memory.user_id == user_id and memory.status == "ready":
                memory.status = "consumed"
                return memory
        return None


def peek_memory(user_id: str) -> MemoryBlock | None:
    with _state_lock:
        for memory in _memory_blocks.values():
            if memory.user_id == user_id and memory.status == "ready":
                return memory
        return None


def clear_memory(user_id: str) -> None:
    with _state_lock:
        memory_ids = [
            mid for mid, mem in _memory_blocks.items()
            if mem.user_id == user_id
        ]
        for mid in memory_ids:
            del _memory_blocks[mid]


# =============================================================================
# Background Jobs
# =============================================================================

def create_job(user_id: str, query: str) -> BackgroundJob:
    with _state_lock:
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        job = BackgroundJob(
            job_id=job_id,
            user_id=user_id,
            query=query,
            status="running",
        )
        _background_jobs[job_id] = job
        return job


def complete_job(job_id: str, results: list[dict[str, Any]]) -> MemoryBlock | None:
    with _state_lock:
        job = _background_jobs.get(job_id)
        if not job:
            return None

        job.status = "completed"

        # Store results in memory
        memory_id = f"mem_{uuid.uuid4().hex[:12]}"
        memory = MemoryBlock(
            memory_id=memory_id,
            user_id=job.user_id,
            results=results,
            query=job.query,
            status="ready",
        )

        _memory_blocks[memory_id] = memory
        return memory


def fail_job(job_id: str, error: str) -> None:
    with _state_lock:
        job = _background_jobs.get(job_id)
        if job:
            job.status = "failed"


def get_pending_results(user_id: str) -> list[MemoryBlock]:
    with _state_lock:
        return [
            mem for mem in _memory_blocks.values()
            if mem.user_id == user_id and mem.status == "ready"
        ]


# =============================================================================
# Conversation Context
# =============================================================================

def update_context(user_id: str, **kwargs: Any) -> None:
    with _state_lock:
        if user_id not in _conversation_context:
            _conversation_context[user_id] = {}
        _conversation_context[user_id].update(kwargs)


def get_context(user_id: str) -> dict[str, Any]:
    with _state_lock:
        return _conversation_context.get(user_id, {}).copy()


def clear_context(user_id: str) -> None:
    with _state_lock:
        if user_id in _conversation_context:
            del _conversation_context[user_id]


def clear_all_user_data(user_id: str) -> None:
    """Remove all data for a user: Supabase DB entries + in-memory session data."""
    from services import persistence
    try:
        persistence.delete_all_for_user(user_id)
    except Exception as exc:
        logger.error("Failed to clear user databases from persistence: %s", exc)
    with _state_lock:
        # Clear memory blocks
        memory_ids = [mid for mid, mem in _memory_blocks.items() if mem.user_id == user_id]
        for mid in memory_ids:
            del _memory_blocks[mid]
        # Clear jobs
        job_ids = [jid for jid, job in _background_jobs.items() if job.user_id == user_id]
        for jid in job_ids:
            del _background_jobs[jid]
        # Clear context
        _conversation_context.pop(user_id, None)