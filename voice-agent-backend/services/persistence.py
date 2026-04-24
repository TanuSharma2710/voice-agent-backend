"""
Persistence layer for user database registrations.

Uses SQLAlchemy ORM with a direct PostgreSQL connection (DATABASE_URL).
Tables are auto-created on startup — no manual SQL migrations needed.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func

from encryption import decrypt_url, encrypt_url
from services.db import SessionLocal, UserDatabase

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_configured() -> bool:
    return SessionLocal is not None


def _check_configured() -> None:
    if not is_configured():
        raise RuntimeError(
            "DATABASE_URL is not set. "
            "Add it to your .env (Supabase → Settings → Database → Connection string → URI)."
        )


def _to_dict(entry: UserDatabase) -> dict[str, Any]:
    return {
        "database_id": entry.database_id,
        "user_id": entry.user_id,
        "nickname": entry.nickname,
        "sub_database_id": entry.sub_database_id,
        "created_at": entry.created_at.isoformat() if entry.created_at else "",
    }


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------

def insert(
    user_id: str,
    nickname: str,
    sub_database_id: str,
    db_url: str,
    database_id: str | None = None,
) -> dict[str, Any]:
    _check_configured()
    session = SessionLocal()
    try:
        entry = UserDatabase(
            database_id=database_id or f"db_{uuid.uuid4().hex[:12]}",
            user_id=user_id,
            nickname=nickname,
            sub_database_id=sub_database_id,
            db_url_encrypted=encrypt_url(db_url),
            created_at=datetime.now(timezone.utc),
        )
        session.add(entry)
        session.commit()
        session.refresh(entry)
        return _to_dict(entry)
    finally:
        session.close()


def list_by_user(user_id: str) -> list[dict[str, Any]]:
    _check_configured()
    session = SessionLocal()
    try:
        entries = (
            session.query(UserDatabase)
            .filter(UserDatabase.user_id == user_id)
            .order_by(UserDatabase.created_at)
            .all()
        )
        return [_to_dict(e) for e in entries]
    finally:
        session.close()


def get_by_id(user_id: str, database_id: str) -> dict[str, Any] | None:
    _check_configured()
    session = SessionLocal()
    try:
        entry = (
            session.query(UserDatabase)
            .filter(
                UserDatabase.user_id == user_id,
                UserDatabase.database_id == database_id,
            )
            .first()
        )
        return _to_dict(entry) if entry else None
    finally:
        session.close()


def get_by_sub_id(user_id: str, sub_database_id: str) -> dict[str, Any] | None:
    _check_configured()
    session = SessionLocal()
    try:
        entry = (
            session.query(UserDatabase)
            .filter(
                UserDatabase.user_id == user_id,
                UserDatabase.sub_database_id == sub_database_id,
            )
            .first()
        )
        return _to_dict(entry) if entry else None
    finally:
        session.close()


def get_by_nickname(user_id: str, nickname: str) -> dict[str, Any] | None:
    """Case-insensitive exact match on nickname."""
    _check_configured()
    session = SessionLocal()
    try:
        entry = (
            session.query(UserDatabase)
            .filter(
                UserDatabase.user_id == user_id,
                func.lower(UserDatabase.nickname) == nickname.lower(),
            )
            .first()
        )
        return _to_dict(entry) if entry else None
    finally:
        session.close()


def update_nickname(user_id: str, database_id: str, nickname: str) -> bool:
    _check_configured()
    session = SessionLocal()
    try:
        updated = (
            session.query(UserDatabase)
            .filter(
                UserDatabase.user_id == user_id,
                UserDatabase.database_id == database_id,
            )
            .update({"nickname": nickname})
        )
        session.commit()
        return updated > 0
    finally:
        session.close()


def delete_by_id(user_id: str, database_id: str) -> dict[str, Any] | None:
    """Delete and return the deleted row dict (caller needs sub_database_id for Qdrant cleanup)."""
    _check_configured()
    session = SessionLocal()
    try:
        entry = (
            session.query(UserDatabase)
            .filter(
                UserDatabase.user_id == user_id,
                UserDatabase.database_id == database_id,
            )
            .first()
        )
        if not entry:
            return None
        result = _to_dict(entry)
        session.delete(entry)
        session.commit()
        return result
    finally:
        session.close()


def delete_all_for_user(user_id: str) -> None:
    _check_configured()
    session = SessionLocal()
    try:
        session.query(UserDatabase).filter(UserDatabase.user_id == user_id).delete()
        session.commit()
    finally:
        session.close()


def get_decrypted_url(user_id: str, sub_database_id: str) -> str | None:
    """Fetch and decrypt the stored database URL for SQL execution."""
    _check_configured()
    session = SessionLocal()
    try:
        entry = (
            session.query(UserDatabase)
            .filter(
                UserDatabase.user_id == user_id,
                UserDatabase.sub_database_id == sub_database_id,
            )
            .first()
        )
        if not entry:
            return None
        return decrypt_url(entry.db_url_encrypted)
    finally:
        session.close()
