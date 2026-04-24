from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from encryption import decrypt_url, encrypt_url

logger = logging.getLogger(__name__)

_DATABASES: dict[str, dict[str, dict[str, str]]] = {}


@dataclass
class DatabaseEntry:
    database_id: str
    user_id: str
    db_url_encrypted: str
    sub_database_id: str
    name: str | None
    created_at: str
    updated_at: str


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _generate_database_id() -> str:
    return f"db_{uuid.uuid4().hex[:12]}"


def register_database(
    user_id: str,
    db_url: str,
    sub_database_id: str,
    name: str | None = None,
) -> dict[str, str]:
    if user_id not in _DATABASES:
        _DATABASES[user_id] = {}

    database_id = _generate_database_id()
    encrypted = encrypt_url(db_url)

    _DATABASES[user_id][database_id] = {
        "database_id": database_id,
        "user_id": user_id,
        "db_url_encrypted": encrypted,
        "sub_database_id": sub_database_id,
        "name": name,
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
    }

    return {
        "database_id": database_id,
        "user_id": user_id,
        "sub_database_id": sub_database_id,
        "name": name,
        "created_at": _now_iso(),
    }


def get_database_entry(
    user_id: str,
    database_id: str,
) -> dict[str, str] | None:
    user_dbs = _DATABASES.get(user_id)
    if not user_dbs:
        return None
    entry = user_dbs.get(database_id)
    if not entry:
        return None
    return {
        "database_id": entry["database_id"],
        "user_id": entry["user_id"],
        "sub_database_id": entry["sub_database_id"],
        "name": entry.get("name"),
        "created_at": entry["created_at"],
        "updated_at": entry["updated_at"],
    }


def get_database_url(
    user_id: str,
    database_id: str,
) -> str | None:
    user_dbs = _DATABASES.get(user_id)
    if not user_dbs:
        return None
    entry = user_dbs.get(database_id)
    if not entry:
        return None
    encrypted = entry.get("db_url_encrypted")
    if not encrypted:
        return None
    return decrypt_url(encrypted)


def list_databases(user_id: str) -> list[dict[str, str]]:
    user_dbs = _DATABASES.get(user_id, {})
    return [
        {
            "database_id": data["database_id"],
            "user_id": data["user_id"],
            "sub_database_id": data["sub_database_id"],
            "name": data.get("name"),
            "created_at": data["created_at"],
            "updated_at": data["updated_at"],
        }
        for data in user_dbs.values()
    ]


def get_user_sub_database_ids(user_id: str) -> list[str]:
    user_dbs = _DATABASES.get(user_id, {})
    return [data["sub_database_id"] for data in user_dbs.values()]


def get_database_by_sub_id(
    user_id: str,
    sub_database_id: str,
) -> dict[str, str] | None:
    user_dbs = _DATABASES.get(user_id, {})
    for entry in user_dbs.values():
        if entry["sub_database_id"] == sub_database_id:
            return {
                "database_id": entry["database_id"],
                "user_id": entry["user_id"],
                "sub_database_id": entry["sub_database_id"],
                "name": entry.get("name"),
                "created_at": entry["created_at"],
                "updated_at": entry["updated_at"],
            }
    return None


def get_database_url_by_sub_id(user_id: str, sub_database_id: str) -> str | None:
    """Fetch the decrypted database URL from Supabase persistence."""
    from services import persistence
    try:
        return persistence.get_decrypted_url(user_id, sub_database_id)
    except Exception as exc:
        logger.error("Failed to get database URL from persistence: %s", exc)
        return None


def delete_database(
    user_id: str,
    database_id: str,
) -> bool:
    user_dbs = _DATABASES.get(user_id)
    if not user_dbs:
        return False
    if database_id not in user_dbs:
        return False
    del user_dbs[database_id]
    return True


def delete_by_sub_database_id(user_id: str, sub_database_id: str) -> bool:
    """Delete a database entry by its sub_database_id (shared across both registries)."""
    user_dbs = _DATABASES.get(user_id)
    if not user_dbs:
        return False
    for db_id, entry in list(user_dbs.items()):
        if entry["sub_database_id"] == sub_database_id:
            del user_dbs[db_id]
            return True
    return False


def clear_all_user_data(user_id: str) -> None:
    """No-op: database entries are now persisted in Supabase via services/persistence.py.
    Deletion is handled by state.clear_all_user_data."""
    pass