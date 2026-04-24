"""
SQLAlchemy setup — engine, session factory, ORM models.

Tables are auto-created on startup via Base.metadata.create_all() in main.py.
No manual SQL migrations needed.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, String, create_engine, func
from sqlalchemy.orm import declarative_base, sessionmaker

from config import DATABASE_URL

logger = logging.getLogger(__name__)

Base = declarative_base()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class UserDatabase(Base):
    """Stores user-registered database connections (URL stored encrypted)."""

    __tablename__ = "user_databases"

    database_id      = Column(String, primary_key=True)
    user_id          = Column(String, nullable=False, index=True)
    nickname         = Column(String, nullable=False)
    sub_database_id  = Column(String, nullable=False)
    db_url_encrypted = Column(String, nullable=False)
    created_at       = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        default=lambda: datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# Engine + session factory
# ---------------------------------------------------------------------------

def _build_engine():
    if not DATABASE_URL:
        logger.warning(
            "DATABASE_URL is not set — database persistence is disabled. "
            "Set DATABASE_URL in .env (Supabase → Settings → Database → URI)."
        )
        return None
    return create_engine(
        DATABASE_URL,
        pool_pre_ping=True,   # detect and recycle stale connections
        pool_recycle=1800,    # recycle connections every 30 min
    )


engine = _build_engine()
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False) if engine else None
