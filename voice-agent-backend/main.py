from __future__ import annotations

import logging

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router as api_router
from api.voice_ws import router as voice_router

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Voice Agent Backend", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)
app.include_router(voice_router)


@app.on_event("startup")
def create_tables() -> None:
    """Auto-create any missing database tables on startup."""
    from services.db import Base, engine
    if engine is None:
        logger.warning("DATABASE_URL not set — skipping table creation.")
        return
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables verified/created.")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}