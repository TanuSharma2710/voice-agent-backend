from __future__ import annotations

import logging

from dotenv import load_dotenv
from fastapi import FastAPI

from api.routes import router

load_dotenv()

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Voice Agent Backend", version="0.1.0")
app.include_router(router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
