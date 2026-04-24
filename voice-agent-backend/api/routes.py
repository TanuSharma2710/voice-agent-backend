from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Header, HTTPException, status

from schemas.datamodels import (
    DatabaseDeleteResponse,
    DatabaseListResponse,
    DatabaseRegisterRequest,
    DatabaseRegisterResponse,
    DatabaseUpdateRequest,
    DatabaseUpdateResponse,
)
from services.auth import AuthError, get_auth_service
from services.state import (
    register_db,
    get_user_databases,
    delete_database,
    update_database_nickname,
    clear_all_user_data as clear_state,
)
from services.database_registry import (
    get_database_url_by_sub_id,
    clear_all_user_data as clear_registry,
)
from services.processing import urlprocessor
from vector_store.retrieval import delete_chunks_by_sub_database_id, wipe_collection

router = APIRouter(prefix="/api", tags=["api"])
logger = logging.getLogger(__name__)


async def get_current_user_id(authorization: str = Header(...)) -> str:
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "INVALID_AUTH_HEADER", "message": "Authorization header must start with 'Bearer'."},
        )
    token = authorization[7:]
    try:
        auth_service = get_auth_service()
        return auth_service.get_user_id(token)
    except AuthError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "AUTH_FAILED", "message": str(exc)},
        )


@router.post("/databases/register", response_model=DatabaseRegisterResponse)
async def register_db_endpoint(
    payload: DatabaseRegisterRequest,
    user_id: str = Depends(get_current_user_id),
) -> DatabaseRegisterResponse:
    result = urlprocessor(
        db_url=payload.db_url,
        user_id=user_id,
    )

    if result.get("status") != "success":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "PROCESSING_FAILED",
                "message": result.get("message", "Processing failed."),
                "tables_processed": result.get("tables_processed", 0),
            },
        )

    sub_database_id = result.get("sub_database_id")
    if not sub_database_id:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "NO_SUB_DATABASE_ID", "message": "Failed to generate sub_database_id"},
        )

    # Store in Supabase via state.py → persistence.py (single write, both registries merged)
    entry = register_db(
        user_id=user_id,
        nickname=payload.nickname or f"db_{sub_database_id}",
        sub_database_id=sub_database_id,
        db_url=payload.db_url,
    )

    return DatabaseRegisterResponse(
        database_id=entry.database_id,
        user_id=entry.user_id,
        nickname=entry.nickname,
        sub_database_id=entry.sub_database_id,
    )


@router.get("/databases", response_model=DatabaseListResponse)
async def list_dbs(
    user_id: str = Depends(get_current_user_id),
) -> DatabaseListResponse:
    databases = get_user_databases(user_id)
    return DatabaseListResponse(
        databases=[
            {
                "database_id": db.database_id,
                "user_id": db.user_id,
                "nickname": db.nickname,
                "sub_database_id": db.sub_database_id,
            }
            for db in databases
        ]
    )


@router.delete("/databases/{database_id}", response_model=DatabaseDeleteResponse)
async def delete_db(
    database_id: str,
    user_id: str = Depends(get_current_user_id),
) -> DatabaseDeleteResponse:
    # Delete from Supabase (via state.py → persistence.py) — returns sub_database_id
    success, sub_database_id = delete_database(user_id, database_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "DATABASE_NOT_FOUND", "message": "Database not found."},
        )

    # Delete vector chunks from Qdrant
    if sub_database_id:
        try:
            delete_chunks_by_sub_database_id([sub_database_id])
        except Exception as exc:
            logger.warning("Failed to delete Qdrant chunks for %s: %s", database_id, exc)

    return DatabaseDeleteResponse(success=True, database_id=database_id)


@router.patch("/databases/{database_id}", response_model=DatabaseUpdateResponse)
async def update_db(
    database_id: str,
    payload: DatabaseUpdateRequest,
    user_id: str = Depends(get_current_user_id),
) -> DatabaseUpdateResponse:
    success = update_database_nickname(user_id, database_id, payload.nickname)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "DATABASE_NOT_FOUND", "message": "Database not found."},
        )
    return DatabaseUpdateResponse(success=True, database_id=database_id, nickname=payload.nickname)


@router.post("/reset")
async def reset_all(
    user_id: str = Depends(get_current_user_id),
) -> dict:
    """
    Wipe the entire Qdrant vector collection and clear all in-memory state
    for the current user. This is a hard reset — all data is permanently removed.
    """
    # 1. Wipe the Qdrant vector store (entire collection, all users)
    qdrant_result = wipe_collection()

    # 2. Clear in-memory state for this user
    clear_state(user_id)
    clear_registry(user_id)

    return {
        "success": True,
        "qdrant": qdrant_result,
        "message": "Vector store wiped and in-memory state cleared.",
    }
