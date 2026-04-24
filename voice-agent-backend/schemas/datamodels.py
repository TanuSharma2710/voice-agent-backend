from __future__ import annotations

from pydantic import BaseModel, Field


class DatabaseRegisterRequest(BaseModel):
    nickname: str = Field(..., description="Nickname for the database.")
    db_url: str = Field(..., description="Database connection URL.")


class DatabaseRegisterResponse(BaseModel):
    database_id: str
    user_id: str
    nickname: str
    sub_database_id: str


class DatabaseListResponse(BaseModel):
    databases: list[dict[str, str]]


class DatabaseDeleteResponse(BaseModel):
    success: bool
    database_id: str


class DatabaseUpdateRequest(BaseModel):
    nickname: str = Field(..., description="New nickname for the database.")


class DatabaseUpdateResponse(BaseModel):
    success: bool
    database_id: str
    nickname: str
