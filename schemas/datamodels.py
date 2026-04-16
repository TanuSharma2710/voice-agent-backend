from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ProcessingRequest(BaseModel):
    db_url: str = Field(..., description="Database connection URL to inspect.")
    user_table_description: str | None = Field(
        default=None,
        description="Optional user-provided description to enrich table metadata.",
    )
    include_schemas: list[str] | None = Field(
        default=None,
        description="Optional schema allow-list. For Postgres, default is ['public'].",
    )
    max_tables: int | None = Field(
        default=100,
        ge=1,
        description="Maximum number of tables to process in one request.",
    )


class ProcessingResponse(BaseModel):
    status: str
    tables_processed: int
    message: str


class ColumnMetadata(BaseModel):
    name: str
    type: str | None = None
    nullable: bool | None = None
    default: Any = None
    primary_key: bool = False
    autoincrement: bool | None = None


class ForeignKeyMetadata(BaseModel):
    constrained_columns: list[str] = Field(default_factory=list)
    referred_schema: str | None = None
    referred_table: str | None = None
    referred_columns: list[str] = Field(default_factory=list)


class TableMetadata(BaseModel):
    sub_database_id: str
    schema_name: str | None = None
    table_name: str
    columns: list[ColumnMetadata] = Field(default_factory=list)
    primary_key: list[str] = Field(default_factory=list)
    foreign_keys: list[ForeignKeyMetadata] = Field(default_factory=list)
    table_comment: str | None = None
    table_description: str | None = None


class ChunkPayload(BaseModel):
    chunk_id: str
    sub_database_id: str
    schema_name: str | None = None
    table_name: str
    table_description: str
    chunk_index: int = 0
    created_at: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
