from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_COLLECTION_NAME = "voice_agent_metadata"
DEFAULT_EMBEDDING_DIMENSION = 768
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-2-preview"

# Collections for which the sub_database_id payload index has already been
# created in this process.  Avoids a redundant PUT on every retrieval call.
_indexed_collections: set[str] = set()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _chunk_columns(columns: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [columns[index : index + size] for index in range(0, len(columns), size)]


def _format_chunk_text(record: dict[str, Any]) -> str:
    schema = record.get("schema", {})
    columns = schema.get("columns", []) if isinstance(schema, dict) else []
    column_lines = []
    for column in columns:
        if not isinstance(column, dict):
            continue
        column_name = column.get("name", "")
        column_type = column.get("type", "")
        nullable = column.get("nullable")
        column_lines.append(
            f"- {column_name} | type={column_type} | nullable={nullable}"
        )

    foreign_keys = schema.get("foreign_keys", []) if isinstance(schema, dict) else []
    fk_lines = []
    for fk in foreign_keys:
        if not isinstance(fk, dict):
            continue
        fk_lines.append(
            "- "
            + ", ".join(
                [
                    f"columns={fk.get('constrained_columns', [])}",
                    f"ref={fk.get('referred_schema')}.{fk.get('referred_table')}",
                    f"ref_columns={fk.get('referred_columns', [])}",
                ]
            )
        )

    lines = [
        f"sub_database_id: {record.get('sub_database_id')}",
        f"schema_name: {record.get('schema_name')}",
        f"table_name: {record.get('table_name')}",
        f"table_description: {record.get('table_description')}",
    ]
    if column_lines:
        lines.append("columns:")
        lines.extend(column_lines)
    if fk_lines:
        lines.append("foreign_keys:")
        lines.extend(fk_lines)

    return "\n".join(lines)


def _format_document_title(record: dict[str, Any]) -> str:
    schema_name = record.get("schema_name")
    table_name = record.get("table_name", "none")
    if schema_name:
        return f"{schema_name}.{table_name}"
    return str(table_name or "none")


def _format_document_text(record: dict[str, Any]) -> str:
    title = _format_document_title(record)
    content = record.get("text", "")
    return f"title: {title} | text: {content}"


def create_chunks(extracted_tables: list[dict[str, Any]], max_columns_per_chunk: int = 12) -> list[dict[str, Any]]:
    """
    Convert extracted table metadata into embedding-ready chunks.

    Each chunk keeps the table name, schema, table description, sub-database id,
    and timestamp so the vector store can later filter and retrieve metadata.
    """

    chunks: list[dict[str, Any]] = []

    for table in extracted_tables:
        schema = table.get("schema", {})
        columns = schema.get("columns", []) if isinstance(schema, dict) else []

        if columns:
            column_groups = _chunk_columns(columns, max_columns_per_chunk)
        else:
            column_groups = [[]]

        for index, column_group in enumerate(column_groups):
            chunk_schema = dict(schema) if isinstance(schema, dict) else {"raw_schema": schema}
            if column_group:
                chunk_schema["columns"] = column_group

            chunk_payload = {
                "chunk_id": str(uuid.uuid4()),
                "sub_database_id": table.get("sub_database_id"),
                "schema_name": table.get("schema_name"),
                "table_name": table.get("table_name"),
                "table_description": table.get("table_description", ""),
                "chunk_index": index,
                "created_at": _now_iso(),
                "schema": chunk_schema,
            }
            chunk_payload["text"] = _format_chunk_text(chunk_payload)
            chunk_payload["metadata"] = {
                "sub_database_id": table.get("sub_database_id"),
                "schema_name": table.get("schema_name"),
                "table_name": table.get("table_name"),
                "chunk_index": index,
                "created_at": chunk_payload["created_at"],
            }
            chunks.append(chunk_payload)

    return chunks


def _normalize_embedding(values: list[float]) -> list[float]:
    magnitude = sum(value * value for value in values) ** 0.5
    if magnitude == 0:
        return values
    return [value / magnitude for value in values]


def _get_embeddings(texts: list[str]) -> list[list[float]]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing. Gemini embeddings cannot be generated.")

    model_name = os.getenv("GEMINI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    output_dimensionality = int(
        os.getenv("GEMINI_EMBEDDING_DIMENSION", str(DEFAULT_EMBEDDING_DIMENSION))
    )

    client = genai.Client(api_key=api_key)
    if model_name == "gemini-embedding-2-preview":
        document_texts = texts
        result = client.models.embed_content(
            model=model_name,
            contents=document_texts,
            config=types.EmbedContentConfig(output_dimensionality=output_dimensionality),
        )
    else:
        result = client.models.embed_content(
            model=model_name,
            contents=texts,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=output_dimensionality,
            ),
        )

    embeddings: list[list[float]] = []
    for embedding in result.embeddings:
        embeddings.append(_normalize_embedding(list(embedding.values)))

    return embeddings


def _get_vector_client() -> QdrantClient:
    endpoint = os.getenv("QDRANT_ENDPOINT")
    api_key = os.getenv("QDRANT_API_KEY")

    if not endpoint or not api_key:
        raise ValueError(
            "QDRANT_ENDPOINT or QDRANT_API_KEY is missing. Cloud Qdrant must be configured."
        )

    return QdrantClient(url=endpoint, api_key=api_key, check_compatibility=False)


def _ensure_payload_index(client: QdrantClient, collection_name: str) -> None:
    """Create a keyword payload index on sub_database_id if it doesn't exist.

    query_points (qdrant-client >= 1.7) requires an index for any field used
    in a filter.  Creating an index that already exists is a no-op.
    Skipped when the collection is already tracked in _indexed_collections.
    """
    if collection_name in _indexed_collections:
        return
    try:
        client.create_payload_index(
            collection_name=collection_name,
            field_name="sub_database_id",
            field_schema=rest.PayloadSchemaType.KEYWORD,
            wait=True,
        )
        _indexed_collections.add(collection_name)
    except Exception as exc:
        # Index already exists or other non-fatal error — log and continue.
        logger.debug("Payload index on sub_database_id: %s", exc)
        _indexed_collections.add(collection_name)  # don't retry on next call


def _ensure_collection(client: QdrantClient, collection_name: str, vector_size: int) -> str:
    if client.collection_exists(collection_name):
        collection = client.get_collection(collection_name=collection_name)
        existing_size = collection.config.params.vectors.size
        if existing_size != vector_size:
            # Avoid hard-failing when an older collection exists with a different
            # embedding size. Route writes to a dimension-specific collection.
            fallback_collection = f"{collection_name}_{vector_size}"
            logger.warning(
                "Collection `%s` has vector size %s, but current embeddings use %s. "
                "Using fallback collection `%s`.",
                collection_name,
                existing_size,
                vector_size,
                fallback_collection,
            )
            if client.collection_exists(fallback_collection):
                fallback = client.get_collection(collection_name=fallback_collection)
                fallback_size = fallback.config.params.vectors.size
                if fallback_size != vector_size:
                    raise ValueError(
                        f"Fallback collection `{fallback_collection}` expects vectors of size "
                        f"{fallback_size}, but current embeddings use {vector_size}."
                    )
            else:
                client.create_collection(
                    collection_name=fallback_collection,
                    vectors_config=rest.VectorParams(
                        size=vector_size,
                        distance=rest.Distance.COSINE,
                    ),
                )
            _ensure_payload_index(client, fallback_collection)
            return fallback_collection
        _ensure_payload_index(client, collection_name)
        return collection_name

    client.create_collection(
        collection_name=collection_name,
        vectors_config=rest.VectorParams(
            size=vector_size,
            distance=rest.Distance.COSINE,
        ),
    )
    _ensure_payload_index(client, collection_name)
    return collection_name


def embed_and_store(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Embed chunks and persist them to Qdrant.
    """

    if not chunks:
        return {"status": "success", "stored_chunks": 0}

    client = _get_vector_client()
    collection_name = os.getenv("QDRANT_COLLECTION", DEFAULT_COLLECTION_NAME)
    texts = [_format_document_text(chunk) for chunk in chunks]
    embeddings = _get_embeddings(texts)
    vector_size = len(embeddings[0]) if embeddings else int(
        os.getenv("GEMINI_EMBEDDING_DIMENSION", str(DEFAULT_EMBEDDING_DIMENSION))
    )
    active_collection = _ensure_collection(client, collection_name, vector_size)

    points = []
    for index, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        payload = dict(chunk)
        payload["text"] = texts[index]
        payload["embedding_model"] = os.getenv("GEMINI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        points.append(
            rest.PointStruct(
                id=chunk["chunk_id"],
                vector=embedding,
                payload=payload,
            )
        )

    client.upsert(collection_name=active_collection, points=points, wait=True)
    return {"status": "success", "stored_chunks": len(points), "collection": active_collection}
