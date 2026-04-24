from __future__ import annotations

import logging
import time
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from qdrant_client.http.exceptions import ResponseHandlingException

from config import QDRANT_COLLECTION
from vector_store.ingest import (
    DEFAULT_COLLECTION_NAME,
    _ensure_payload_index,
    _get_embeddings,
    _get_vector_client,
)

logger = logging.getLogger(__name__)

_QDRANT_CONNECT_ERRORS = (ResponseHandlingException, OSError, ConnectionError)


def _resolve_active_collection(
    client: QdrantClient,
    base_collection_name: str,
    vector_size: int,
) -> str | None:
    if client.collection_exists(base_collection_name):
        collection = client.get_collection(collection_name=base_collection_name)
        existing_size = collection.config.params.vectors.size
        if existing_size == vector_size:
            return base_collection_name

    fallback_name = f"{base_collection_name}_{vector_size}"
    if client.collection_exists(fallback_name):
        fallback = client.get_collection(collection_name=fallback_name)
        fallback_size = fallback.config.params.vectors.size
        if fallback_size == vector_size:
            return fallback_name

    return None


def _build_sub_database_filter(sub_database_ids: list[str] | None) -> rest.Filter | None:
    if not sub_database_ids:
        return None

    if len(sub_database_ids) == 1:
        return rest.Filter(
            must=[
                rest.FieldCondition(
                    key="sub_database_id",
                    match=rest.MatchValue(value=sub_database_ids[0]),
                )
            ]
        )

    return rest.Filter(
        should=[
            rest.FieldCondition(
                key="sub_database_id",
                match=rest.MatchValue(value=sub_id),
            )
            for sub_id in sub_database_ids
        ],
        min_should_match=1,
    )


def retrieve_schema_context(
    query_text: str,
    sub_database_ids: list[str] | None = None,
    limit: int = 10,
    _retries: int = 1,
) -> dict[str, Any]:
    """
    Retrieve schema chunks from Qdrant for a user query.

    Args:
        query_text: The user's natural language query
        sub_database_ids: List of user's sub_database_ids to filter by
        limit: Maximum number of chunks to retrieve
        _retries: Number of automatic retries on transient network errors

    Returns:
        Dict with chunks, context_text, and collection info.
        On connectivity failure returns {"chunks": [], "error": "qdrant_unreachable"}.
    """

    if not query_text.strip():
        return {"chunks": [], "context_text": "", "collection": None}

    for attempt in range(_retries + 1):
        try:
            client = _get_vector_client()
            query_embedding = _get_embeddings([query_text])[0]
            vector_size = len(query_embedding)
            base_collection = QDRANT_COLLECTION or DEFAULT_COLLECTION_NAME
            active_collection = _resolve_active_collection(client, base_collection, vector_size)

            if not active_collection:
                logger.warning(
                    "No Qdrant collection found for collection `%s` and vector size %s.",
                    base_collection,
                    vector_size,
                )
                return {"chunks": [], "context_text": "", "collection": None}

            # Guarantee the payload index exists on the active collection.
            # This is a no-op if the index was already created during ingest,
            # but fixes collections that pre-date the index requirement.
            _ensure_payload_index(client, active_collection)

            search_results = client.query_points(
                collection_name=active_collection,
                query=query_embedding,
                query_filter=_build_sub_database_filter(sub_database_ids),
                limit=limit,
                with_payload=True,
            ).points

            chunks: list[dict[str, Any]] = []
            context_blocks: list[str] = []

            for point in search_results:
                payload = point.payload or {}
                chunk = {
                    "score": point.score,
                    "chunk_id": payload.get("chunk_id"),
                    "sub_database_id": payload.get("sub_database_id"),
                    "schema_name": payload.get("schema_name"),
                    "table_name": payload.get("table_name"),
                    "table_description": payload.get("table_description"),
                    "text": payload.get("text", ""),
                    "schema": payload.get("schema", {}),
                }
                chunks.append(chunk)
                if chunk["text"]:
                    context_blocks.append(str(chunk["text"]))

            return {
                "chunks": chunks,
                "context_text": "\n\n".join(context_blocks),
                "collection": active_collection,
            }

        except _QDRANT_CONNECT_ERRORS as exc:
            if attempt < _retries:
                logger.warning(
                    "Qdrant connection error on attempt %d, retrying in 2s: %s",
                    attempt + 1, exc,
                )
                time.sleep(2)
            else:
                logger.error("Qdrant unreachable after %d attempt(s): %s", attempt + 1, exc)
                return {
                    "chunks": [],
                    "context_text": "",
                    "collection": None,
                    "error": "qdrant_unreachable",
                }


def retrieve_chunk_by_id(chunk_id: str) -> dict[str, Any] | None:
    """
    Retrieve a specific chunk by its ID.
    """
    client = _get_vector_client()
    collection_name = QDRANT_COLLECTION or DEFAULT_COLLECTION_NAME

    try:
        results = client.retrieve(
            collection_name=collection_name,
            ids=[chunk_id],
            with_payload=True,
        )
        if results:
            payload = results[0].payload or {}
            return {
                "chunk_id": payload.get("chunk_id"),
                "sub_database_id": payload.get("sub_database_id"),
                "schema_name": payload.get("schema_name"),
                "table_name": payload.get("table_name"),
                "table_description": payload.get("table_description"),
                "text": payload.get("text", ""),
                "schema": payload.get("schema", {}),
            }
    except Exception:
        pass

    return None


def summarize_tables_from_chunks(chunks: list[dict[str, Any]], max_tables: int = 12) -> str:
    """
    Build a compact table summary from retrieved schema chunks.
    """

    seen: set[str] = set()
    labels: list[str] = []

    for chunk in chunks:
        schema_name = chunk.get("schema_name")
        table_name = chunk.get("table_name")
        if not table_name:
            continue

        label = f"{schema_name}.{table_name}" if schema_name else str(table_name)
        if label in seen:
            continue

        seen.add(label)
        labels.append(label)
        if len(labels) >= max_tables:
            break

    if not labels:
        return ""

    return ", ".join(labels)


def delete_chunks_by_sub_database_id(sub_database_ids: list[str]) -> dict[str, Any]:
    """
    Delete all chunks from Qdrant that match the given sub_database_ids.
    Uses scroll + delete in batches to avoid payload size limits.
    """
    if not sub_database_ids:
        return {"status": "success", "deleted_count": 0}

    client = _get_vector_client()
    collection_name = QDRANT_COLLECTION or DEFAULT_COLLECTION_NAME

    if not client.collection_exists(collection_name):
        return {"status": "success", "deleted_count": 0, "note": "collection not found"}

    filter_cond = _build_sub_database_filter(sub_database_ids)
    total_deleted = 0
    batch_size = 100

    try:
        while True:
            results, _ = client.scroll(
                collection_name=collection_name,
                scroll_filter=filter_cond,
                limit=batch_size,
                with_payload=False,
            )
            if not results:
                break
            point_ids = [p.id for p in results]
            client.delete(
                collection_name=collection_name,
                points=point_ids,
                wait=True,
            )
            total_deleted += len(point_ids)
            if len(point_ids) < batch_size:
                break
    except Exception as exc:
        logger.warning("Error deleting chunks from Qdrant: %s", exc)
        return {"status": "partial", "deleted_count": total_deleted, "error": str(exc)}

    return {"status": "success", "deleted_count": total_deleted}


def wipe_collection() -> dict[str, Any]:
    """
    Delete and recreate the entire Qdrant collection.
    This is a hard reset — all vectors are permanently removed.
    """
    client = _get_vector_client()
    base_collection = QDRANT_COLLECTION or DEFAULT_COLLECTION_NAME
    deleted_collections: list[str] = []

    try:
        # Delete the base collection
        if client.collection_exists(base_collection):
            client.delete_collection(base_collection)
            deleted_collections.append(base_collection)

        # Also delete any dimension-specific fallback collections
        all_collections = client.get_collections().collections
        for col in all_collections:
            if col.name.startswith(f"{base_collection}_"):
                client.delete_collection(col.name)
                deleted_collections.append(col.name)

    except Exception as exc:
        logger.error("Error wiping Qdrant collections: %s", exc)
        return {"status": "error", "error": str(exc), "deleted_collections": deleted_collections}

    return {"status": "success", "deleted_collections": deleted_collections}