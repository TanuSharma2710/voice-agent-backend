from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from schemas.datamodels import ProcessingRequest, ProcessingResponse
from services.processing import urlprocessor

router = APIRouter(prefix="/api", tags=["processing"])


def _map_error_status(message: str) -> tuple[int, str]:
    """
    Map service-layer errors to appropriate HTTP statuses.

    FastAPI guidance: use 4xx for client errors and 5xx for server/provider errors.
    """
    if message.startswith("Unable to create a database engine"):
        return status.HTTP_400_BAD_REQUEST, "INVALID_DATABASE_URL"
    if message.startswith("Database metadata extraction failed"):
        return status.HTTP_400_BAD_REQUEST, "METADATA_EXTRACTION_FAILED"
    if message.startswith("Table enrichment failed"):
        return status.HTTP_503_SERVICE_UNAVAILABLE, "LLM_ENRICHMENT_FAILED"
    if message.startswith("Vector storage failed"):
        return status.HTTP_502_BAD_GATEWAY, "VECTOR_STORAGE_FAILED"
    return status.HTTP_500_INTERNAL_SERVER_ERROR, "PROCESSING_FAILED"


@router.post("/process/database-metadata", response_model=ProcessingResponse)
def process_database_metadata(payload: ProcessingRequest) -> ProcessingResponse:
    result = urlprocessor(
        db_url=payload.db_url,
        user_table_description=payload.user_table_description,
        include_schemas=payload.include_schemas,
        max_tables=payload.max_tables,
    )

    if result.get("status") != "success":
        message = result.get("message", "Processing failed.")
        status_code, error_code = _map_error_status(message)
        raise HTTPException(
            status_code=status_code,
            detail={
                "code": error_code,
                "message": message,
                "tables_processed": result.get("tables_processed", 0),
            },
        )

    return ProcessingResponse(**result)
