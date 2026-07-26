from pathlib import Path

from fastapi import HTTPException, status

from services.common.agent_factory import create_search_agent
from services.common.chroma_store import ChromaDocumentStore
from services.common.schemas import (
    ExperimentDeleteResponse,
    ExperimentDocumentRequest,
    ExperimentDocumentResponse,
    ExperimentResetResponse,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

app = create_search_agent(
    service_name="local-db-agent",
    default_data_file=PROJECT_ROOT / "datasets/sample/nq_sample.json",
    default_search_backend="chroma",
)


@app.post(
    "/documents",
    response_model=ExperimentDocumentResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_experiment_document(
    request: ExperimentDocumentRequest,
) -> ExperimentDocumentResponse:
    store = app.state.document_store
    if not isinstance(store, ChromaDocumentStore):
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Experiment document writes require the Chroma backend.",
        )

    try:
        store.add_document(
            document_id=request.document_id,
            source=request.source,
            trust="untrusted",
            tags=request.tags,
            text=request.text,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    return ExperimentDocumentResponse(
        document_id=request.document_id,
        source=request.source,
        tags=request.tags,
        document_count=store.count(),
    )


@app.delete(
    "/documents/untrusted",
    response_model=ExperimentResetResponse,
)
async def delete_untrusted_documents() -> ExperimentResetResponse:
    store = app.state.document_store
    if not isinstance(store, ChromaDocumentStore):
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Experiment document deletion requires the Chroma backend.",
        )

    deleted_count = store.delete_untrusted_documents()
    return ExperimentResetResponse(
        deleted_count=deleted_count,
        document_count=store.count(),
    )


@app.delete(
    "/documents/{document_id}",
    response_model=ExperimentDeleteResponse,
)
async def delete_untrusted_document(
    document_id: str,
) -> ExperimentDeleteResponse:
    store = app.state.document_store
    if not isinstance(store, ChromaDocumentStore):
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Experiment document deletion requires the Chroma backend.",
        )

    try:
        deleted = store.delete_untrusted_document(document_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(exc),
        ) from exc
    return ExperimentDeleteResponse(
        document_id=document_id,
        deleted=deleted,
        document_count=store.count(),
    )
