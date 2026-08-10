import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, status

from services.common.chroma_store import ChromaDocumentStore
from services.common.embeddings import create_embeddings
from services.common.ragpart import RagPartConfig
from services.common.schemas import HealthResponse, SearchRequest, SearchResponse
from services.common.search import JsonDocumentStore


def create_search_agent(
    *,
    service_name: str,
    default_data_file: Path,
    default_search_backend: str = "lexical",
) -> FastAPI:
    data_file = Path(os.getenv("DATA_FILE", str(default_data_file)))
    search_backend = os.getenv("SEARCH_BACKEND", default_search_backend).lower()
    if search_backend == "lexical":
        store = JsonDocumentStore(data_file)
    elif search_backend == "chroma":
        persist_value = os.getenv("CHROMA_PERSIST_DIRECTORY", "").strip()
        persist_directory = Path(persist_value) if persist_value else None
        store = ChromaDocumentStore(
            data_file,
            collection_name=os.getenv(
                "CHROMA_COLLECTION",
                service_name.replace("-agent", ""),
            ),
            persist_directory=persist_directory,
            embedding=create_embeddings(),
            ragpart=RagPartConfig(
                fragments=int(os.getenv("RAGPART_FRAGMENTS", "5")),
                combination_size=int(
                    os.getenv("RAGPART_COMBINATION_SIZE", "3")
                ),
                enabled=os.getenv("RAGPART_ENABLED", "false").lower()
                in {"1", "true", "yes"},
            ),
        )
    else:
        raise ValueError(f"Unsupported SEARCH_BACKEND: {search_backend}")

    version = os.getenv("APP_VERSION", "0.1.0")

    app = FastAPI(title=service_name, version=version)
    app.state.search_backend = search_backend
    app.state.document_store = store

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(service=service_name, version=version)

    @app.post("/search", response_model=SearchResponse)
    async def search(request: SearchRequest) -> SearchResponse:
        if request.defense == "ragpart":
            if not isinstance(store, ChromaDocumentStore):
                raise HTTPException(
                    status_code=status.HTTP_501_NOT_IMPLEMENTED,
                    detail="RAGPart requires the Chroma backend.",
                )
            hits = store.search_ragpart(request.query, request.limit)
        else:
            hits = store.search(request.query, request.limit)
        return SearchResponse(
            service=service_name,
            query=request.query,
            hits=hits,
        )

    return app
