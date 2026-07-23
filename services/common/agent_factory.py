import os
from pathlib import Path

from fastapi import FastAPI

from services.common.schemas import HealthResponse, SearchRequest, SearchResponse
from services.common.search import JsonDocumentStore


def create_search_agent(
    *,
    service_name: str,
    default_data_file: Path,
) -> FastAPI:
    data_file = Path(os.getenv("DATA_FILE", str(default_data_file)))
    store = JsonDocumentStore(data_file)
    version = os.getenv("APP_VERSION", "0.1.0")

    app = FastAPI(title=service_name, version=version)

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(service=service_name, version=version)

    @app.post("/search", response_model=SearchResponse)
    async def search(request: SearchRequest) -> SearchResponse:
        return SearchResponse(
            service=service_name,
            query=request.query,
            hits=store.search(request.query, request.limit),
        )

    return app
