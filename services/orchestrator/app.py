import asyncio
import os
from typing import Any

import httpx
from fastapi import FastAPI

from services.common.schemas import HealthResponse, OrchestratorQueryRequest

SERVICE_NAME = "orchestrator"
VERSION = os.getenv("APP_VERSION", "0.1.0")
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "5"))
AGENT_URLS = {
    "local_db": os.getenv("LOCAL_DB_AGENT_URL", "http://localhost:8001"),
    "gmail": os.getenv("GMAIL_AGENT_URL", "http://localhost:8002"),
    "drive": os.getenv("DRIVE_AGENT_URL", "http://localhost:8003"),
}

app = FastAPI(title=SERVICE_NAME, version=VERSION)


async def _request_json(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    **kwargs: Any,
) -> dict[str, Any]:
    response = await client.request(method, url, **kwargs)
    response.raise_for_status()
    return response.json()


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(service=SERVICE_NAME, version=VERSION)


@app.get("/agents")
async def agents() -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        responses = await asyncio.gather(
            *[
                _request_json(client, "GET", f"{url}/health")
                for url in AGENT_URLS.values()
            ],
            return_exceptions=True,
        )

    status: dict[str, Any] = {}
    for source, response in zip(AGENT_URLS, responses, strict=True):
        if isinstance(response, Exception):
            status[source] = {"status": "unavailable", "error": str(response)}
        else:
            status[source] = response
    return {"service": SERVICE_NAME, "agents": status}


@app.post("/query")
async def query(request: OrchestratorQueryRequest) -> dict[str, Any]:
    selected_sources = list(dict.fromkeys(request.sources))
    payload = {"query": request.query, "limit": request.limit}

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        responses = await asyncio.gather(
            *[
                _request_json(
                    client,
                    "POST",
                    f"{AGENT_URLS[source]}/search",
                    json=payload,
                )
                for source in selected_sources
            ],
            return_exceptions=True,
        )

    results: dict[str, Any] = {}
    for source, response in zip(selected_sources, responses, strict=True):
        if isinstance(response, Exception):
            results[source] = {"status": "error", "error": str(response)}
        else:
            results[source] = {"status": "ok", **response}

    return {
        "service": SERVICE_NAME,
        "query": request.query,
        "results": results,
    }
