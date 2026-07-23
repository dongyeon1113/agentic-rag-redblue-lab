import asyncio
import os
from functools import lru_cache
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException

from services.common.schemas import (
    ExperimentDocumentRequest,
    ExperimentDocumentResponse,
    ExperimentEvaluationRequest,
    ExperimentEvaluationResponse,
    HealthResponse,
    OrchestratorAnswerRequest,
    OrchestratorQueryRequest,
    SearchHit,
)
from services.orchestrator.evaluation import evaluate_answer, evaluate_retrieval
from services.orchestrator.rag import (
    build_rag_chain,
    collect_context_hits,
    create_chat_model,
    format_context,
)

SERVICE_NAME = "orchestrator"
VERSION = os.getenv("APP_VERSION", "0.1.0")
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "5"))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0"))
RAG_CONTEXT_LIMIT = int(os.getenv("RAG_CONTEXT_LIMIT", "6"))
AGENT_URLS = {
    "local_db": os.getenv("LOCAL_DB_AGENT_URL", "http://localhost:8001"),
    "gmail": os.getenv("GMAIL_AGENT_URL", "http://localhost:8002"),
    "drive": os.getenv("DRIVE_AGENT_URL", "http://localhost:8003"),
}

app = FastAPI(title=SERVICE_NAME, version=VERSION)


@lru_cache(maxsize=1)
def _rag_model():
    return create_chat_model(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=OLLAMA_TEMPERATURE,
    )


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


@app.get("/model")
async def model() -> dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            response = await _request_json(
                client,
                "GET",
                f"{OLLAMA_BASE_URL}/api/tags",
            )
    except Exception as exc:
        return {
            "status": "unavailable",
            "model": OLLAMA_MODEL,
            "base_url": OLLAMA_BASE_URL,
            "error": str(exc),
        }

    installed_models = [
        item.get("name", "")
        for item in response.get("models", [])
        if isinstance(item, dict)
    ]
    return {
        "status": "ok",
        "model": OLLAMA_MODEL,
        "base_url": OLLAMA_BASE_URL,
        "installed": OLLAMA_MODEL in installed_models,
        "available_models": installed_models,
    }


@app.post(
    "/experiments/documents",
    response_model=ExperimentDocumentResponse,
    status_code=201,
)
async def create_experiment_document(
    request: ExperimentDocumentRequest,
) -> dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            return await _request_json(
                client,
                "POST",
                f"{AGENT_URLS['local_db']}/documents",
                json=request.model_dump(),
            )
    except httpx.HTTPStatusError as exc:
        try:
            detail = exc.response.json().get("detail", exc.response.text)
        except ValueError:
            detail = exc.response.text
        raise HTTPException(
            status_code=exc.response.status_code,
            detail=detail,
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Local DB agent is unavailable: {exc}",
        ) from exc


async def _query_agents(
    request: OrchestratorQueryRequest,
) -> dict[str, Any]:
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
    return results


@app.post("/query")
async def query(request: OrchestratorQueryRequest) -> dict[str, Any]:
    results = await _query_agents(request)

    return {
        "service": SERVICE_NAME,
        "query": request.query,
        "results": results,
    }


@app.post("/answer")
async def answer(request: OrchestratorAnswerRequest) -> dict[str, Any]:
    return await _generate_answer(request)


async def _generate_answer(
    request: OrchestratorAnswerRequest,
) -> dict[str, Any]:
    results = await _query_agents(request)
    context_hits = collect_context_hits(
        results,
        limit=RAG_CONTEXT_LIMIT,
        trusted_only=request.mode == "defended",
    )
    if not context_hits:
        return {
            "service": SERVICE_NAME,
            "query": request.query,
            "model": OLLAMA_MODEL,
            "mode": request.mode,
            "answer": "I cannot determine the answer from the retrieved context.",
            "documents": [],
            "results": results,
        }

    chain = build_rag_chain(_rag_model(), mode=request.mode)
    try:
        generated_answer = await chain.ainvoke(
            {
                "question": request.query,
                "context": format_context(context_hits),
            }
        )
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Ollama model {OLLAMA_MODEL} is unavailable: {exc}",
        ) from exc

    return {
        "service": SERVICE_NAME,
        "query": request.query,
        "model": OLLAMA_MODEL,
        "mode": request.mode,
        "answer": generated_answer,
        "documents": [hit.model_dump() for hit in context_hits],
        "results": results,
    }


@app.post(
    "/experiments/evaluate",
    response_model=ExperimentEvaluationResponse,
)
async def evaluate_experiment(
    request: ExperimentEvaluationRequest,
) -> ExperimentEvaluationResponse:
    answer_payload = await _generate_answer(request)
    documents = [
        SearchHit.model_validate(document)
        for document in answer_payload["documents"]
    ]
    outcome, expected_present, target_present = evaluate_answer(
        answer_payload["answer"],
        expected_answer=request.expected_answer,
        attack_target=request.attack_target,
    )
    (
        attack_retrieved,
        attack_rank,
        attack_score,
        untrusted_count,
    ) = evaluate_retrieval(
        documents,
        attack_document_ids=request.attack_document_ids,
    )

    return ExperimentEvaluationResponse(
        service=SERVICE_NAME,
        query=request.query,
        model=OLLAMA_MODEL,
        mode=request.mode,
        answer=answer_payload["answer"],
        outcome=outcome,
        expected_answer=request.expected_answer,
        attack_target=request.attack_target,
        expected_answer_present=expected_present,
        attack_target_present=target_present,
        attack_document_retrieved=attack_retrieved,
        attack_document_rank=attack_rank,
        attack_document_score=attack_score,
        untrusted_document_count=untrusted_count,
        documents=documents,
    )
