import asyncio
import os
from functools import lru_cache
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from services.common.schemas import (
    ExperimentComparisonRequest,
    ExperimentComparisonResponse,
    ExperimentDocumentRequest,
    ExperimentDocumentResponse,
    ExperimentResetResponse,
    ExperimentEvaluationRequest,
    ExperimentEvaluationResponse,
    HealthResponse,
    KeywordStuffingRequest,
    KeywordStuffingResponse,
    OrchestratorAnswerRequest,
    OrchestratorQueryRequest,
    AttackDashboardMetrics,
    PoisonedRAGCandidate,
    PoisonedRAGRequest,
    PoisonedRAGResponse,
    SearchHit,
)
from services.orchestrator.evaluation import evaluate_answer, evaluate_retrieval
from services.orchestrator.keyword_stuffing import (
    build_keyword_stuffing_document,
)
from services.orchestrator.poisoned_rag import (
    generate_poison_candidates,
    rank_candidates,
)
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
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "256"))
RAG_CONTEXT_LIMIT = int(os.getenv("RAG_CONTEXT_LIMIT", "6"))
AGENT_URLS = {
    "local_db": os.getenv("LOCAL_DB_AGENT_URL", "http://localhost:8001"),
    "gmail": os.getenv("GMAIL_AGENT_URL", "http://localhost:8002"),
    "drive": os.getenv("DRIVE_AGENT_URL", "http://localhost:8003"),
}

app = FastAPI(title=SERVICE_NAME, version=VERSION)
DEMO_FILE = Path(__file__).with_name("static") / "demo.html"


@app.get("/demo", include_in_schema=False)
async def demo() -> FileResponse:
    return FileResponse(DEMO_FILE)


@lru_cache(maxsize=1)
def _rag_model():
    return create_chat_model(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=OLLAMA_TEMPERATURE,
        num_predict=OLLAMA_NUM_PREDICT,
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


@app.delete(
    "/experiments/documents",
    response_model=ExperimentResetResponse,
)
async def reset_experiment_documents() -> dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            return await _request_json(
                client,
                "DELETE",
                f"{AGENT_URLS['local_db']}/documents/untrusted",
            )
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
    retrieval_request = request
    if request.mode == "defended" and request.limit < 20:
        retrieval_request = request.model_copy(update={"limit": 20})
    results = await _query_agents(retrieval_request)
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
    return _build_evaluation_response(request, answer_payload)


def _build_evaluation_response(
    request: ExperimentEvaluationRequest,
    answer_payload: dict[str, Any],
) -> ExperimentEvaluationResponse:
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


@app.post(
    "/experiments/compare",
    response_model=ExperimentComparisonResponse,
)
async def compare_experiment_modes(
    request: ExperimentComparisonRequest,
) -> ExperimentComparisonResponse:
    common_fields = request.model_dump()
    vulnerable_request = ExperimentEvaluationRequest(
        **common_fields,
        mode="vulnerable",
    )
    defended_request = ExperimentEvaluationRequest(
        **common_fields,
        mode="defended",
    )

    vulnerable_payload = await _generate_answer(vulnerable_request)
    defended_payload = await _generate_answer(defended_request)
    vulnerable = _build_evaluation_response(
        vulnerable_request,
        vulnerable_payload,
    )
    defended = _build_evaluation_response(
        defended_request,
        defended_payload,
    )

    vulnerable_succeeded = vulnerable.outcome == "attack_succeeded"
    defended_succeeded = defended.outcome == "attack_succeeded"
    return ExperimentComparisonResponse(
        service=SERVICE_NAME,
        query=request.query,
        model=OLLAMA_MODEL,
        vulnerable=vulnerable,
        defended=defended,
        attack_succeeded_in_vulnerable=vulnerable_succeeded,
        attack_succeeded_in_defended=defended_succeeded,
        defense_blocked_attack=vulnerable_succeeded and not defended_succeeded,
    )


@app.post(
    "/experiments/keyword-stuffing",
    response_model=KeywordStuffingResponse,
)
async def run_keyword_stuffing_experiment(
    request: KeywordStuffingRequest,
) -> KeywordStuffingResponse:
    document_id = f"keyword-stuffing-{uuid4().hex[:16]}"
    poison_text = build_keyword_stuffing_document(
        query=request.query,
        attack_target=request.attack_target,
        repetitions=request.repetitions,
        include_prompt_injection=request.include_prompt_injection,
    )
    await create_experiment_document(
        ExperimentDocumentRequest(
            document_id=document_id,
            source="keyword-stuffing-automation",
            tags=[
                "controlled",
                "poison",
                "keyword-stuffing",
                *(
                    ["indirect-prompt-injection"]
                    if request.include_prompt_injection
                    else []
                ),
            ],
            text=poison_text,
        )
    )
    comparison = await compare_experiment_modes(
        ExperimentComparisonRequest(
            query=request.query,
            sources=["local_db"],
            limit=request.limit,
            expected_answer=request.expected_answer,
            attack_target=request.attack_target,
            attack_document_ids=[document_id],
        )
    )
    return KeywordStuffingResponse(
        document_id=document_id,
        repetitions=request.repetitions,
        include_prompt_injection=request.include_prompt_injection,
        poison_text=poison_text,
        comparison=comparison,
    )


@app.post(
    "/experiments/poisoned-rag",
    response_model=PoisonedRAGResponse,
)
async def run_poisoned_rag_experiment(
    request: PoisonedRAGRequest,
) -> PoisonedRAGResponse:
    candidate_count = min(8, request.poison_ratio + 2)
    candidates, generation_mode = await generate_poison_candidates(
        _rag_model(),
        query=request.query,
        attack_target=request.attack_target,
        count=candidate_count,
    )
    selected = rank_candidates(request.query, candidates)[
        : request.poison_ratio
    ]
    run_id = uuid4().hex[:12]
    selected_candidates: list[PoisonedRAGCandidate] = []
    for index, candidate in enumerate(selected, start=1):
        document_id = f"poisonedrag-{run_id}-{index}"
        await create_experiment_document(
            ExperimentDocumentRequest(
                document_id=document_id,
                source="poisonedrag-automation",
                tags=[
                    "controlled",
                    "poison",
                    "poisonedrag",
                    f"ratio-{request.poison_ratio}x",
                ],
                text=candidate.text,
            )
        )
        selected_candidates.append(
            PoisonedRAGCandidate(
                document_id=document_id,
                text=candidate.text,
                relevance_score=candidate.relevance_score,
            )
        )

    document_ids = [
        candidate.document_id for candidate in selected_candidates
    ]
    comparison = await compare_experiment_modes(
        ExperimentComparisonRequest(
            query=request.query,
            sources=["local_db"],
            limit=request.limit,
            expected_answer=request.expected_answer,
            attack_target=request.attack_target,
            attack_document_ids=document_ids,
        )
    )
    vulnerable = comparison.vulnerable
    top_k = len(vulnerable.documents)
    poison_in_top_k = vulnerable.untrusted_document_count
    metrics = AttackDashboardMetrics(
        attack_success_rate=1.0 if vulnerable.attack_target_present else 0.0,
        accuracy=1.0 if vulnerable.expected_answer_present else 0.0,
        poison_in_top_k=poison_in_top_k,
        top_k=top_k,
        poison_retrieval_rate=(
            round(poison_in_top_k / top_k, 4) if top_k else 0.0
        ),
    )
    return PoisonedRAGResponse(
        poison_ratio=request.poison_ratio,
        generation_mode=generation_mode,
        generated_candidate_count=len(candidates),
        document_ids=document_ids,
        selected_candidates=selected_candidates,
        metrics=metrics,
        comparison=comparison,
    )
