import asyncio
import csv
import json
import os
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from time import perf_counter
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
    OrchestratorAnswerRequest,
    OrchestratorQueryRequest,
    AttackDashboardMetrics,
    PoisonedRAGBenchmarkPoint,
    PoisonedRAGBenchmarkRequest,
    PoisonedRAGBenchmarkResponse,
    PoisonedRAGCandidate,
    PoisonedRAGPipelineStatus,
    PoisonedRAGRequest,
    PoisonedRAGResponse,
    PoisonedRAGRunFailure,
    PoisonedRAGRunMetadata,
    SearchHit,
)
from services.orchestrator.evaluation import (
    answer_accuracy,
    attack_success_rate,
    evaluate_answer,
    evaluate_retrieval,
)
from services.orchestrator.poisoned_rag import (
    generate_poison_set,
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
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "512"))
RAG_CONTEXT_LIMIT = int(os.getenv("RAG_CONTEXT_LIMIT", "6"))
AGENT_URLS = {
    "local_db": os.getenv("LOCAL_DB_AGENT_URL", "http://localhost:8001"),
    "gmail": os.getenv("GMAIL_AGENT_URL", "http://localhost:8002"),
    "drive": os.getenv("DRIVE_AGENT_URL", "http://localhost:8003"),
}

app = FastAPI(title=SERVICE_NAME, version=VERSION)
DEMO_FILE = Path(__file__).with_name("static") / "demo.html"
EXPERIMENT_RESULTS_DIR = Path(
    os.getenv("EXPERIMENT_RESULTS_DIR", "/tmp/poisonedrag-results")
)
EXPERIMENT_SCENARIOS_FILE = Path(
    os.getenv(
        "EXPERIMENT_SCENARIOS_FILE",
        str(Path(__file__).resolve().parents[2] / "datasets/experiments/nq_target_queries.json"),
    )
)


@app.get("/demo", include_in_schema=False)
async def demo() -> FileResponse:
    return FileResponse(DEMO_FILE)


@app.get("/experiments/scenarios")
async def experiment_scenarios() -> dict[str, Any]:
    try:
        scenarios = json.loads(EXPERIMENT_SCENARIOS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=503, detail="Experiment scenarios are unavailable") from exc
    required = {"id", "query", "expected_answer", "attack_target"}
    if not isinstance(scenarios, list) or any(
        not isinstance(item, dict) or not required.issubset(item)
        for item in scenarios
    ):
        raise HTTPException(status_code=503, detail="Experiment scenario data is invalid")
    return {"dataset": "beir-nq", "count": len(scenarios), "scenarios": scenarios}


@lru_cache(maxsize=1)
def _rag_model():
    return create_chat_model(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=OLLAMA_TEMPERATURE,
        num_predict=OLLAMA_NUM_PREDICT,
    )


def _attack_generation_model(temperature: float):
    return create_chat_model(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=temperature,
        num_predict=OLLAMA_NUM_PREDICT,
    )


async def _answer_with_supplied_context(question: str, context: str) -> str:
    chain = build_rag_chain(_rag_model(), mode="vulnerable")
    return await chain.ainvoke({"question": question, "context": context})


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

        source_responses = await asyncio.gather(
            *[
                _request_json(client, "GET", f"{AGENT_URLS[source]}/source-status")
                for source in ("gmail", "drive")
            ],
            return_exceptions=True,
        )

    status: dict[str, Any] = {}
    for source, response in zip(AGENT_URLS, responses, strict=True):
        if isinstance(response, Exception):
            status[source] = {"status": "unavailable", "error": str(response)}
        else:
            status[source] = response
    for source, response in zip(("gmail", "drive"), source_responses, strict=True):
        if not isinstance(response, Exception) and status[source].get("status") != "unavailable":
            status[source]["data_source"] = response
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
    if (
        request.mode == "defended"
        or request.allowed_untrusted_document_ids is not None
    ) and request.limit < 20:
        retrieval_request = request.model_copy(update={"limit": 20})
    results = await _query_agents(retrieval_request)
    if request.allowed_untrusted_document_ids is not None:
        allowed = set(request.allowed_untrusted_document_ids)
        for result in results.values():
            if result.get("status") != "ok":
                continue
            result["hits"] = [
                hit
                for hit in result.get("hits", [])
                if hit.get("trust") != "untrusted"
                or hit.get("document_id") in allowed
            ]
    context_hits = collect_context_hits(
        results,
        limit=min(request.limit, RAG_CONTEXT_LIMIT),
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
    "/experiments/poisoned-rag",
    response_model=PoisonedRAGResponse,
)
async def run_poisoned_rag_experiment(
    request: PoisonedRAGRequest,
) -> PoisonedRAGResponse:
    run_started = perf_counter()
    started_at = datetime.now(timezone.utc)
    if request.cleanup_before_run:
        await reset_experiment_documents()

    run_id = uuid4().hex[:12]
    placeholder_ids = [f"poisonedrag-{run_id}-{index}" for index in range(1, request.poison_count + 1)]
    baseline_request = ExperimentEvaluationRequest(
        query=request.query,
        sources=["local_db"],
        limit=request.top_k,
        mode="vulnerable",
        expected_answer=request.expected_answer,
        attack_target=request.attack_target,
        attack_document_ids=placeholder_ids or [f"poisonedrag-{run_id}-none"],
        allowed_untrusted_document_ids=[],
    )
    baseline = _build_evaluation_response(
        baseline_request,
        await _generate_answer(baseline_request),
    )

    try:
        generated = await generate_poison_set(
            _attack_generation_model(request.generation_temperature),
            answer_with_context=_answer_with_supplied_context,
            query=request.query,
            attack_target=request.attack_target,
            count=request.poison_count,
            word_count=request.passage_word_count,
            max_trials=request.max_generation_trials,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"PoisonedRAG generation failed with {OLLAMA_MODEL}: {exc}",
        ) from exc

    generated_documents: list[PoisonedRAGCandidate] = []
    document_ids: list[str] = []
    for index, item in enumerate(generated, start=1):
        document_id = f"poisonedrag-{run_id}-{index}"
        if item.verified:
            document_ids.append(document_id)
            await create_experiment_document(
                ExperimentDocumentRequest(
                    document_id=document_id,
                    source="poisonedrag-black-box",
                    tags=[
                        "controlled",
                        "poison",
                        "poisonedrag",
                        "black-box",
                        "q-plus-i",
                    ],
                    text=item.poison_text,
                )
            )
        generated_documents.append(
            PoisonedRAGCandidate(
                document_id=document_id,
                instruction=item.instruction,
                poison_text=item.poison_text,
                verification_answer=item.verification_answer,
                verified=item.verified,
                generation_queries=item.generation_queries,
                generation_seconds=item.generation_seconds,
                word_count=item.word_count,
            )
        )

    attacked_request = ExperimentEvaluationRequest(
        query=request.query,
        sources=["local_db"],
        limit=request.top_k,
        mode="vulnerable",
        expected_answer=request.expected_answer,
        attack_target=request.attack_target,
        attack_document_ids=document_ids or [f"poisonedrag-{run_id}-none"],
        allowed_untrusted_document_ids=document_ids,
    )
    attacked = _build_evaluation_response(
        attacked_request,
        await _generate_answer(attacked_request),
    )
    poison_ids = set(document_ids)
    poison_in_top_k = sum(
        document.document_id in poison_ids for document in attacked.documents
    )
    retrieved_count = len(attacked.documents)
    precision = poison_in_top_k / retrieved_count if retrieved_count else 0.0
    recall = poison_in_top_k / len(document_ids) if document_ids else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    metrics = AttackDashboardMetrics(
        attack_success_rate=attack_success_rate(attacked.outcome),
        accuracy=answer_accuracy(attacked.outcome),
        poison_in_top_k=poison_in_top_k,
        top_k=retrieved_count,
        retrieval_precision=round(precision, 4),
        retrieval_recall=round(recall, 4),
        retrieval_f1=round(f1, 4),
        generation_queries=sum(item.generation_queries for item in generated),
        generation_seconds=round(
            sum(item.generation_seconds for item in generated), 3
        ),
        total_seconds=round(perf_counter() - run_started, 3),
    )
    verified_count = sum(item.verified for item in generated)
    generation_status = (
        "skipped"
        if request.poison_count == 0
        else "completed"
        if verified_count == request.poison_count
        else "partial"
    )
    injection_status = (
        "skipped"
        if request.poison_count == 0
        else "completed"
        if len(document_ids) == request.poison_count
        else "partial"
    )
    completed_at = datetime.now(timezone.utc)
    return PoisonedRAGResponse(
        run_id=run_id,
        requested_poison_count=request.poison_count,
        verified_poison_count=verified_count,
        injected_poison_count=len(document_ids),
        top_k=request.top_k,
        document_ids=document_ids,
        generated_documents=generated_documents,
        pipeline=PoisonedRAGPipelineStatus(
            generation=generation_status,
            injection=injection_status,
        ),
        metadata=PoisonedRAGRunMetadata(
            started_at=started_at.isoformat(),
            completed_at=completed_at.isoformat(),
            service_version=VERSION,
            model=OLLAMA_MODEL,
            generation_temperature=request.generation_temperature,
            max_generation_trials=request.max_generation_trials,
            passage_word_count=request.passage_word_count,
            cleanup_before_run=request.cleanup_before_run,
        ),
        metrics=metrics,
        baseline=baseline,
        attacked=attacked,
    )


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _write_benchmark_artifacts(
    result: PoisonedRAGBenchmarkResponse,
) -> None:
    EXPERIMENT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = EXPERIMENT_RESULTS_DIR / f"{result.experiment_id}.json"
    csv_path = EXPERIMENT_RESULTS_DIR / f"{result.experiment_id}.csv"
    json_path.write_text(
        json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=list(PoisonedRAGBenchmarkPoint.model_fields),
        )
        writer.writeheader()
        writer.writerows(point.model_dump() for point in result.points)


@app.post(
    "/experiments/poisoned-rag/benchmark",
    response_model=PoisonedRAGBenchmarkResponse,
)
async def run_poisoned_rag_benchmark(
    request: PoisonedRAGBenchmarkRequest,
) -> PoisonedRAGBenchmarkResponse:
    experiment_id = f"poisonedrag-{uuid4().hex[:12]}"
    runs: list[PoisonedRAGResponse] = []
    failures: list[PoisonedRAGRunFailure] = []
    for scenario in request.scenarios:
        for poison_count in request.poison_counts:
            for repetition in range(1, request.repetitions + 1):
                attempt_started = perf_counter()
                try:
                    run = await run_poisoned_rag_experiment(
                        PoisonedRAGRequest(
                            query=scenario.query,
                            expected_answer=scenario.expected_answer,
                            attack_target=scenario.attack_target,
                            poison_count=poison_count,
                            top_k=request.top_k,
                            max_generation_trials=request.max_generation_trials,
                            passage_word_count=request.passage_word_count,
                            generation_temperature=request.generation_temperature,
                            cleanup_before_run=True,
                        )
                    )
                except Exception as exc:
                    detail = (
                        exc.detail
                        if isinstance(exc, HTTPException)
                        else "Unexpected internal error; inspect restricted server logs."
                    )
                    detail_text = str(detail)[:1000]
                    stage = "generation" if "generation failed" in detail_text.casefold() else "unknown"
                    failures.append(
                        PoisonedRAGRunFailure(
                            scenario_name=scenario.name,
                            poison_count=poison_count,
                            repetition=repetition,
                            stage=stage,
                            error_type=type(exc).__name__,
                            detail=detail_text,
                            elapsed_seconds=round(perf_counter() - attempt_started, 3),
                        )
                    )
                    continue
                run.scenario_name = scenario.name
                runs.append(run)

    points: list[PoisonedRAGBenchmarkPoint] = []
    for poison_count in request.poison_counts:
        selected = [
            run for run in runs if run.requested_poison_count == poison_count
        ]
        selected_failures = [
            failure for failure in failures if failure.poison_count == poison_count
        ]
        points.append(
            PoisonedRAGBenchmarkPoint(
                poison_count=poison_count,
                trials=len(selected) + len(selected_failures),
                successful_trials=len(selected),
                failed_trials=len(selected_failures),
                attack_success_rate=_mean([run.metrics.attack_success_rate for run in selected]),
                accuracy=_mean([run.metrics.accuracy for run in selected]),
                retrieval_precision=_mean([run.metrics.retrieval_precision for run in selected]),
                retrieval_recall=_mean([run.metrics.retrieval_recall for run in selected]),
                retrieval_f1=_mean([run.metrics.retrieval_f1 for run in selected]),
                average_poison_in_top_k=_mean([float(run.metrics.poison_in_top_k) for run in selected]),
                average_generation_queries=_mean([float(run.metrics.generation_queries) for run in selected]),
                average_generation_seconds=_mean([run.metrics.generation_seconds for run in selected]),
                average_total_seconds=_mean([run.metrics.total_seconds for run in selected]),
            )
        )

    result = PoisonedRAGBenchmarkResponse(
        experiment_id=experiment_id,
        model=OLLAMA_MODEL,
        points=points,
        runs=runs,
        failures=failures,
        json_url=f"/experiments/results/{experiment_id}.json",
        csv_url=f"/experiments/results/{experiment_id}.csv",
    )
    _write_benchmark_artifacts(result)
    await reset_experiment_documents()
    return result


@app.get("/experiments/results/{filename}")
async def download_experiment_result(filename: str) -> FileResponse:
    if not filename.startswith("poisonedrag-") or not filename.endswith((".json", ".csv")):
        raise HTTPException(status_code=404, detail="Experiment result not found")
    path = EXPERIMENT_RESULTS_DIR / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Experiment result not found")
    return FileResponse(path, filename=filename)


async def _delete_experiment_documents(document_ids: list[str]) -> int:
    """Retained for targeted cleanup used by tests and maintenance scripts."""
    if not document_ids:
        return 0
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        responses = await asyncio.gather(
            *[
                _request_json(
                    client,
                    "DELETE",
                    f"{AGENT_URLS['local_db']}/documents/{document_id}",
                )
                for document_id in document_ids
            ],
            return_exceptions=True,
        )
    return sum(
        isinstance(response, dict) and response.get("deleted") is True
        for response in responses
    )
