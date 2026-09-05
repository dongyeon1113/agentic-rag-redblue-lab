from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from agentdojo.attacks.attack_registry import ATTACKS
from agentdojo.task_suite.load_suites import get_suites
from agentdojo_benchmark.runner import (
    DEFENSE_COMPONENTS,
    DEFENSE_PROFILES,
    BenchmarkOptions,
    defense_config,
    run_benchmark,
)

DEFAULT_VERSION = "v1.2.2"
STATIC = Path(__file__).with_name("static")
JOB_TASKS: dict[str, asyncio.Task[Any]] = {}
JOB_CANCEL_EVENTS: dict[str, asyncio.Event] = {}
JOB_EXECUTION_LOCK = asyncio.Lock()


class RunRequest(BaseModel):
    suite: str
    user_task: str | None = None
    injection_task: str | None = None
    benign_only: bool = False
    baseline_success_only: bool = False
    baseline_source_job: str | None = None
    attack: str = "important_instructions"
    benchmark_version: str = DEFAULT_VERSION
    force_rerun: bool = True
    defense_profiles: list[str] = Field(default_factory=lambda: ["baseline"])


def _output_dir() -> Path:
    return Path(os.getenv("AGENTDOJO_OUTPUT_DIR", "runs/agentdojo")).resolve()


def _job_root() -> Path:
    return _output_dir() / "jobs"


def _job_dir(job_id: str) -> Path:
    return _job_root() / job_id


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _job_state(job_id: str) -> dict[str, Any]:
    state = _read_json(_job_dir(job_id) / "state.json")
    if not isinstance(state, dict):
        raise HTTPException(status_code=404, detail="Run not found")
    return state


def _update_job(job_id: str, **changes: Any) -> dict[str, Any]:
    state = _job_state(job_id)
    state.update(changes)
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    _write_json_atomic(_job_dir(job_id) / "state.json", state)
    return state


def _reconcile_job(job_id: str) -> dict[str, Any]:
    state = _job_state(job_id)
    if state.get("status") in {"running", "cancelling"} and job_id not in JOB_TASKS:
        state = _update_job(
            job_id,
            status="interrupted",
            current=None,
            error="Server task is no longer active; resume from the saved checkpoint.",
        )
    return state


def _text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if not isinstance(content, list):
        return ""
    return "\n".join(
        str(block.get("content", ""))
        for block in content
        if isinstance(block, dict)
    )


def _trace_view(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if payload is None:
        return None
    messages = payload.get("messages", [])
    user_message = next(
        (item for item in messages if item.get("role") == "user"), {}
    )
    assistant_messages = [
        item for item in messages if item.get("role") == "assistant"
    ]
    current_agent = payload.get("current_agent")
    if not isinstance(current_agent, dict):
        current_agent = {}
    calls: list[dict[str, Any]] = []
    for message in assistant_messages:
        for call in message.get("tool_calls") or []:
            calls.append({
                "sequence": len(calls) + 1,
                "function": call.get("function", ""),
                "args": call.get("args", {}),
            })
    tool_results: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") != "tool":
            continue
        call = message.get("tool_call") or {}
        tool_results.append({
            "sequence": len(tool_results) + 1,
            "function": call.get("function", ""),
            "output": _text(message),
            "error": message.get("error"),
        })
    return {
        "query": _text(user_message),
        "answer": "\n".join(_text(item) for item in assistant_messages).strip(),
        "tool_calls": calls,
        "tool_results": tool_results,
        "defense_profile": current_agent.get("defense_profile"),
        "defense_config": current_agent.get("defense_config", {}),
        "defense_report": current_agent.get("defense_report", {}),
        "utility": payload.get("utility"),
        "attack_succeeded": (
            payload.get("security")
            if payload.get("injection_task_id") is not None
            else None
        ),
        "duration_seconds": payload.get("duration"),
        "error": payload.get("error"),
        "timestamp": payload.get("evaluation_timestamp"),
    }


def _load_traces() -> list[tuple[Path, dict[str, Any]]]:
    root = _output_dir()
    if not root.exists():
        return []
    traces = []
    for path in root.rglob("*.json"):
        if path.name == "current-agent-summary.json":
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and "suite_name" in payload:
            traces.append((path, payload))
    return traces


def _case_key(payload: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(payload.get("pipeline_name", "")),
        str(payload.get("suite_name", "")),
        str(payload.get("user_task_id", "")),
        str(payload.get("attack_type") or "none"),
        str(payload.get("injection_task_id") or "none"),
    )


def _baseline_success_tasks(
    job_id: str,
    version: str,
) -> dict[str, tuple[str, ...]]:
    if not job_id.startswith("agentdojo-") or "/" in job_id or "\\" in job_id:
        raise HTTPException(status_code=422, detail="Invalid baseline source job")
    source_request = _read_json(_job_dir(job_id) / "request.json", {})
    if source_request.get("benchmark_version") != version:
        raise HTTPException(
            status_code=422,
            detail="Baseline source benchmark version does not match this run",
        )
    checkpoint = _read_json(_job_dir(job_id) / "checkpoint.json", {})
    baseline = checkpoint.get("reports", {}).get("baseline", {})
    suites_report = (
        baseline.get("profiles", {})
        .get("baseline", {})
        .get("suites", {})
    )
    if not suites_report:
        raise HTTPException(
            status_code=422,
            detail="Selected job has no completed baseline profile",
        )

    catalog_suites = get_suites(version)
    selected: dict[str, tuple[str, ...]] = {}
    for suite_name, suite_report in suites_report.items():
        if suite_name not in catalog_suites:
            continue
        utility_results = (
            suite_report.get("benign", {}).get("utility_results", {})
        )
        successful = {
            key.split("::", 1)[0]
            for key, passed in utility_results.items()
            if passed is True
        }
        ordered = tuple(
            task_id
            for task_id in catalog_suites[suite_name].user_tasks
            if task_id in successful
        )
        if ordered:
            selected[suite_name] = ordered
    if not selected:
        raise HTTPException(
            status_code=422,
            detail="Baseline source has no successful benign UserTasks",
        )
    return selected


def _eligible_task_snapshot(job_id: str) -> Path:
    return _job_dir(job_id) / "eligible-user-tasks.json"


def _catalog_task(version: str, suite_name: str, task_id: str, kind: str) -> Any:
    try:
        suite = get_suites(version)[suite_name]
        if kind == "user":
            return suite.get_user_task_by_id(task_id)
        return suite.get_injection_task_by_id(task_id)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=f"Unknown {kind} task") from exc


app = FastAPI(title="AgentDojo Current Agent Trace GUI")


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "healthy", "output_dir": str(_output_dir())}


@app.get("/api/catalog")
async def catalog(version: str = DEFAULT_VERSION) -> dict[str, Any]:
    try:
        suites = get_suites(version)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Unknown benchmark version") from exc
    return {
        "version": version,
        "attacks": sorted(name for name in ATTACKS if name != "manual"),
        "defense_profiles": list(DEFENSE_PROFILES),
        "defense_components": list(DEFENSE_COMPONENTS),
        "suites": {
            name: {
                "user_tasks": [
                    {"id": task_id, "prompt": task.PROMPT}
                    for task_id, task in suite.user_tasks.items()
                ],
                "injection_tasks": [
                    {"id": task_id, "goal": task.GOAL}
                    for task_id, task in suite.injection_tasks.items()
                ],
            }
            for name, suite in suites.items()
        },
    }


@app.get("/api/baseline-success-sources")
async def baseline_success_sources(
    version: str = DEFAULT_VERSION,
) -> dict[str, Any]:
    sources: list[dict[str, Any]] = []
    if _job_root().is_dir():
        for directory in _job_root().iterdir():
            if not directory.is_dir():
                continue
            try:
                selected = _baseline_success_tasks(directory.name, version)
            except HTTPException:
                continue
            state = _read_json(directory / "state.json", {})
            sources.append({
                "job_id": directory.name,
                "status": state.get("status"),
                "created_at": state.get("created_at"),
                "suites": {
                    suite: list(task_ids)
                    for suite, task_ids in selected.items()
                },
                "total": sum(len(task_ids) for task_ids in selected.values()),
            })
    sources.sort(key=lambda item: item.get("created_at") or "", reverse=True)
    return {"sources": sources}


@app.get("/api/cases")
async def cases() -> list[dict[str, Any]]:
    rows = []
    for path, payload in _load_traces():
        if payload.get("injection_task_id") is None:
            continue
        pipeline, suite, user_task, attack, injection_task = _case_key(payload)
        rows.append({
            "pipeline": pipeline,
            "suite": suite,
            "user_task": user_task,
            "injection_task": injection_task,
            "attack": attack,
            "utility": payload.get("utility"),
            "attack_succeeded": payload.get("security"),
            "duration_seconds": payload.get("duration"),
            "timestamp": payload.get("evaluation_timestamp"),
            "path": str(path.relative_to(_output_dir())),
        })
    return sorted(rows, key=lambda item: item["timestamp"] or "", reverse=True)


@app.get("/api/cases/detail")
async def case_detail(
    pipeline: str = Query(min_length=1),
    suite: str = Query(min_length=1),
    user_task: str = Query(min_length=1),
    injection_task: str = Query(min_length=1),
    attack: str = Query(min_length=1),
    version: str = DEFAULT_VERSION,
) -> dict[str, Any]:
    benign = None
    attacked = None
    for _, payload in _load_traces():
        key = _case_key(payload)
        if key == (pipeline, suite, user_task, attack, injection_task):
            attacked = payload
        elif key == (pipeline, suite, user_task, "none", "none"):
            benign = payload
    if attacked is None:
        raise HTTPException(status_code=404, detail="Attack trace not found")
    user = _catalog_task(version, suite, user_task, "user")
    injection = _catalog_task(version, suite, injection_task, "injection")
    return {
        "identity": {
            "pipeline": pipeline,
            "suite": suite,
            "user_task": user_task,
            "injection_task": injection_task,
            "attack": attack,
            "benchmark_version": version,
        },
        "user_prompt": user.PROMPT,
        "attack_goal": injection.GOAL,
        "injections": attacked.get("injections", {}),
        "benign": _trace_view(benign),
        "attacked": _trace_view(attacked),
        "verdict": {
            "normal_task_changed": (
                benign is not None
                and benign.get("utility") != attacked.get("utility")
            ),
            "attack_succeeded": attacked.get("security"),
        },
    }


async def _execute_job(job_id: str, request: RunRequest) -> None:
    checkpoint_path = _job_dir(job_id) / "checkpoint.json"
    checkpoint = _read_json(checkpoint_path, {"completed_profiles": [], "reports": {}})
    completed = set(checkpoint.get("completed_profiles", []))
    cancel_event = JOB_CANCEL_EVENTS[job_id]
    user_tasks_by_suite: dict[str, tuple[str, ...]] | None = None
    if request.baseline_success_only:
        snapshot = _read_json(_eligible_task_snapshot(job_id), {})
        if not isinstance(snapshot, dict) or not snapshot:
            raise RuntimeError("Baseline-success UserTask snapshot is missing")
        user_tasks_by_suite = {
            suite: tuple(str(task_id) for task_id in task_ids)
            for suite, task_ids in snapshot.items()
            if isinstance(task_ids, list) and task_ids
        }
    _update_job(job_id, status="running", error=None)
    try:
        for profile in request.defense_profiles:
            if profile in completed:
                continue
            if cancel_event.is_set():
                _update_job(job_id, status="cancelled", current=None)
                return
            _update_job(job_id, current=profile)
            async with JOB_EXECUTION_LOCK:
                report = await asyncio.to_thread(
                    run_benchmark,
                    BenchmarkOptions(
                        suites=(
                            ("workspace", "slack", "travel", "banking")
                            if request.suite == "all"
                            else (request.suite,)
                        ),
                        attack=request.attack,
                        benchmark_version=request.benchmark_version,
                        user_tasks=((request.user_task,) if request.user_task else ()),
                        user_tasks_by_suite=user_tasks_by_suite,
                        injection_tasks=(
                            (request.injection_task,)
                            if request.injection_task and not request.benign_only
                            else ()
                        ),
                        output_dir=_job_dir(job_id) / "results",
                        # Every job has a unique result directory. Reusing valid
                        # traces lets a failed profile resume at task granularity.
                        force_rerun=False,
                        include_benign=True,
                        include_attack=not request.benign_only,
                        defense_profiles=(profile,),
                    ),
                )
            checkpoint["reports"][profile] = report
            completed.add(profile)
            checkpoint["completed_profiles"] = sorted(completed)
            _write_json_atomic(checkpoint_path, checkpoint)
            _update_job(job_id, completed=len(completed))
        final_report = {
            "job_id": job_id,
            "request": request.model_dump(mode="json"),
            "profiles": checkpoint["reports"],
        }
        _write_json_atomic(_job_dir(job_id) / "result.json", final_report)
        _update_job(job_id, status="completed", current=None, report=final_report)
    except Exception as exc:  # pragma: no cover - provider/runtime failures
        _update_job(
            job_id, status="failed", current=None,
            error=f"{type(exc).__name__}: {exc}",
        )


def _finish_job(job_id: str, task: asyncio.Task[Any]) -> None:
    JOB_TASKS.pop(job_id, None)
    JOB_CANCEL_EVENTS.pop(job_id, None)


def _schedule_job(job_id: str, request: RunRequest) -> None:
    JOB_CANCEL_EVENTS[job_id] = asyncio.Event()
    task = asyncio.create_task(_execute_job(job_id, request))
    JOB_TASKS[job_id] = task
    task.add_done_callback(lambda finished: _finish_job(job_id, finished))


@app.post("/api/runs", status_code=202)
async def start_run(request: RunRequest) -> dict[str, Any]:
    if request.benign_only and request.baseline_success_only:
        raise HTTPException(
            status_code=422,
            detail="Baseline-success defense mode already includes benign evaluation",
        )
    if request.baseline_success_only and not request.baseline_source_job:
        raise HTTPException(
            status_code=422,
            detail="A completed baseline source job is required",
        )
    if request.baseline_success_only and request.user_task:
        raise HTTPException(
            status_code=422,
            detail="A single UserTask cannot be combined with baseline-success mode",
        )
    if request.baseline_source_job and not request.baseline_success_only:
        raise HTTPException(
            status_code=422,
            detail="Baseline source job requires baseline-success mode",
        )
    if request.benign_only and request.injection_task:
        raise HTTPException(
            status_code=422,
            detail="Injection task cannot be used in utility-only mode",
        )
    if request.suite != "all":
        if request.user_task:
            _catalog_task(
                request.benchmark_version, request.suite, request.user_task, "user"
            )
        if request.injection_task:
            _catalog_task(
                request.benchmark_version,
                request.suite,
                request.injection_task,
                "injection",
            )
    elif request.user_task or request.injection_task:
        raise HTTPException(
            status_code=422, detail="Task filters cannot be used with suite=all"
        )
    if (
        not request.benign_only
        and (request.attack not in ATTACKS or request.attack == "manual")
    ):
        raise HTTPException(status_code=422, detail="Unknown or interactive attack")
    if not request.defense_profiles:
        raise HTTPException(status_code=422, detail="At least one defense combination is required")
    if len(set(request.defense_profiles)) != len(request.defense_profiles):
        raise HTTPException(status_code=422, detail="Duplicate defense combination")
    try:
        for profile in request.defense_profiles:
            defense_config(profile)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    eligible_user_tasks: dict[str, tuple[str, ...]] | None = None
    if request.baseline_success_only:
        eligible_user_tasks = _baseline_success_tasks(
            str(request.baseline_source_job),
            request.benchmark_version,
        )
        requested_suites = (
            {"workspace", "slack", "travel", "banking"}
            if request.suite == "all"
            else {request.suite}
        )
        eligible_user_tasks = {
            suite: task_ids
            for suite, task_ids in eligible_user_tasks.items()
            if suite in requested_suites
        }
        if not eligible_user_tasks:
            raise HTTPException(
                status_code=422,
                detail="No successful baseline UserTasks match the selected suite",
            )

    job_id = f"agentdojo-{uuid4().hex[:12]}"
    selection = (
        {
            "mode": "baseline_success_only",
            "source_job": request.baseline_source_job,
            "total": sum(len(ids) for ids in eligible_user_tasks.values()),
            "suites": {
                suite: list(ids)
                for suite, ids in eligible_user_tasks.items()
            },
        }
        if eligible_user_tasks is not None
        else None
    )
    state = {
        "job_id": job_id,
        "status": "queued",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "completed": 0,
        "total": len(request.defense_profiles),
        "current": None,
        "request": request.model_dump(mode="json"),
        "selection": selection,
    }
    _write_json_atomic(_job_dir(job_id) / "request.json", state["request"])
    if eligible_user_tasks is not None:
        _write_json_atomic(
            _eligible_task_snapshot(job_id),
            {
                suite: list(task_ids)
                for suite, task_ids in eligible_user_tasks.items()
            },
        )
    _write_json_atomic(_job_dir(job_id) / "checkpoint.json", {
        "completed_profiles": [], "reports": {}
    })
    _write_json_atomic(_job_dir(job_id) / "state.json", state)
    _schedule_job(job_id, request)
    return state


@app.get("/api/runs")
async def list_runs() -> dict[str, Any]:
    jobs = []
    if _job_root().is_dir():
        for directory in _job_root().iterdir():
            if directory.is_dir() and (directory / "state.json").is_file():
                jobs.append(_reconcile_job(directory.name))
    jobs.sort(key=lambda item: item.get("created_at", ""), reverse=True)
    return {"jobs": jobs}


@app.get("/api/runs/{job_id}")
async def run_status(job_id: str) -> dict[str, Any]:
    state = _reconcile_job(job_id)
    result = _read_json(_job_dir(job_id) / "result.json")
    return {**state, "report": result or state.get("report")}


@app.post("/api/runs/{job_id}/cancel", status_code=202)
async def cancel_run(job_id: str) -> dict[str, Any]:
    state = _job_state(job_id)
    if state["status"] not in {"queued", "running"}:
        raise HTTPException(status_code=409, detail="Run is not active")
    event = JOB_CANCEL_EVENTS.get(job_id)
    if event is None:
        return _update_job(job_id, status="interrupted")
    event.set()
    return _update_job(job_id, status="cancelling")


@app.post("/api/runs/{job_id}/resume", status_code=202)
async def resume_run(job_id: str) -> dict[str, Any]:
    state = _job_state(job_id)
    if state["status"] not in {"cancelled", "interrupted", "failed"}:
        raise HTTPException(status_code=409, detail="Run cannot be resumed")
    request = RunRequest.model_validate(_read_json(_job_dir(job_id) / "request.json"))
    state = _update_job(job_id, status="queued", error=None)
    _schedule_job(job_id, request)
    return state


@app.get("/api/runs/{job_id}/download")
async def download_run(job_id: str) -> FileResponse:
    path = _job_dir(job_id) / "result.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Result not found")
    return FileResponse(path, filename=f"{job_id}-result.json")
