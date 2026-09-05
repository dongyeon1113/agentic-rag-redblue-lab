from __future__ import annotations

from contextlib import asynccontextmanager
import os
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Query, Response
from pydantic import BaseModel, ConfigDict, Field

from agent_system import __version__
from agent_system.application.memory_extraction import LlmMemoryExtractor
from agent_system.application.memory_service import MemoryService
from agent_system.application.tool_calling_agent import (
    AgentQueryRequest,
    AgentRunResponse,
    ApproveWorkflowRequest,
    PendingRunStore,
    ToolCallingAgent,
)
from agent_system.config import AgentSettings
from agent_system.infrastructure.embeddings import OllamaEmbeddingClient
from agent_system.infrastructure.json_memory import JsonSessionMemoryRepository
from agent_system.infrastructure.memory_contexts import (
    MEMORY_CONTEXT_PATTERN,
    VectorMemoryContextRegistry,
)
from agent_system.infrastructure.ollama import OllamaChatModel, OllamaUnavailableError
from agent_system.infrastructure.remote_executor import RemoteExecutorClient
from agent_system.ports.executors import ExecutorRegistry
from agent_system.ports.memory import MemoryItem
from agent_system.security.ports import AllowAllAgentGuard
from defense import DefenseConfig, DefensePipeline, TaskShield


class CreateMemoryRequest(BaseModel):
    user_id: str
    memory_context: str = Field(default="context1", pattern=MEMORY_CONTEXT_PATTERN)
    content: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class UpdateMemoryRequest(BaseModel):
    user_id: str
    memory_context: str = Field(default="context1", pattern=MEMORY_CONTEXT_PATTERN)
    content: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentQueryApiRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    memory_context: str = Field(
        default="context1",
        pattern=MEMORY_CONTEXT_PATTERN,
    )
    defense: DefenseConfig = Field(default_factory=DefenseConfig)
    retrieval_limit: int | None = Field(default=None, ge=1, le=100)


def create_app(
    *,
    agent: ToolCallingAgent | None = None,
    settings: AgentSettings | None = None,
) -> FastAPI:
    settings = settings or AgentSettings.from_env()
    http_client: httpx.AsyncClient | None = None
    model: OllamaChatModel | None = None
    executors: ExecutorRegistry | None = None

    memory_directory = Path(settings.memory_data_dir)
    session_repository = JsonSessionMemoryRepository(
        memory_directory / "sessions.json"
    )
    embedding_client = OllamaEmbeddingClient(
        model=settings.ollama_embedding_model,
        base_url=settings.ollama_embedding_base_url,
        timeout_seconds=settings.embedding_timeout_seconds,
    )
    memory_contexts = VectorMemoryContextRegistry(
        memory_directory=memory_directory,
        collection_prefix=settings.memory_chroma_collection,
        embedding=embedding_client,
        batch_size=settings.chroma_index_batch_size,
    )
    memory = MemoryService(session_repository, memory_contexts)

    prompt_guard_detector = None
    if os.getenv("PROMPT_GUARD_ENABLED", "false").casefold() in {"1", "true", "yes", "on"}:
        from defense.prompt_guard import PromptGuardDetector

        prompt_guard_detector = PromptGuardDetector(
            model_id=os.getenv("PROMPT_GUARD_MODEL", "meta-llama/Prompt-Guard-86M"),
            device=os.getenv("PROMPT_GUARD_DEVICE") or None,
            threshold=float(os.getenv("PROMPT_GUARD_THRESHOLD", "0.9")),
        )

    if agent is None:
        http_client = httpx.AsyncClient(
            timeout=settings.request_timeout_seconds
        )
        model = OllamaChatModel(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=settings.temperature,
            num_predict=settings.num_predict,
            think=settings.ollama_think,
            timeout_seconds=settings.request_timeout_seconds,
            http_client=http_client,
        )
        executors = ExecutorRegistry([
            RemoteExecutorClient(
                name="local_db",
                base_url=settings.local_db_agent_url,
                http_client=http_client,
            ),
            RemoteExecutorClient(
                name="gmail",
                base_url=settings.gmail_agent_url,
                http_client=http_client,
            ),
            RemoteExecutorClient(
                name="drive",
                base_url=settings.drive_agent_url,
                http_client=http_client,
            ),
        ])
        agent = ToolCallingAgent(
            model=model,
            executors=executors,
            memory=memory,
            defense_pipeline=DefensePipeline(
                prompt_guard_detector=prompt_guard_detector,
                embedding=embedding_client,
                task_shield=TaskShield(
                    model,
                    threshold=settings.taskshield_threshold,
                    fail_closed=settings.taskshield_fail_closed,
                ),
            ),
            memory_extractor=(
                LlmMemoryExtractor(
                    model,
                    minimum_confidence=settings.auto_memory_min_confidence,
                    max_memories=settings.auto_memory_max_items,
                )
                if settings.auto_memory_enabled
                else None
            ),
            guard=AllowAllAgentGuard(),
            pending_runs=PendingRunStore(),
            max_iterations=settings.max_tool_iterations,
            taskshield_max_feedback_rounds=(
                settings.taskshield_max_feedback_rounds
            ),
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        del app
        yield
        if http_client is not None:
            await http_client.aclose()

    app = FastAPI(
        title="complete-agent-orchestrator",
        version=__version__,
        lifespan=lifespan,
    )
    app.state.agent = agent
    app.state.model = model
    app.state.executors = executors
    app.state.memory = memory
    app.state.agent_permissions = settings.agent_permissions
    app.state.taskshield_default = settings.taskshield_enabled

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {
            "service": "agent-orchestrator",
            "status": "healthy",
            "version": __version__,
            "model": settings.ollama_model,
        }

    @app.get("/ready")
    async def ready() -> dict[str, Any]:
        if app.state.model is None or app.state.executors is None:
            return {"status": "ready", "mode": "injected-test-agent"}
        try:
            models = await app.state.model.installed_models()
            capabilities = await app.state.executors.capabilities()
        except (OllamaUnavailableError, httpx.HTTPError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        model_ready = any(
            name == settings.ollama_model
            or name == f"{settings.ollama_model}:latest"
            for name in models
        )
        if not model_ready:
            raise HTTPException(
                status_code=503,
                detail=f"Ollama model is not installed: {settings.ollama_model}",
            )
        return {
            "status": "ready",
            "model": settings.ollama_model,
            "tool_count": len(capabilities),
        }

    @app.get("/v1/permissions", response_model=list[str])
    async def list_permissions() -> list[str]:
        return sorted(app.state.agent_permissions)

    @app.post("/v1/agent/query", response_model=AgentRunResponse)
    async def query(request: AgentQueryApiRequest) -> AgentRunResponse:
        payload = request.model_dump()
        if (
            "defense" not in request.model_fields_set
            and app.state.taskshield_default
        ):
            payload["defense"] = request.defense.model_copy(
                update={"task_shield": True}
            )
        trusted_request = AgentQueryRequest(
            **payload,
            permissions=set(app.state.agent_permissions),
        )
        try:
            return await app.state.agent.run(trusted_request)
        except OllamaUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except (ValueError, LookupError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post(
        "/v1/agent/workflows/{workflow_id}/approve",
        response_model=AgentRunResponse,
    )
    async def approve(
        workflow_id: str,
        request: ApproveWorkflowRequest,
    ) -> AgentRunResponse:
        try:
            return await app.state.agent.approve(workflow_id, request)
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except OllamaUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.delete("/v1/agent/workflows/{workflow_id}")
    async def cancel(
        workflow_id: str,
        user_id: str = Query(min_length=1),
        session_id: str = Query(min_length=1),
    ) -> dict[str, str]:
        try:
            app.state.agent.cancel(workflow_id, user_id, session_id)
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"workflow_id": workflow_id, "status": "cancelled"}

    @app.get("/v1/memories/{user_id}", response_model=list[MemoryItem])
    async def list_memories(
        user_id: str,
        memory_context: str = Query(default="context1", pattern=MEMORY_CONTEXT_PATTERN),
    ) -> list[MemoryItem]:
        return await app.state.memory.long_term(memory_context).list(user_id)

    @app.post("/v1/memories", response_model=MemoryItem, status_code=201)
    async def create_memory(request: CreateMemoryRequest) -> MemoryItem:
        return await app.state.memory.long_term(request.memory_context).create(
            request.user_id,
            request.content,
            request.metadata,
        )

    @app.put("/v1/memories/{memory_id}", response_model=MemoryItem)
    async def update_memory(
        memory_id: str,
        request: UpdateMemoryRequest,
    ) -> MemoryItem:
        item = MemoryItem(
            memory_id=memory_id,
            content=request.content,
            metadata=request.metadata,
        )
        try:
            await app.state.memory.long_term(request.memory_context).update(
                request.user_id, item
            )
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return item

    @app.delete("/v1/memories/{memory_id}", status_code=204, response_class=Response)
    async def delete_memory(
        memory_id: str,
        user_id: str,
        memory_context: str = Query(default="context1", pattern=MEMORY_CONTEXT_PATTERN),
    ) -> Response:
        await app.state.memory.long_term(memory_context).delete(user_id, memory_id)
        return Response(status_code=204)

    @app.get("/v1/memory-contexts", response_model=list[str])
    async def list_memory_contexts() -> list[str]:
        return app.state.memory.list_contexts()

    return app


app = create_app()

