from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from agent_system import __version__
from agent_system.application.memory_service import MemoryService
from agent_system.application.tool_calling_agent import (
    AgentQueryRequest,
    AgentRunResponse,
    ApproveWorkflowRequest,
    PendingRunStore,
    ToolCallingAgent,
)
from agent_system.config import AgentSettings
from agent_system.infrastructure.json_memory import (
    JsonLongTermMemoryRepository,
    JsonSessionMemoryRepository,
)
from agent_system.infrastructure.ollama import OllamaChatModel, OllamaUnavailableError
from agent_system.infrastructure.remote_executor import RemoteExecutorClient
from agent_system.ports.executors import ExecutorRegistry
from agent_system.ports.memory import MemoryItem
from agent_system.security.ports import AllowAllAgentGuard


class CreateMemoryRequest(BaseModel):
    user_id: str
    content: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class UpdateMemoryRequest(BaseModel):
    user_id: str
    content: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


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
    long_term_repository = JsonLongTermMemoryRepository(
        memory_directory / "long_term.json"
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
            memory=MemoryService(session_repository, long_term_repository),
            guard=AllowAllAgentGuard(),
            pending_runs=PendingRunStore(),
            max_iterations=settings.max_tool_iterations,
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
    app.state.long_term_memory = long_term_repository

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

    @app.post("/v1/agent/query", response_model=AgentRunResponse)
    async def query(request: AgentQueryRequest) -> AgentRunResponse:
        try:
            return await app.state.agent.run(request)
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
    async def list_memories(user_id: str) -> list[MemoryItem]:
        return await app.state.long_term_memory.list(user_id)

    @app.post("/v1/memories", response_model=MemoryItem, status_code=201)
    async def create_memory(request: CreateMemoryRequest) -> MemoryItem:
        return await app.state.long_term_memory.create(
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
            await app.state.long_term_memory.update(request.user_id, item)
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return item

    @app.delete("/v1/memories/{memory_id}", status_code=204)
    async def delete_memory(memory_id: str, user_id: str) -> None:
        await app.state.long_term_memory.delete(user_id, memory_id)

    return app


app = create_app()

