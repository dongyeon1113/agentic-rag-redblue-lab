from __future__ import annotations

import os
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException

from agent_system import __version__
from agent_system.application.memory_service import MemoryService
from agent_system.application.orchestrator import Orchestrator
from agent_system.application.planning import ExplicitTaskPlanner, TemplateResponseGenerator
from agent_system.application.workflow import PlanValidationError, WorkflowEngine
from agent_system.contracts import CommandRequest, CommandResponse
from agent_system.infrastructure.memory import (
    InMemoryLongTermMemoryRepository,
    InMemorySessionMemoryRepository,
)
from agent_system.infrastructure.remote_executor import RemoteExecutorClient
from agent_system.ports.executors import ExecutorRegistry
from agent_system.security.ports import AllowAllAgentGuard


def create_app(
    orchestrator: Orchestrator | None = None,
    registry: ExecutorRegistry | None = None,
) -> FastAPI:
    http_client: httpx.AsyncClient | None = None

    if orchestrator is None:
        http_client = httpx.AsyncClient()
        registry = registry or ExecutorRegistry([
            RemoteExecutorClient(
                name="local_db",
                base_url=os.getenv("LOCAL_DB_AGENT_URL", "http://localhost:8001"),
                http_client=http_client,
            ),
            RemoteExecutorClient(
                name="gmail",
                base_url=os.getenv("GMAIL_AGENT_URL", "http://localhost:8002"),
                http_client=http_client,
            ),
            RemoteExecutorClient(
                name="drive",
                base_url=os.getenv("DRIVE_AGENT_URL", "http://localhost:8003"),
                http_client=http_client,
            ),
        ])
        guard = AllowAllAgentGuard()
        memory = MemoryService(
            InMemorySessionMemoryRepository(),
            InMemoryLongTermMemoryRepository(),
        )
        orchestrator = Orchestrator(
            planner=ExplicitTaskPlanner(),
            response_generator=TemplateResponseGenerator(),
            memory=memory,
            executors=registry,
            workflow=WorkflowEngine(registry, guard),
            guard=guard,
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        del app
        yield
        if http_client is not None:
            await http_client.aclose()

    app = FastAPI(title="agent-orchestrator", version=__version__, lifespan=lifespan)
    app.state.orchestrator = orchestrator
    app.state.executors = registry

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"service": "orchestrator", "status": "healthy", "version": __version__}

    @app.get("/v1/capabilities")
    async def capabilities() -> dict[str, object]:
        if app.state.executors is None:
            return {"executors": []}
        try:
            items = await app.state.executors.capabilities()
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {"executors": items}

    @app.post("/v1/commands", response_model=CommandResponse)
    async def commands(command: CommandRequest) -> CommandResponse:
        try:
            return await app.state.orchestrator.handle(command)
        except (ValueError, LookupError, PlanValidationError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    return app


app = create_app()

