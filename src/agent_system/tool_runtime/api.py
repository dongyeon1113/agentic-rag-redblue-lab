from __future__ import annotations

from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agent_system import __version__
from agent_system.contracts import (
    AgentTask,
    ApprovalReceipt,
    DispatchRequest,
    Principal,
    TaskResult,
)
from agent_system.tool_runtime.executor import DomainExecutor


class DirectToolRequest(BaseModel):
    parameters: dict = Field(default_factory=dict)
    principal: Principal
    approval: ApprovalReceipt | None = None
    workflow_id: str = "direct-tool-call"


def create_tool_service_app(executor: DomainExecutor) -> FastAPI:
    app = FastAPI(title=f"{executor.name}-agent", version=__version__)
    app.state.executor = executor

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"service": executor.name, "status": "healthy", "version": __version__}

    @app.get("/v1/capabilities")
    async def capabilities() -> dict[str, object]:
        return {"executor": executor.name, "actions": executor.capabilities()}

    @app.post("/v1/tasks", response_model=TaskResult)
    async def execute_task(request: DispatchRequest) -> TaskResult:
        return await executor.execute(request.task, request.principal)

    @app.post("/v1/tools/{action}", response_model=TaskResult)
    async def execute_tool(action: str, request: DirectToolRequest) -> TaskResult:
        known_actions = {item.action for item in executor.capabilities()}
        if action not in known_actions:
            raise HTTPException(status_code=404, detail=f"Unknown action: {action}")

        task_id = f"direct-{uuid4()}"
        task = AgentTask(
            task_id=task_id,
            workflow_id=request.workflow_id,
            executor=executor.name,
            action=action,
            parameters=request.parameters,
            idempotency_key=task_id,
            approval=request.approval,
        )
        return await executor.execute(task, request.principal)

    return app

