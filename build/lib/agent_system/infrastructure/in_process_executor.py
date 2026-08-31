from __future__ import annotations

from agent_system.contracts import AgentTask, Capability, Principal, TaskResult
from agent_system.tool_runtime.executor import DomainExecutor


class InProcessExecutorClient:
    """Adapter used by unit tests and optional single-process deployments."""

    def __init__(self, executor: DomainExecutor) -> None:
        self._executor = executor

    @property
    def name(self) -> str:
        return self._executor.name

    async def capabilities(self) -> list[Capability]:
        return self._executor.capabilities()

    async def execute(self, task: AgentTask, principal: Principal) -> TaskResult:
        return await self._executor.execute(task, principal)

