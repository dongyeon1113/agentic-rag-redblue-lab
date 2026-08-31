from __future__ import annotations

from typing import Protocol

from agent_system.contracts import AgentTask, Capability, Principal, TaskResult


class ExecutorClient(Protocol):
    @property
    def name(self) -> str: ...

    async def capabilities(self) -> list[Capability]: ...

    async def execute(self, task: AgentTask, principal: Principal) -> TaskResult: ...


class ExecutorRegistry:
    def __init__(self, executors: list[ExecutorClient]) -> None:
        self._executors = {executor.name: executor for executor in executors}
        if len(self._executors) != len(executors):
            raise ValueError("Executor names must be unique")

    def get(self, name: str) -> ExecutorClient:
        try:
            return self._executors[name]
        except KeyError as exc:
            raise LookupError(f"Unknown executor: {name}") from exc

    async def capabilities(self) -> list[Capability]:
        capabilities: list[Capability] = []
        for executor in self._executors.values():
            capabilities.extend(await executor.capabilities())
        return capabilities

