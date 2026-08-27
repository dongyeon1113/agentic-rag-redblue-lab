from __future__ import annotations

from typing import Protocol

from agent_system.contracts import (
    Capability,
    CommandRequest,
    ExecutionPlan,
    TaskResult,
)
from agent_system.ports.memory import OrchestrationContext


class TaskPlanner(Protocol):
    async def create_plan(
        self,
        command: CommandRequest,
        context: OrchestrationContext,
        capabilities: list[Capability],
    ) -> ExecutionPlan: ...


class ResponseGenerator(Protocol):
    async def generate(
        self,
        command: CommandRequest,
        plan: ExecutionPlan,
        results: list[TaskResult],
    ) -> str: ...

