from __future__ import annotations

from typing import Protocol

from agent_system.contracts import AgentTask, Capability, Principal, TaskResult


class AuthorizationService(Protocol):
    async def authorize(
        self,
        principal: Principal,
        capability: Capability,
    ) -> None: ...


class ApprovalVerifier(Protocol):
    async def verify(self, task: AgentTask, capability: Capability) -> None: ...


class AgentGuard(Protocol):
    """Hook points for a later TaskShield implementation."""

    async def inspect_tool_call(self, task: AgentTask) -> AgentTask: ...

    async def inspect_tool_result(
        self,
        task: AgentTask,
        result: TaskResult,
    ) -> TaskResult: ...

    async def inspect_response(self, response: str) -> str: ...


class AllowAllAgentGuard:
    async def inspect_tool_call(self, task: AgentTask) -> AgentTask:
        return task

    async def inspect_tool_result(
        self,
        task: AgentTask,
        result: TaskResult,
    ) -> TaskResult:
        return result

    async def inspect_response(self, response: str) -> str:
        return response

