from __future__ import annotations

from pydantic import ValidationError

from agent_system.contracts import AgentTask, Capability, Principal, TaskResult
from agent_system.security.ports import ApprovalVerifier, AuthorizationService
from agent_system.tool_runtime.handler import ToolHandler
from agent_system.tool_runtime.policies import ApprovalError, AuthorizationError


class DomainExecutor:
    def __init__(
        self,
        *,
        name: str,
        handlers: list[ToolHandler],
        authorization: AuthorizationService,
        approval_verifier: ApprovalVerifier,
    ) -> None:
        self.name = name
        self._handlers = {handler.capability.action: handler for handler in handlers}
        if len(self._handlers) != len(handlers):
            raise ValueError("Tool action names must be unique")
        if any(handler.capability.executor != name for handler in handlers):
            raise ValueError("Every capability must belong to this executor")
        self._authorization = authorization
        self._approval_verifier = approval_verifier

    def capabilities(self) -> list[Capability]:
        return [handler.capability for handler in self._handlers.values()]

    async def execute(self, task: AgentTask, principal: Principal) -> TaskResult:
        if task.executor != self.name:
            return TaskResult.failed(
                task.task_id,
                code="WRONG_EXECUTOR",
                message=f"Task targets {task.executor}, not {self.name}",
            )

        handler = self._handlers.get(task.action)
        if handler is None:
            return TaskResult.failed(
                task.task_id,
                code="UNKNOWN_ACTION",
                message=f"Unknown action: {task.action}",
            )

        try:
            request = handler.validate(task.parameters)
            await self._authorization.authorize(principal, handler.capability)
            await self._approval_verifier.verify(task, handler.capability)
            output = await handler.execute(request, principal)
            return TaskResult.succeeded(task.task_id, output)
        except ValidationError as exc:
            return TaskResult.failed(
                task.task_id,
                code="INVALID_PARAMETERS",
                message=str(exc),
            )
        except AuthorizationError as exc:
            return TaskResult.failed(
                task.task_id,
                code="FORBIDDEN",
                message=str(exc),
            )
        except ApprovalError as exc:
            return TaskResult.failed(
                task.task_id,
                code="APPROVAL_REQUIRED",
                message=str(exc),
            )
        except Exception as exc:  # Boundary maps adapter failures to a contract.
            return TaskResult.failed(
                task.task_id,
                code="TOOL_EXECUTION_FAILED",
                message=str(exc),
                retryable=True,
            )

