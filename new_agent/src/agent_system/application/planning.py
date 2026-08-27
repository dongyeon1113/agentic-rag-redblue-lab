from __future__ import annotations

from uuid import uuid4

from agent_system.contracts import (
    Capability,
    CommandRequest,
    ExecutionPlan,
    TaskResult,
)
from agent_system.ports.memory import OrchestrationContext


class ExplicitTaskPlanner:
    """Runnable placeholder replaced later by an LLM-backed planner."""

    async def create_plan(
        self,
        command: CommandRequest,
        context: OrchestrationContext,
        capabilities: list[Capability],
    ) -> ExecutionPlan:
        del context
        workflow_id = (
            command.requested_tasks[0].workflow_id
            if command.requested_tasks
            else f"workflow-{uuid4()}"
        )
        known = {(item.executor, item.action) for item in capabilities}
        for task in command.requested_tasks:
            if (task.executor, task.action) not in known:
                raise ValueError(
                    f"Unknown executor action: {task.executor}.{task.action}"
                )
            if task.workflow_id != workflow_id:
                raise ValueError("All tasks must have the same workflow_id")

        direct_response = None
        if not command.requested_tasks:
            direct_response = (
                "LLM planner is not configured yet. Supply requested_tasks explicitly "
                "or replace ExplicitTaskPlanner with an LLM TaskPlanner adapter."
            )
        return ExecutionPlan(
            workflow_id=workflow_id,
            user_id=command.user_id,
            user_goal=command.query,
            tasks=command.requested_tasks,
            direct_response=direct_response,
        )


class TemplateResponseGenerator:
    async def generate(
        self,
        command: CommandRequest,
        plan: ExecutionPlan,
        results: list[TaskResult],
    ) -> str:
        del command
        if plan.direct_response is not None:
            return plan.direct_response
        succeeded = sum(result.status == "succeeded" for result in results)
        failed = sum(result.status == "failed" for result in results)
        return f"Workflow completed: {succeeded} succeeded, {failed} failed."

