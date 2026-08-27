from __future__ import annotations

from agent_system.contracts import (
    AgentTask,
    ErrorDetail,
    ExecutionPlan,
    Principal,
    TaskResult,
    TaskStatus,
)
from agent_system.ports.executors import ExecutorRegistry
from agent_system.security.ports import AgentGuard


class PlanValidationError(ValueError):
    pass


class WorkflowEngine:
    def __init__(
        self,
        executors: ExecutorRegistry,
        guard: AgentGuard,
    ) -> None:
        self._executors = executors
        self._guard = guard

    async def execute(
        self,
        plan: ExecutionPlan,
        principal: Principal,
    ) -> list[TaskResult]:
        ordered_tasks = self._topological_order(plan)
        results: dict[str, TaskResult] = {}

        for task in ordered_tasks:
            if any(
                results[dependency].status != TaskStatus.SUCCEEDED
                for dependency in task.depends_on
            ):
                results[task.task_id] = TaskResult(
                    task_id=task.task_id,
                    status=TaskStatus.SKIPPED,
                    error=ErrorDetail(
                        code="DEPENDENCY_FAILED",
                        message="A required task did not succeed",
                    ),
                )
                continue

            guarded_task = await self._guard.inspect_tool_call(task)
            executor = self._executors.get(guarded_task.executor)
            result = await executor.execute(guarded_task, principal)
            results[task.task_id] = await self._guard.inspect_tool_result(
                guarded_task, result
            )

        return [results[task.task_id] for task in ordered_tasks]

    @staticmethod
    def _topological_order(plan: ExecutionPlan) -> list[AgentTask]:
        tasks = {task.task_id: task for task in plan.tasks}
        if len(tasks) != len(plan.tasks):
            raise PlanValidationError("Task IDs must be unique")
        if any(task.workflow_id != plan.workflow_id for task in plan.tasks):
            raise PlanValidationError("Task workflow_id does not match the plan")

        for task in plan.tasks:
            missing = set(task.depends_on) - tasks.keys()
            if missing:
                raise PlanValidationError(
                    f"Task {task.task_id} has missing dependencies: {sorted(missing)}"
                )

        ordered: list[AgentTask] = []
        temporary: set[str] = set()
        permanent: set[str] = set()

        def visit(task_id: str) -> None:
            if task_id in permanent:
                return
            if task_id in temporary:
                raise PlanValidationError("Task dependency cycle detected")
            temporary.add(task_id)
            for dependency in tasks[task_id].depends_on:
                visit(dependency)
            temporary.remove(task_id)
            permanent.add(task_id)
            ordered.append(tasks[task_id])

        for task in plan.tasks:
            visit(task.task_id)
        return ordered

