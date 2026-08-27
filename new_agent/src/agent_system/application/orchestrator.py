from __future__ import annotations

from agent_system.application.memory_service import MemoryService
from agent_system.application.workflow import WorkflowEngine
from agent_system.contracts import CommandRequest, CommandResponse, Principal
from agent_system.ports.executors import ExecutorRegistry
from agent_system.ports.planning import ResponseGenerator, TaskPlanner
from agent_system.security.ports import AgentGuard


class Orchestrator:
    def __init__(
        self,
        *,
        planner: TaskPlanner,
        response_generator: ResponseGenerator,
        memory: MemoryService,
        executors: ExecutorRegistry,
        workflow: WorkflowEngine,
        guard: AgentGuard,
    ) -> None:
        self._planner = planner
        self._response_generator = response_generator
        self._memory = memory
        self._executors = executors
        self._workflow = workflow
        self._guard = guard

    async def handle(self, command: CommandRequest) -> CommandResponse:
        context = await self._memory.load_context(
            user_id=command.user_id,
            session_id=command.session_id,
            query=command.query,
        )
        capabilities = await self._executors.capabilities()
        plan = await self._planner.create_plan(command, context, capabilities)

        principal = Principal(
            user_id=command.user_id,
            session_id=command.session_id,
            permissions=command.permissions,
        )
        results = [] if plan.direct_response else await self._workflow.execute(plan, principal)
        answer = await self._response_generator.generate(command, plan, results)
        answer = await self._guard.inspect_response(answer)

        await self._memory.record_turn(
            user_id=command.user_id,
            session_id=command.session_id,
            query=command.query,
            answer=answer,
        )
        return CommandResponse(
            workflow_id=plan.workflow_id,
            answer=answer,
            plan=plan,
            results=results,
        )

