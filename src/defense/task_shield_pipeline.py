from __future__ import annotations

from typing import Any

from agent_system.contracts import AgentTask, Capability, TaskResult
from defense.models import DefenseConfig, DefenseFinding, DefenseReport
from defense.pipeline import DefensePipeline as BaseDefensePipeline
from defense.task_shield import TaskShield, TaskShieldDecision, TaskShieldSession


class DefensePipeline(BaseDefensePipeline):
    """Existing retrieval defenses plus TaskShield message-boundary checks."""

    def __init__(
        self,
        prompt_guard_detector: Any | None = None,
        embedding: Any | None = None,
        task_shield: TaskShield | None = None,
    ) -> None:
        super().__init__(
            prompt_guard_detector=prompt_guard_detector,
            embedding=embedding,
        )
        self._task_shield = task_shield

    async def start_task_shield(
        self,
        *,
        workflow_id: str,
        user_goal: str,
        config: DefenseConfig,
        report: DefenseReport,
    ) -> TaskShieldSession | None:
        if not config.task_shield:
            return None
        shield = self._require_task_shield()
        session, latency_ms, checker_failed = await shield.start_session(
            workflow_id=workflow_id,
            user_goal=user_goal,
        )
        report.taskshield_checks += 1
        report.detector_latency_ms += latency_ms
        report.taskshield_user_tasks = list(session.user_tasks)
        if checker_failed:
            report.taskshield_checker_failures += 1
            report.findings.append(DefenseFinding(
                defense="task_shield",
                record_id=workflow_id,
                action="flagged",
                reason=(
                    "User-task extraction failed; the literal user request was "
                    "used as the alignment target."
                ),
            ))
        return session

    async def inspect_tool_call(
        self,
        *,
        session: TaskShieldSession | None,
        task: AgentTask,
        capability: Capability,
        config: DefenseConfig,
        report: DefenseReport,
    ) -> tuple[bool, str | None]:
        if not config.task_shield:
            return True, None
        shield = self._require_task_shield()
        if session is None:
            raise RuntimeError("TaskShield session was not initialized")
        decision = await shield.inspect_tool_call(session, task, capability)
        self._record_decision(report, task.task_id, decision)
        if decision.allowed:
            return True, None
        report.taskshield_blocked_calls += 1
        report.findings.append(DefenseFinding(
            defense="task_shield",
            record_id=task.task_id,
            action="action_blocked",
            reason=decision.reason,
            metadata={
                "executor": task.executor,
                "action": task.action,
                "score": round(decision.total_score, 6),
                "candidate": decision.candidate,
            },
        ))
        feedback = shield.feedback(
            session,
            [decision],
            source="assistant tool call",
        )
        if feedback:
            report.taskshield_feedback_messages += 1
        return False, feedback

    async def inspect_result(
        self,
        task: AgentTask,
        result: TaskResult,
        config: DefenseConfig,
        report: DefenseReport,
        task_shield_session: TaskShieldSession | None = None,
    ) -> tuple[TaskResult, list[str]]:
        taskshield_feedback: list[str] = []
        if config.task_shield and result.error is None and result.output:
            shield = self._require_task_shield()
            if task_shield_session is None:
                raise RuntimeError("TaskShield session was not initialized")
            inspection = await shield.inspect_tool_output(
                task_shield_session,
                task,
                result.output,
            )
            report.taskshield_checks += 1
            report.detector_latency_ms += inspection.latency_ms
            if inspection.checker_failed:
                report.taskshield_checker_failures += 1
            report.taskshield_flagged_instructions += len(inspection.misaligned)
            for decision in inspection.misaligned:
                report.findings.append(DefenseFinding(
                    defense="task_shield",
                    record_id=task.task_id,
                    action="flagged",
                    reason=decision.reason,
                    metadata={
                        "source": "tool_output",
                        "score": round(decision.total_score, 6),
                        "instruction": decision.candidate,
                    },
                ))
            if inspection.feedback:
                taskshield_feedback.append(inspection.feedback)
                report.taskshield_feedback_messages += 1

        inspected, instructions = await super().inspect_result(
            task,
            result,
            config,
            report,
        )
        return inspected, [*taskshield_feedback, *instructions]

    async def inspect_response(
        self,
        *,
        session: TaskShieldSession | None,
        workflow_id: str,
        response: str,
        config: DefenseConfig,
        report: DefenseReport,
    ) -> str | None:
        if not config.task_shield:
            return None
        shield = self._require_task_shield()
        if session is None:
            raise RuntimeError("TaskShield session was not initialized")
        inspection = await shield.inspect_response(session, response)
        report.taskshield_checks += 1
        report.detector_latency_ms += inspection.latency_ms
        if inspection.checker_failed:
            report.taskshield_checker_failures += 1
        report.taskshield_flagged_instructions += len(inspection.misaligned)
        for decision in inspection.misaligned:
            report.findings.append(DefenseFinding(
                defense="task_shield",
                record_id=workflow_id,
                action="blocked",
                reason=decision.reason,
                metadata={
                    "source": "assistant_response",
                    "score": round(decision.total_score, 6),
                    "instruction": decision.candidate,
                },
            ))
        if inspection.feedback:
            report.taskshield_feedback_messages += 1
        return inspection.feedback

    def _require_task_shield(self) -> TaskShield:
        if self._task_shield is None:
            raise RuntimeError(
                "TaskShield is enabled but no alignment checker is configured"
            )
        return self._task_shield

    @staticmethod
    def _record_decision(
        report: DefenseReport,
        record_id: str,
        decision: TaskShieldDecision,
    ) -> None:
        report.taskshield_checks += 1
        report.detector_latency_ms += decision.latency_ms
        if decision.checker_failed:
            report.taskshield_checker_failures += 1

