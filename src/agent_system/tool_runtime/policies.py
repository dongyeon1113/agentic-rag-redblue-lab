from __future__ import annotations

import hashlib
import json

from agent_system.contracts import AgentTask, Capability, Principal


class AuthorizationError(PermissionError):
    pass


class ApprovalError(PermissionError):
    pass


def resource_digest(task: AgentTask) -> str:
    canonical = json.dumps(
        {
            "task_id": task.task_id,
            "action": task.action,
            "parameters": task.parameters,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class PermissionAuthorizationService:
    async def authorize(
        self,
        principal: Principal,
        capability: Capability,
    ) -> None:
        if capability.permission not in principal.permissions:
            raise AuthorizationError(
                f"Missing permission: {capability.permission}"
            )


class DigestApprovalVerifier:
    """Local verifier scaffold; replace with signed approval verification."""

    async def verify(self, task: AgentTask, capability: Capability) -> None:
        if not capability.approval_required:
            return
        receipt = task.approval
        if receipt is None:
            raise ApprovalError("This action requires user approval")
        if receipt.task_id != task.task_id or receipt.action != task.action:
            raise ApprovalError("Approval does not match the task")
        if receipt.is_expired():
            raise ApprovalError("Approval has expired")
        if receipt.resource_digest != resource_digest(task):
            raise ApprovalError("Task parameters changed after approval")

