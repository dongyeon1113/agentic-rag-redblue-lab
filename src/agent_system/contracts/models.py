from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class TaskStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class RiskLevel(StrEnum):
    READ = "read"
    INTERNAL_WRITE = "internal_write"
    EXTERNAL_WRITE = "external_write"
    DELETE = "delete"
    SECRET_READ = "secret_read"


class Principal(BaseModel):
    user_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    permissions: set[str] = Field(default_factory=set)


class ApprovalReceipt(BaseModel):
    approval_id: str
    task_id: str
    action: str
    resource_digest: str
    expires_at: datetime

    def is_expired(self) -> bool:
        value = self.expires_at
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value <= datetime.now(timezone.utc)


class AgentTask(BaseModel):
    task_id: str = Field(min_length=1)
    workflow_id: str = Field(min_length=1)
    executor: str = Field(min_length=1)
    action: str = Field(min_length=1)
    parameters: dict[str, Any] = Field(default_factory=dict)
    idempotency_key: str = Field(min_length=1)
    approval: ApprovalReceipt | None = None



class ErrorDetail(BaseModel):
    code: str
    message: str
    retryable: bool = False


class TaskResult(BaseModel):
    task_id: str
    status: TaskStatus
    output: dict[str, Any] = Field(default_factory=dict)
    error: ErrorDetail | None = None

    @classmethod
    def succeeded(cls, task_id: str, output: dict[str, Any]) -> TaskResult:
        return cls(task_id=task_id, status=TaskStatus.SUCCEEDED, output=output)

    @classmethod
    def failed(
        cls,
        task_id: str,
        *,
        code: str,
        message: str,
        retryable: bool = False,
    ) -> TaskResult:
        return cls(
            task_id=task_id,
            status=TaskStatus.FAILED,
            error=ErrorDetail(code=code, message=message, retryable=retryable),
        )


class Capability(BaseModel):
    executor: str
    action: str
    description: str
    permission: str
    risk: RiskLevel
    approval_required: bool = False
    input_schema: dict[str, Any] = Field(default_factory=dict)


class DispatchRequest(BaseModel):
    task: AgentTask
    principal: Principal
