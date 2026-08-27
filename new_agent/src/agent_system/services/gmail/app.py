from fastapi import FastAPI

from agent_system.services.gmail.domain import InMemoryGmailGateway
from agent_system.services.gmail.tools import create_handlers
from agent_system.tool_runtime.api import create_tool_service_app
from agent_system.tool_runtime.executor import DomainExecutor
from agent_system.tool_runtime.policies import (
    DigestApprovalVerifier,
    PermissionAuthorizationService,
)


def create_app() -> FastAPI:
    gateway = InMemoryGmailGateway()
    executor = DomainExecutor(
        name="gmail",
        handlers=create_handlers(gateway),
        authorization=PermissionAuthorizationService(),
        approval_verifier=DigestApprovalVerifier(),
    )
    return create_tool_service_app(executor)


app = create_app()

