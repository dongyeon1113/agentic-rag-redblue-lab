from fastapi import FastAPI

from agent_system.services.local_db.domain import InMemoryDocumentRepository
from agent_system.services.local_db.tools import create_handlers
from agent_system.tool_runtime.api import create_tool_service_app
from agent_system.tool_runtime.executor import DomainExecutor
from agent_system.tool_runtime.policies import (
    DigestApprovalVerifier,
    PermissionAuthorizationService,
)


def create_app() -> FastAPI:
    repository = InMemoryDocumentRepository()
    executor = DomainExecutor(
        name="local_db",
        handlers=create_handlers(repository),
        authorization=PermissionAuthorizationService(),
        approval_verifier=DigestApprovalVerifier(),
    )
    return create_tool_service_app(executor)


app = create_app()

