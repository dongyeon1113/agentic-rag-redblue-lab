import os
from pathlib import Path

from fastapi import FastAPI

from agent_system.services.local_db.json_repository import JsonDocumentRepository
from agent_system.services.local_db.tools import create_handlers
from agent_system.tool_runtime.api import create_tool_service_app
from agent_system.tool_runtime.executor import DomainExecutor
from agent_system.tool_runtime.policies import (
    DigestApprovalVerifier,
    PermissionAuthorizationService,
)


def create_app() -> FastAPI:
    data_directory = Path(os.getenv("LOCAL_DB_DATA_DIR", "data/local_db"))
    repository = JsonDocumentRepository(
        data_file=data_directory / "documents.json",
        knowledge_seed_file=Path(
            os.getenv(
                "KNOWLEDGE_SEED_FILE",
                "mock_data/local_db/nq_10000.json",
            )
        ),
        secret_seed_file=Path(
            os.getenv(
                "SECRET_SEED_FILE",
                "mock_data/local_db/secrets.json",
            )
        ),
    )
    executor = DomainExecutor(
        name="local_db",
        handlers=create_handlers(repository),
        authorization=PermissionAuthorizationService(),
        approval_verifier=DigestApprovalVerifier(),
    )
    return create_tool_service_app(executor)


app = create_app()

