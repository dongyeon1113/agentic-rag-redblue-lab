import os
from pathlib import Path

from fastapi import FastAPI

from agent_system.services.drive.json_gateway import JsonDriveGateway
from agent_system.services.drive.tools import create_handlers
from agent_system.tool_runtime.api import create_tool_service_app
from agent_system.tool_runtime.executor import DomainExecutor
from agent_system.tool_runtime.policies import (
    DigestApprovalVerifier,
    PermissionAuthorizationService,
)


def create_app() -> FastAPI:
    data_directory = Path(os.getenv("DRIVE_DATA_DIR", "data/drive"))
    gateway = JsonDriveGateway(
        data_file=data_directory / "items.json",
        seed_file=Path(os.getenv("DRIVE_SEED_FILE", "mock_data/drive/items.json")),
    )
    executor = DomainExecutor(
        name="drive",
        handlers=create_handlers(gateway),
        authorization=PermissionAuthorizationService(),
        approval_verifier=DigestApprovalVerifier(),
    )
    return create_tool_service_app(executor)


app = create_app()

