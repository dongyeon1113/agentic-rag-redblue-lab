from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

from fastapi.testclient import TestClient

from agent_system.contracts import AgentTask, ApprovalReceipt
from agent_system.services.drive.app import create_app as create_drive_app
from agent_system.services.gmail.app import create_app as create_gmail_app
from agent_system.services.local_db.app import create_app as create_local_db_app
from agent_system.tool_runtime.policies import resource_digest


def _principal(*permissions: str) -> dict:
    return {
        "user_id": "user-1",
        "session_id": "session-1",
        "permissions": list(permissions),
    }


def test_each_tool_service_exposes_health_and_capabilities() -> None:
    expected = {
        "local_db": "document_search",
        "gmail": "message_search",
        "drive": "item_search",
    }
    for app in (create_local_db_app(), create_gmail_app(), create_drive_app()):
        client = TestClient(app)
        health = client.get("/health")
        assert health.status_code == 200
        service = health.json()["service"]

        response = client.get("/v1/capabilities")
        assert response.status_code == 200
        actions = {item["action"] for item in response.json()["actions"]}
        assert expected[service] in actions


def test_local_db_tools_are_directly_accessible_over_fastapi() -> None:
    client = TestClient(create_local_db_app())
    document_id = f"doc-{uuid4()}"

    created = client.post(
        "/v1/tools/document_create",
        json={
            "parameters": {
                "document_id": document_id,
                "namespace": "knowledge",
                "title": "AgentDojo notes",
                "content": "Task alignment experiment",
            },
            "principal": _principal("document:write"),
        },
    )
    assert created.json()["status"] == "succeeded"

    searched = client.post(
        "/v1/tools/document_search",
        json={
            "parameters": {
                "query": "Task alignment",
                "namespace": "knowledge",
            },
            "principal": _principal("document:read"),
        },
    )
    assert searched.json()["output"]["documents"][0]["document_id"] == document_id


def test_local_db_rejects_missing_permission() -> None:
    client = TestClient(create_local_db_app())
    response = client.post(
        "/v1/tools/document_search",
        json={
            "parameters": {"query": "anything", "namespace": "knowledge"},
            "principal": _principal(),
        },
    )
    assert response.json()["error"]["code"] == "FORBIDDEN"


def test_gmail_send_requires_task_bound_approval() -> None:
    client = TestClient(create_gmail_app())
    task = AgentTask(
        task_id="send-1",
        workflow_id="workflow-1",
        executor="gmail",
        action="message_send",
        parameters={
            "sender": "me@example.com",
            "recipients": ["user@example.com"],
            "subject": "Test",
            "body": "Hello",
        },
        idempotency_key="workflow-1:send-1",
    )

    denied = client.post(
        "/v1/tasks",
        json={
            "task": task.model_dump(mode="json"),
            "principal": _principal("gmail:send"),
        },
    )
    assert denied.json()["error"]["code"] == "APPROVAL_REQUIRED"

    receipt = ApprovalReceipt(
        approval_id="approval-1",
        task_id=task.task_id,
        action=task.action,
        resource_digest=resource_digest(task),
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
    )
    approved_task = task.model_copy(update={"approval": receipt})
    allowed = client.post(
        "/v1/tasks",
        json={
            "task": approved_task.model_dump(mode="json"),
            "principal": _principal("gmail:send"),
        },
    )
    assert allowed.json()["status"] == "succeeded"

