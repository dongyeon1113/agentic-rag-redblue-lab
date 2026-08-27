from __future__ import annotations

from fastapi.testclient import TestClient

from agent_system.api.orchestrator import create_app as create_orchestrator_app
from agent_system.infrastructure.in_process_executor import InProcessExecutorClient
from agent_system.ports.executors import ExecutorRegistry
from agent_system.services.local_db.app import create_app as create_local_db_app


def test_orchestrator_executes_explicit_dependency_plan() -> None:
    local_db_app = create_local_db_app()
    registry = ExecutorRegistry([
        InProcessExecutorClient(local_db_app.state.executor),
    ])
    client = TestClient(create_orchestrator_app(registry=registry))

    response = client.post(
        "/v1/commands",
        json={
            "user_id": "user-1",
            "session_id": "session-1",
            "query": "문서를 추가한 후 검색해줘",
            "permissions": ["document:write", "document:read"],
            "requested_tasks": [
                {
                    "task_id": "create-1",
                    "workflow_id": "workflow-1",
                    "executor": "local_db",
                    "action": "document_create",
                    "parameters": {
                        "document_id": "doc-1",
                        "namespace": "knowledge",
                        "title": "Test document",
                        "content": "Searchable content",
                    },
                    "idempotency_key": "workflow-1:create-1",
                },
                {
                    "task_id": "search-1",
                    "workflow_id": "workflow-1",
                    "executor": "local_db",
                    "action": "document_search",
                    "parameters": {
                        "query": "Searchable",
                        "namespace": "knowledge",
                    },
                    "depends_on": ["create-1"],
                    "idempotency_key": "workflow-1:search-1",
                },
            ],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert [result["status"] for result in payload["results"]] == [
        "succeeded",
        "succeeded",
    ]
    assert payload["results"][1]["output"]["documents"][0]["document_id"] == "doc-1"


def test_orchestrator_returns_safe_message_without_llm_planner() -> None:
    local_db_app = create_local_db_app()
    registry = ExecutorRegistry([
        InProcessExecutorClient(local_db_app.state.executor),
    ])
    client = TestClient(create_orchestrator_app(registry=registry))

    response = client.post(
        "/v1/commands",
        json={
            "user_id": "user-1",
            "session_id": "session-1",
            "query": "안녕하세요",
        },
    )

    assert response.status_code == 200
    assert "LLM planner is not configured" in response.json()["answer"]

