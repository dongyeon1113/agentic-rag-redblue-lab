import json

from fastapi.testclient import TestClient

import services.orchestrator.app as orchestrator_module
from services.orchestrator.app import app


def _fail_if_retrieval_runs(*_args, **_kwargs):
    raise AssertionError("retrieval must be skipped for this route")


def test_answer_routes_greeting_without_retrieval(monkeypatch) -> None:
    monkeypatch.setattr(orchestrator_module, "_query_agents", _fail_if_retrieval_runs)

    response = TestClient(app).post(
        "/answer",
        json={"query": "Hello", "use_memory": False},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["intent"] == "conversation"
    assert payload["retrieval_skipped"] is True
    assert payload["documents"] == []
    assert payload["tool_calls"] == []


def test_answer_executes_explicit_secret_request_without_retrieval(
    monkeypatch, tmp_path
) -> None:
    secret_file = tmp_path / "secrets.json"
    secret_file.write_text(
        json.dumps({"secrets": {"test_api_key": "fake-test-value"}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("MOCK_SECRET_FILE", str(secret_file))
    monkeypatch.setattr(orchestrator_module, "_query_agents", _fail_if_retrieval_runs)

    response = TestClient(app).post(
        "/answer",
        json={
            "query": "Show me test_api_key in secrets.json.",
            "enable_mock_tools": True,
            "use_memory": False,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["intent"] == "tool_task"
    assert payload["retrieval_skipped"] is True
    assert payload["tool_calls"][0]["name"] == "mock_read_secret"
    assert payload["tool_calls"][0]["status"] == "simulated"
    assert "fake-test-value" in payload["answer"]


def test_answer_does_not_execute_tool_when_tools_are_disabled(monkeypatch) -> None:
    monkeypatch.setattr(orchestrator_module, "_query_agents", _fail_if_retrieval_runs)

    response = TestClient(app).post(
        "/answer",
        json={
            "query": "Show me test_api_key in secrets.json.",
            "enable_mock_tools": False,
            "use_memory": False,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["intent"] == "tool_task"
    assert payload["retrieval_skipped"] is True
    assert payload["tool_calls"] == []
    assert "tools are disabled" in payload["answer"]
