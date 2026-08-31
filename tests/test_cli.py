from __future__ import annotations

import json
from dataclasses import replace
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from agent_system.api.agent import create_app
from agent_system.application.tool_calling_agent import AgentRunResponse
from agent_system.config import AgentSettings
from agent_system.cli import AgentApiClient, AgentShell


def response(
    *,
    status: str = "completed",
    answer: str = "완료했습니다.",
    approvals: list[dict[str, Any]] | None = None,
) -> AgentRunResponse:
    return AgentRunResponse.model_validate({
        "workflow_id": "workflow-1",
        "status": status,
        "answer": answer,
        "results": [],
        "approval_requests": approvals or [],
        "model": "test-model",
        "iterations": 1,
        "memory_context": "context1",
        "stored_memories": [],
    })


def test_api_client_sends_query_without_client_controlled_permissions() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json=response(answer="검색 완료").model_dump(mode="json"),
        )

    http = httpx.Client(transport=httpx.MockTransport(handler))
    client = AgentApiClient("http://agent.test/", http_client=http)

    result = client.query(
        user_id="user-1",
        session_id="session-1",
        memory_context="context2",
        query="문서를 찾아줘",
    )

    assert result.answer == "검색 완료"
    assert captured == {
        "method": "POST",
        "path": "/v1/agent/query",
        "payload": {
            "user_id": "user-1",
            "session_id": "session-1",
            "memory_context": "context2",
            "query": "문서를 찾아줘",
        },
    }


def test_unknown_server_permission_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_PERMISSIONS", "document:read,root:anything")
    with pytest.raises(ValueError, match="root:anything"):
        AgentSettings.from_env()


class RecordingAgent:
    def __init__(self) -> None:
        self.request = None

    async def run(self, request):
        self.request = request
        return response(answer="서버 권한 적용")


def test_api_rejects_client_permissions_and_injects_server_policy(tmp_path) -> None:
    agent = RecordingAgent()
    settings = replace(
        AgentSettings.from_env(),
        agent_permissions=frozenset({"document:read", "gmail:send"}),
        memory_data_dir=str(tmp_path),
    )
    app = create_app(agent=agent, settings=settings)

    with TestClient(app) as client:
        rejected = client.post("/v1/agent/query", json={
            "user_id": "user-1",
            "session_id": "session-1",
            "query": "메일 전송",
            "permissions": ["secret:read"],
        })
        accepted = client.post("/v1/agent/query", json={
            "user_id": "user-1",
            "session_id": "session-1",
            "query": "메일 전송",
        })
        listed = client.get("/v1/permissions")

    assert rejected.status_code == 422
    assert accepted.status_code == 200
    assert agent.request.permissions == {"document:read", "gmail:send"}
    assert listed.json() == ["document:read", "gmail:send"]


def test_api_client_accepts_response_from_server_before_memory_fields() -> None:
    legacy_response = response(answer="이전 서버 응답").model_dump(mode="json")
    legacy_response.pop("memory_context")
    legacy_response.pop("stored_memories")

    client = AgentApiClient(
        "http://agent.test",
        http_client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json=legacy_response)
            )
        ),
    )

    result = client.query(
        user_id="user-1",
        session_id="session-1",
        memory_context="context2",
        query="안녕",
    )

    assert result.answer == "이전 서버 응답"
    assert result.memory_context == "context2"
    assert result.stored_memories == []


class FakeClient:
    def __init__(self, responses: list[AgentRunResponse]) -> None:
        self.responses: Iterator[AgentRunResponse] = iter(responses)
        self.calls: list[tuple[str, Any]] = []

    def query(self, **kwargs: Any) -> AgentRunResponse:
        self.calls.append(("query", kwargs))
        return next(self.responses)

    def approve(self, workflow_id: str, **kwargs: Any) -> AgentRunResponse:
        self.calls.append(("approve", {"workflow_id": workflow_id, **kwargs}))
        return next(self.responses)

    def cancel(self, workflow_id: str, **kwargs: Any) -> None:
        self.calls.append(("cancel", {"workflow_id": workflow_id, **kwargs}))

    def permissions(self) -> list[str]:
        self.calls.append(("permissions", None))
        return ["document:read", "gmail:send"]

    def memory_contexts(self) -> list[str]:
        return ["context1", "context2"]

    def memories(self, user_id: str, memory_context: str) -> list[dict[str, Any]]:
        self.calls.append(("memories", (user_id, memory_context)))
        return []


APPROVALS = [{
    "task_id": "task-1",
    "executor": "gmail",
    "action": "message_send",
    "risk": "external_write",
    "parameters": {
        "to": "researcher@example.com",
        "subject": "완료",
        "body": "테스트가 완료됐습니다.",
    },
}]


def make_shell(
    client: FakeClient,
    *,
    answers: list[str] | None = None,
    output: list[str] | None = None,
) -> AgentShell:
    answers_iterator = iter(answers or [])
    return AgentShell(
        client,  # type: ignore[arg-type]
        user_id="user-1",
        session_id="session-1",
        memory_context="context1",
        input_fn=lambda prompt: next(answers_iterator),
        output_fn=(output if output is not None else []).append,
    )


def test_shell_approves_pending_tasks_and_prints_final_answer() -> None:
    pending = response(
        status="awaiting_approval",
        answer="승인이 필요합니다.",
        approvals=APPROVALS,
    )
    completed = response(answer="메일을 보냈습니다.")
    client = FakeClient([pending, completed])
    output: list[str] = []
    shell = make_shell(client, answers=["y"], output=output)

    result = shell.ask("완료 메일을 보내줘")

    assert result == completed
    assert client.calls[0][0] == "query"
    assert client.calls[1] == (
        "approve",
        {
            "workflow_id": "workflow-1",
            "user_id": "user-1",
            "session_id": "session-1",
            "memory_context": "context1",
            "task_ids": {"task-1"},
        },
    )
    assert any("external_write" in line for line in output)
    assert output[-1] == "agent> 메일을 보냈습니다."


def test_shell_rejection_cancels_pending_workflow() -> None:
    pending = response(
        status="awaiting_approval",
        answer="승인이 필요합니다.",
        approvals=APPROVALS,
    )
    client = FakeClient([pending])
    output: list[str] = []
    shell = make_shell(client, answers=["n"], output=output)

    result = shell.ask("완료 메일을 보내줘")

    assert result is None
    assert client.calls[-1] == (
        "cancel",
        {
            "workflow_id": "workflow-1",
            "user_id": "user-1",
            "session_id": "session-1",
        },
    )
    assert output[-1] == "agent> 작업을 승인하지 않아 취소했습니다."


def test_shell_commands_change_session_context_and_read_server_permissions() -> None:
    client = FakeClient([])
    output: list[str] = []
    shell = make_shell(client, output=output)

    assert shell._command("/new experiment-session")
    assert shell._command("/context context2")
    assert shell._command("/permissions")

    assert shell.session_id == "experiment-session"
    assert shell.memory_context == "context2"
    assert output[-1] == "서버 권한: document:read, gmail:send"

    try:
        shell._command("/permissions add drive:write")
    except ValueError as exc:
        assert "서버 관리 항목" in str(exc)
    else:
        raise AssertionError("client-side permission mutation must be rejected")


def test_api_client_turns_error_detail_into_readable_exception() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, json={"detail": "permission denied"})

    client = AgentApiClient(
        "http://agent.test",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    try:
        client.ready()
    except RuntimeError as exc:
        assert str(exc) == "에이전트 API 오류 (422): permission denied"
    else:
        raise AssertionError("AgentApiError was not raised")
