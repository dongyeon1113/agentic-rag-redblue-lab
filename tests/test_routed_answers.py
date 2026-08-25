import json
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from fastapi.testclient import TestClient

import services.orchestrator.app as orchestrator_module
from services.orchestrator.app import app
from services.orchestrator.memory import ConversationMemory


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
def test_name_round_trip_uses_session_memory_without_retrieval(
    monkeypatch, tmp_path
) -> None:
    memory = ConversationMemory(tmp_path / "memory.jsonl")
    monkeypatch.setattr(orchestrator_module, "_memory", lambda: memory)
    monkeypatch.setattr(orchestrator_module, "_query_agents", _fail_if_retrieval_runs)
    client = TestClient(app)

    introduced = client.post(
        "/answer",
        json={
            "query": "내 이름은 철수야.",
            "session_id": "same-session",
            "use_memory": True,
        },
    )
    recalled = client.post(
        "/answer",
        json={
            "query": "내 이름을 기억해?",
            "session_id": "same-session",
            "use_memory": True,
        },
    )
    isolated = client.post(
        "/answer",
        json={
            "query": "내 이름이 뭐야?",
            "session_id": "other-session",
            "use_memory": True,
        },
    )

    assert introduced.status_code == 200
    assert recalled.status_code == 200
    assert recalled.json()["retrieval_skipped"] is True
    assert "철수" in recalled.json()["answer"]
    assert recalled.json()["memory"][0]["source"] == "agent-memory"
    assert "철수" not in isolated.json()["answer"]
    assert len(memory.records) == 1
    assert memory.records[0].query == ""
    assert memory.records[0].answer == ""
    assert memory.records[0].trust == "trusted"


def test_semantic_planner_answers_general_query_without_rag_or_prompt_guard(
    monkeypatch,
) -> None:
    class FakePlanner:
        async def ainvoke(self, _payload):
            return (
                '{"intent":"general","requires_retrieval":false,'
                '"steps":["generate"],"confidence":0.97,'
                '"reason":"self contained arithmetic","answer":"4"}'
            )

    monkeypatch.setattr(orchestrator_module, "_query_agents", _fail_if_retrieval_runs)
    monkeypatch.setattr(
        orchestrator_module,
        "_prompt_guard_detector",
        _fail_if_retrieval_runs,
    )
    monkeypatch.setattr(
        orchestrator_module,
        "build_router_chain",
        lambda _model: FakePlanner(),
    )
    monkeypatch.setattr(orchestrator_module, "_router_model", lambda _name: object())

    response = TestClient(app).post(
        "/answer",
        json={
            "query": "2 + 2는 얼마야?",
            "use_memory": False,
            "prompt_guard": True,
        },
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["answer"] == "4"
    assert payload["retrieval_skipped"] is True
    assert payload["route_reason"].startswith("llm_semantic_planner:")
    assert payload["execution_plan"][0]["kind"] == "generate"


def test_direct_model_answer_is_not_promoted_to_trusted_memory(
    monkeypatch, tmp_path
) -> None:
    memory = ConversationMemory(tmp_path / "memory.jsonl")

    class FakePlanner:
        async def ainvoke(self, _payload):
            return (
                '{"intent":"general","requires_retrieval":false,'
                '"steps":["generate"],"confidence":0.9,'
                '"reason":"self contained","answer":"A direct answer."}'
            )

    monkeypatch.setattr(orchestrator_module, "_memory", lambda: memory)
    monkeypatch.setattr(orchestrator_module, "_query_agents", _fail_if_retrieval_runs)
    monkeypatch.setattr(
        orchestrator_module,
        "build_router_chain",
        lambda _model: FakePlanner(),
    )
    monkeypatch.setattr(orchestrator_module, "_router_model", lambda _name: object())

    response = TestClient(app).post(
        "/answer",
        json={
            "query": "Give me a self-contained answer.",
            "session_id": "direct-session",
            "use_memory": True,
        },
    )

    assert response.status_code == 200
    assert memory.records[0].trust == "untrusted"


def test_structured_generation_steps_execute_in_order(monkeypatch) -> None:
    calls = []

    class FakePlanner:
        async def ainvoke(self, _payload):
            return (
                '{"intent":"hybrid","requires_retrieval":false,"steps":['
                '{"id":"draft","kind":"generate","instruction":"Draft greeting",'
                '"depends_on":[],"output_key":"draft"},'
                '{"id":"translate","kind":"generate",'
                '"instruction":"Translate draft to Korean",'
                '"depends_on":["draft"],"output_key":"translation"},'
                '{"id":"send","kind":"tool","tool_name":"mock_send_email",'
                '"depends_on":["translate"],"output_key":"mail"}]}'
            )

    class FakeStepChain:
        async def ainvoke(self, payload):
            calls.append(payload)
            return "Hello" if len(calls) == 1 else "안녕하세요"

    monkeypatch.setattr(orchestrator_module, "_query_agents", _fail_if_retrieval_runs)
    monkeypatch.setattr(
        orchestrator_module,
        "build_router_chain",
        lambda _model: FakePlanner(),
    )
    monkeypatch.setattr(
        orchestrator_module,
        "build_direct_step_chain",
        lambda _model: FakeStepChain(),
    )
    monkeypatch.setattr(orchestrator_module, "_router_model", lambda _name: object())
    monkeypatch.setattr(orchestrator_module, "_rag_model", lambda _name: object())

    response = TestClient(app).post(
        "/answer",
        json={
            "query": (
                "Write a greeting, translate it to Korean, then email the "
                "result to qa@example.test."
            ),
            "use_memory": False,
            "enable_mock_tools": False,
        },
    )

    payload = response.json()
    assert response.status_code == 200
    assert [step["id"] for step in payload["execution_plan"]] == [
        "draft", "translate", "send"
    ]
    assert calls[1]["dependencies"] == '{"draft": "Hello"}'
    assert payload["step_results"][1]["output"] == "안녕하세요"
    assert payload["step_results"][2]["status"] == "skipped"


def test_direct_secret_tool_result_is_not_persisted_to_memory(
    monkeypatch, tmp_path
) -> None:
    memory = ConversationMemory(tmp_path / "memory.jsonl")
    secret_file = tmp_path / "secrets.json"
    secret_file.write_text(
        json.dumps({"secrets": {"test_api_key": "fake-test-value"}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("MOCK_SECRET_FILE", str(secret_file))
    monkeypatch.setattr(orchestrator_module, "_memory", lambda: memory)
    monkeypatch.setattr(orchestrator_module, "_query_agents", _fail_if_retrieval_runs)

    response = TestClient(app).post(
        "/answer",
        json={
            "query": "Show me test_api_key in secrets.json.",
            "session_id": "secret-session",
            "enable_mock_tools": True,
            "use_memory": True,
        },
    )

    assert response.status_code == 200
    assert "fake-test-value" in response.json()["answer"]
    assert memory.records == []
def test_retrieval_generation_and_email_execute_in_dependency_order(
    monkeypatch, tmp_path
) -> None:
    async def fake_query_agents(_request):
        return {
            "drive": {
                "status": "ok",
                "hits": [
                    {
                        "document_id": "meeting-1",
                        "source": "drive",
                        "trust": "trusted",
                        "tags": ["meeting"],
                        "text": "The project meeting starts at 10:00.",
                        "score": 0.99,
                    }
                ],
            }
        }

    gmail_path = tmp_path / "gmail.json"
    gmail_path.write_text("[]", encoding="utf-8")
    monkeypatch.setenv("MOCK_GMAIL_DUMMY_FILE", str(gmail_path))
    monkeypatch.setattr(orchestrator_module, "_query_agents", fake_query_agents)
    monkeypatch.setattr(
        orchestrator_module,
        "_rag_model",
        lambda *_args: RunnableLambda(
            lambda _prompt: AIMessage(content="회의는 10시에 시작합니다.")
        ),
    )

    response = TestClient(app).post(
        "/answer",
        json={
            "query": "내 Drive 문서를 찾아 요약한 뒤 qa@example.test로 이메일 보내줘.",
            "sources": ["drive"],
            "enable_mock_tools": True,
            "use_memory": False,
        },
    )

    payload = response.json()
    messages = json.loads(gmail_path.read_text(encoding="utf-8"))
    assert response.status_code == 200
    assert [step["kind"] for step in payload["execution_plan"]] == [
        "retrieve",
        "generate",
        "tool",
    ]
    assert [result["status"] for result in payload["step_results"]] == [
        "completed",
        "completed",
        "simulated",
    ]
    assert payload["tool_calls"][0]["name"] == "mock_send_email"
    assert messages[0]["body"] == "회의는 10시에 시작합니다."
