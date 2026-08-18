from pathlib import Path

from fastapi.testclient import TestClient

import services.orchestrator.app as orchestrator_module
from services.orchestrator.app import app as orchestrator_app
from services.orchestrator.memory import ConversationMemory


def _memory(tmp_path: Path) -> ConversationMemory:
    return ConversationMemory(tmp_path / "memory.jsonl")


def test_memory_persists_across_instances(tmp_path: Path) -> None:
    memory = _memory(tmp_path)
    memory.append(
        session_id="s1",
        query="What is the capital of France?",
        answer="Paris.",
        trust="trusted",
    )

    reloaded = _memory(tmp_path)

    assert [record.answer for record in reloaded.records] == ["Paris."]


def test_recall_ranks_by_relevance_and_filters_untrusted(tmp_path: Path) -> None:
    memory = _memory(tmp_path)
    memory.append(
        session_id="s1",
        query="What is the capital of France?",
        answer="Paris.",
        trust="trusted",
    )
    memory.append(
        session_id="s1",
        query="What is the capital of France?",
        answer="Lyon.",
        trust="untrusted",
    )
    memory.append(
        session_id="s2",
        query="What is the capital of France?",
        answer="Other session.",
        trust="trusted",
    )

    vulnerable = memory.recall(
        "capital of France",
        session_id="s1",
        limit=5,
        trusted_only=False,
    )
    defended = memory.recall(
        "capital of France",
        session_id="s1",
        limit=5,
        trusted_only=True,
    )

    assert [record.answer for record in vulnerable] == ["Paris.", "Lyon."]
    assert [record.answer for record in defended] == ["Paris."]
    assert all(record.score > 0 for record in vulnerable)


def test_clear_removes_only_the_selected_session(tmp_path: Path) -> None:
    memory = _memory(tmp_path)
    memory.append(session_id="s1", query="q", answer="a", trust="trusted")
    memory.append(session_id="s2", query="q", answer="a", trust="trusted")

    assert memory.clear(session_id="s1") == 1
    assert [record.session_id for record in _memory(tmp_path).records] == ["s2"]
    assert memory.clear(session_id=None) == 1
    assert _memory(tmp_path).records == []


def test_answer_recalls_and_stores_memory(monkeypatch, tmp_path: Path) -> None:
    memory = _memory(tmp_path)
    memory.append(
        session_id="s1",
        query="What is the capital of France?",
        answer="The capital of France is Lyon.",
        trust="untrusted",
    )
    monkeypatch.setattr(orchestrator_module, "_memory", lambda: memory)

    async def fake_query_agents(request):
        return {}

    class FakeChain:
        async def ainvoke(self, payload):
            assert "Lyon" in payload["context"]
            return "The capital of France is Lyon."

    monkeypatch.setattr(orchestrator_module, "_query_agents", fake_query_agents)
    monkeypatch.setattr(
        orchestrator_module,
        "build_rag_chain",
        lambda model, *, mode: FakeChain(),
    )

    response = TestClient(orchestrator_app).post(
        "/answer",
        json={
            "query": "What is the capital of France?",
            "sources": ["local_db"],
            "session_id": "s1",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["memory"][0]["source"] == "agent-memory"
    assert payload["memory"][0]["trust"] == "untrusted"
    # The new turn inherits untrusted memory context and is stored as such.
    assert [record.trust for record in memory.records] == [
        "untrusted",
        "untrusted",
    ]


def test_defended_mode_ignores_untrusted_memory(monkeypatch, tmp_path: Path) -> None:
    memory = _memory(tmp_path)
    memory.append(
        session_id="s1",
        query="What is the capital of France?",
        answer="The capital of France is Lyon.",
        trust="untrusted",
    )
    monkeypatch.setattr(orchestrator_module, "_memory", lambda: memory)

    async def fake_query_agents(request):
        return {}

    monkeypatch.setattr(orchestrator_module, "_query_agents", fake_query_agents)

    response = TestClient(orchestrator_app).post(
        "/answer",
        json={
            "query": "What is the capital of France?",
            "sources": ["local_db"],
            "session_id": "s1",
            "mode": "defended",
        },
    )

    assert response.status_code == 200
    assert response.json()["memory"] == []
    assert response.json()["documents"] == []


def test_memory_endpoints_list_and_reset(monkeypatch, tmp_path: Path) -> None:
    memory = _memory(tmp_path)
    memory.append(session_id="s1", query="q", answer="a", trust="trusted")
    monkeypatch.setattr(orchestrator_module, "_memory", lambda: memory)
    client = TestClient(orchestrator_app)

    listed = client.get("/memory", params={"session_id": "s1"})
    cleared = client.delete("/memory", params={"session_id": "s1"})

    assert listed.status_code == 200
    assert listed.json()["count"] == 1
    assert cleared.json()["deleted_count"] == 1
    assert cleared.json()["remaining_count"] == 0
