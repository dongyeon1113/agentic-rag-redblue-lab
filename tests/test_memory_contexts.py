from __future__ import annotations

import asyncio

import pytest

from agent_system.application.memory_extraction import (
    LlmMemoryExtractor,
    MemoryCandidate,
    MemoryExtraction,
)
from agent_system.application.memory_service import MemoryService
from agent_system.infrastructure.embeddings import DeterministicEmbeddingClient
from agent_system.infrastructure.memory import InMemorySessionMemoryRepository
from agent_system.infrastructure.memory_contexts import VectorMemoryContextRegistry
from agent_system.ports.memory import ConversationTurn, MemoryItem


class ExtractionModel:
    def __init__(self, extraction: MemoryExtraction) -> None:
        self.extraction = extraction
        self.messages = None

    async def generate_structured(self, messages, output_schema):
        assert output_schema is MemoryExtraction
        self.messages = messages
        return self.extraction


def _registry(tmp_path):
    return VectorMemoryContextRegistry(
        memory_directory=tmp_path / "memory",
        embedding=DeterministicEmbeddingClient(),
        collection_prefix="test-memory",
    )


def test_named_contexts_use_separate_json_and_chroma_collections(tmp_path) -> None:
    registry = _registry(tmp_path)
    context1 = registry.get("context1")
    context2 = registry.get("context2")
    asyncio.run(context1.save(
        "user-1", MemoryItem(memory_id="style", content="Use concise reports")
    ))
    asyncio.run(context2.save(
        "user-1", MemoryItem(memory_id="style", content="Likes hiking")
    ))

    assert (tmp_path / "memory/long_term.json").exists()
    assert (tmp_path / "memory/contexts/context2.json").exists()
    assert [item.content for item in asyncio.run(context1.list("user-1"))] == [
        "Use concise reports"
    ]
    assert [item.content for item in asyncio.run(context2.list("user-1"))] == [
        "Likes hiking"
    ]
    assert registry.list_contexts() == ["context1", "context2"]


def test_memory_context_rejects_paths(tmp_path) -> None:
    registry = _registry(tmp_path)
    with pytest.raises(ValueError, match="memory_context"):
        registry.get("../other-user")
    with pytest.raises(ValueError, match="memory_context"):
        registry.get("/tmp/memory")
    with pytest.raises(ValueError, match="memory_context"):
        registry.get("work")


def test_llm_extractor_filters_low_confidence_and_secrets() -> None:
    model = ExtractionModel(MemoryExtraction(memories=[
        MemoryCandidate(
            content="사용자는 한국어 답변을 선호한다.",
            category="preference",
            confidence=0.95,
        ),
        MemoryCandidate(
            content="api_key = abcdefghijklmnop",
            category="profile",
            confidence=0.99,
        ),
        MemoryCandidate(
            content="사용자는 오늘만 짧은 답변을 원한다.",
            category="preference",
            confidence=0.4,
        ),
    ]))
    extractor = LlmMemoryExtractor(model, minimum_confidence=0.8)

    candidates = asyncio.run(extractor.extract(
        session_turns=[ConversationTurn(role="user", content="이전 대화")],
        query="앞으로 답변은 한국어로 해줘",
        answer="알겠습니다.",
    ))

    assert [item.content for item in candidates] == [
        "사용자는 한국어 답변을 선호한다."
    ]
    assert "앞으로 답변은 한국어로 해줘" in model.messages[1]["content"]


def test_memory_service_stores_candidates_only_in_selected_context(tmp_path) -> None:
    registry = _registry(tmp_path)
    service = MemoryService(InMemorySessionMemoryRepository(), registry)
    candidates = [
        MemoryCandidate(
            content="사용자는 한국어 답변을 선호한다.",
            category="preference",
            confidence=0.95,
        ),
        MemoryCandidate(
            content="사용자는 주간 보고서를 간결하게 작성한다.",
            category="workflow",
            confidence=0.92,
        ),
    ]

    first = asyncio.run(service.remember(
        user_id="user-1", memory_context="context1", candidates=candidates
    ))
    duplicate = asyncio.run(service.remember(
        user_id="user-1", memory_context="context1", candidates=candidates
    ))

    assert {item.metadata["category"] for item in first} == {
        "preference", "workflow"
    }
    assert duplicate == []
    assert asyncio.run(registry.get("context3").list("user-1")) == []
    context1_items = asyncio.run(registry.get("context1").list("user-1"))
    assert len(context1_items) == 2


def test_extractor_skips_secret_bearing_turn_before_llm_call() -> None:
    model = ExtractionModel(MemoryExtraction(memories=[
        MemoryCandidate(
            content="database_password의 값은 p@ssw0rd!23입니다.",
            category="profile",
            confidence=0.99,
        ),
    ]))
    extractor = LlmMemoryExtractor(model)

    candidates = asyncio.run(extractor.extract(
        session_turns=[],
        query="secret namespace에서 database_password를 찾아줘",
        answer="database_password의 값은 p@ssw0rd!23입니다.",
    ))

    assert candidates == []
    assert model.messages is None


def test_memory_service_rejects_sensitive_candidate_and_loaded_context(tmp_path) -> None:
    registry = _registry(tmp_path)
    sessions = InMemorySessionMemoryRepository()
    service = MemoryService(sessions, registry)
    sensitive_candidate = MemoryCandidate(
        content="database_password의 값은 p@ssw0rd!23입니다.",
        category="profile",
        confidence=0.99,
    )

    stored = asyncio.run(service.remember(
        user_id="user-1",
        memory_context="context1",
        candidates=[sensitive_candidate],
    ))
    asyncio.run(registry.get("context1").save(
        "user-1",
        MemoryItem(
            memory_id="contaminated",
            content="database_password의 값은 p@ssw0rd!23입니다.",
        ),
    ))
    asyncio.run(sessions.append(
        "user-1",
        "session-1",
        ConversationTurn(role="user", content="test_api_key가 뭐야?"),
    ))
    asyncio.run(sessions.append(
        "user-1",
        "session-1",
        ConversationTurn(role="assistant", content="값은 opaque-value-123입니다."),
    ))

    context = asyncio.run(service.load_context(
        user_id="user-1",
        session_id="session-1",
        query="database_password",
        memory_context="context1",
    ))

    assert stored == []
    assert context.memories == []
    assert context.session_turns == []
