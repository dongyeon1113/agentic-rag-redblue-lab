from __future__ import annotations

import asyncio

from agent_system.infrastructure.embeddings import DeterministicEmbeddingClient
from agent_system.infrastructure.vector_memory import VectorLongTermMemoryRepository
from agent_system.ports.memory import MemoryItem


def _repository(tmp_path, collection_name="test-long-term-memory"):
    return VectorLongTermMemoryRepository(
        data_file=tmp_path / "memory/long_term.json",
        persist_directory=tmp_path / "memory/chroma",
        embedding=DeterministicEmbeddingClient(),
        collection_name=collection_name,
    )


def test_vector_memory_is_user_isolated_and_persists(tmp_path) -> None:
    repository = _repository(tmp_path)
    asyncio.run(repository.save(
        "user-1",
        MemoryItem(memory_id="preference", content="Prefers Korean responses"),
    ))
    asyncio.run(repository.save(
        "user-2",
        MemoryItem(memory_id="preference", content="Prefers English responses"),
    ))

    reloaded = _repository(tmp_path, "recovered-long-term-memory")
    user_one = asyncio.run(reloaded.search("user-1", "Korean language", 3))
    user_two = asyncio.run(reloaded.search("user-2", "Korean language", 3))

    assert [item.content for item in user_one] == ["Prefers Korean responses"]
    assert [item.content for item in user_two] == ["Prefers English responses"]


def test_vector_memory_updates_and_deletes_json_and_chroma(tmp_path) -> None:
    repository = _repository(tmp_path)
    asyncio.run(repository.save(
        "user-1",
        MemoryItem(memory_id="delivery", content="Send reports by email"),
    ))
    asyncio.run(repository.update(
        "user-1",
        MemoryItem(memory_id="delivery", content="Send reports in Slack"),
    ))

    updated = asyncio.run(repository.search("user-1", "Slack reports", 1))
    assert [item.content for item in updated] == ["Send reports in Slack"]

    asyncio.run(repository.delete("user-1", "delivery"))
    assert asyncio.run(repository.list("user-1")) == []
    assert asyncio.run(repository.search("user-1", "Slack reports", 1)) == []
