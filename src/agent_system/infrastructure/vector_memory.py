from __future__ import annotations

import asyncio
from pathlib import Path

from agent_system.infrastructure.chroma_index import ChromaVectorIndex, VectorRecord
from agent_system.infrastructure.embeddings import EmbeddingClient
from agent_system.infrastructure.json_memory import JsonLongTermMemoryRepository
from agent_system.ports.memory import MemoryItem


class VectorLongTermMemoryRepository(JsonLongTermMemoryRepository):
    """JSON-backed long-term memory with a user-isolated Chroma search index."""

    def __init__(
        self,
        *,
        data_file: Path,
        persist_directory: Path,
        embedding: EmbeddingClient,
        collection_name: str = "long-term-memory",
        batch_size: int = 500,
    ) -> None:
        super().__init__(data_file)
        self._index = ChromaVectorIndex(
            persist_directory=persist_directory,
            collection_name=collection_name,
            embedding=embedding,
            batch_size=batch_size,
        )
        self._index.index_missing([
            self._record(user_id, MemoryItem(**raw))
            for user_id, items in self._items.items()
            for raw in items.values()
        ])

    @staticmethod
    def _record_id(user_id: str, memory_id: str) -> str:
        return f"{user_id}:{memory_id}"

    @classmethod
    def _record(cls, user_id: str, item: MemoryItem) -> VectorRecord:
        return VectorRecord(
            record_id=cls._record_id(user_id, item.memory_id),
            text=item.content,
            metadata={
                "user_id": user_id,
                "memory_id": item.memory_id,
            },
        )

    async def search(
        self,
        user_id: str,
        query: str,
        limit: int,
    ) -> list[MemoryItem]:
        hits = await asyncio.to_thread(
            self._index.search,
            query,
            limit=limit,
            where={"user_id": user_id},
        )
        user_items = self._items.get(user_id, {})
        return [
            MemoryItem(**user_items[memory_id])
            for hit in hits
            if (memory_id := str(hit.metadata.get("memory_id", ""))) in user_items
        ]

    async def save(self, user_id: str, item: MemoryItem) -> None:
        await super().save(user_id, item)
        await asyncio.to_thread(self._index.upsert, [self._record(user_id, item)])

    async def update(self, user_id: str, item: MemoryItem) -> None:
        await super().update(user_id, item)
        await asyncio.to_thread(self._index.upsert, [self._record(user_id, item)])

    async def delete(self, user_id: str, memory_id: str) -> None:
        await super().delete(user_id, memory_id)
        await asyncio.to_thread(
            self._index.delete,
            self._record_id(user_id, memory_id),
        )
