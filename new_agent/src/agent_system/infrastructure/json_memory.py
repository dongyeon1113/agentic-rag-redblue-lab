from __future__ import annotations

import asyncio
import re
from pathlib import Path
from uuid import uuid4

from agent_system.infrastructure.json_files import read_json, write_json_atomic
from agent_system.ports.memory import ConversationTurn, MemoryItem


class JsonSessionMemoryRepository:
    def __init__(self, data_file: Path) -> None:
        self._data_file = data_file
        self._lock = asyncio.Lock()
        if not data_file.exists():
            write_json_atomic(data_file, {})
        self._sessions: dict[str, list[dict]] = read_json(data_file)

    @staticmethod
    def _key(user_id: str, session_id: str) -> str:
        return f"{user_id}:{session_id}"

    async def load(self, user_id: str, session_id: str) -> list[ConversationTurn]:
        return [
            ConversationTurn(**item)
            for item in self._sessions.get(self._key(user_id, session_id), [])
        ]

    async def append(
        self,
        user_id: str,
        session_id: str,
        turn: ConversationTurn,
    ) -> None:
        key = self._key(user_id, session_id)
        async with self._lock:
            self._sessions.setdefault(key, []).append({
                "role": turn.role,
                "content": turn.content,
                "metadata": turn.metadata,
            })
            write_json_atomic(self._data_file, self._sessions)


class JsonLongTermMemoryRepository:
    def __init__(self, data_file: Path) -> None:
        self._data_file = data_file
        self._lock = asyncio.Lock()
        if not data_file.exists():
            write_json_atomic(data_file, {})
        self._items: dict[str, dict[str, dict]] = read_json(data_file)

    async def search(
        self,
        user_id: str,
        query: str,
        limit: int,
    ) -> list[MemoryItem]:
        terms = {
            term for term in re.findall(r"[\w가-힣]+", query.casefold())
            if len(term) > 1
        }
        scored: list[tuple[int, MemoryItem]] = []
        for raw in self._items.get(user_id, {}).values():
            item = MemoryItem(**raw)
            content = item.content.casefold()
            score = sum(term in content for term in terms)
            if score:
                scored.append((score, item))
        scored.sort(key=lambda pair: (-pair[0], pair[1].memory_id))
        return [item for _, item in scored[:limit]]

    async def list(self, user_id: str) -> list[MemoryItem]:
        return [
            MemoryItem(**raw)
            for raw in self._items.get(user_id, {}).values()
        ]

    async def save(self, user_id: str, item: MemoryItem) -> None:
        async with self._lock:
            user_items = self._items.setdefault(user_id, {})
            if item.memory_id in user_items:
                raise ValueError(f"Memory already exists: {item.memory_id}")
            user_items[item.memory_id] = {
                "memory_id": item.memory_id,
                "content": item.content,
                "metadata": item.metadata,
            }
            write_json_atomic(self._data_file, self._items)

    async def create(
        self,
        user_id: str,
        content: str,
        metadata: dict,
    ) -> MemoryItem:
        item = MemoryItem(
            memory_id=f"memory-{uuid4()}",
            content=content,
            metadata=metadata,
        )
        await self.save(user_id, item)
        return item

    async def update(self, user_id: str, item: MemoryItem) -> None:
        async with self._lock:
            if item.memory_id not in self._items.get(user_id, {}):
                raise LookupError(f"Memory not found: {item.memory_id}")
            self._items[user_id][item.memory_id] = {
                "memory_id": item.memory_id,
                "content": item.content,
                "metadata": item.metadata,
            }
            write_json_atomic(self._data_file, self._items)

    async def delete(self, user_id: str, memory_id: str) -> None:
        async with self._lock:
            self._items.get(user_id, {}).pop(memory_id, None)
            write_json_atomic(self._data_file, self._items)

