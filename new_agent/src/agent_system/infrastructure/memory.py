from __future__ import annotations

from collections import defaultdict

from agent_system.ports.memory import ConversationTurn, MemoryItem


class InMemorySessionMemoryRepository:
    def __init__(self) -> None:
        self._turns: dict[tuple[str, str], list[ConversationTurn]] = defaultdict(list)

    async def load(self, user_id: str, session_id: str) -> list[ConversationTurn]:
        return list(self._turns[(user_id, session_id)])

    async def append(
        self,
        user_id: str,
        session_id: str,
        turn: ConversationTurn,
    ) -> None:
        self._turns[(user_id, session_id)].append(turn)


class InMemoryLongTermMemoryRepository:
    def __init__(self) -> None:
        self._items: dict[str, dict[str, MemoryItem]] = defaultdict(dict)

    async def search(
        self,
        user_id: str,
        query: str,
        limit: int,
    ) -> list[MemoryItem]:
        needle = query.casefold()
        matches = [
            item for item in self._items[user_id].values()
            if needle in item.content.casefold()
        ]
        return matches[:limit]

    async def save(self, user_id: str, item: MemoryItem) -> None:
        self._items[user_id][item.memory_id] = item

    async def update(self, user_id: str, item: MemoryItem) -> None:
        if item.memory_id not in self._items[user_id]:
            raise LookupError(f"Memory not found: {item.memory_id}")
        self._items[user_id][item.memory_id] = item

    async def delete(self, user_id: str, memory_id: str) -> None:
        self._items[user_id].pop(memory_id, None)

