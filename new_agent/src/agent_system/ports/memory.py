from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ConversationTurn:
    role: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryItem:
    memory_id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OrchestrationContext:
    session_turns: list[ConversationTurn]
    memories: list[MemoryItem]


class SessionMemoryRepository(Protocol):
    async def load(self, user_id: str, session_id: str) -> list[ConversationTurn]: ...

    async def append(
        self,
        user_id: str,
        session_id: str,
        turn: ConversationTurn,
    ) -> None: ...


class LongTermMemoryRepository(Protocol):
    async def search(
        self,
        user_id: str,
        query: str,
        limit: int,
    ) -> list[MemoryItem]: ...

    async def save(self, user_id: str, item: MemoryItem) -> None: ...

    async def update(self, user_id: str, item: MemoryItem) -> None: ...

    async def delete(self, user_id: str, memory_id: str) -> None: ...

