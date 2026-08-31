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


class LongTermMemoryContextProvider(Protocol):
    def get(self, context_id: str) -> LongTermMemoryRepository: ...

    def list_contexts(self) -> list[str]: ...


class LongTermMemoryRepository(Protocol):
    async def list(self, user_id: str) -> list[MemoryItem]: ...

    async def create(
        self, user_id: str, content: str, metadata: dict[str, Any]
    ) -> MemoryItem: ...

    async def search(
        self,
        user_id: str,
        query: str,
        limit: int,
    ) -> list[MemoryItem]: ...

    async def save(self, user_id: str, item: MemoryItem) -> None: ...

    async def update(self, user_id: str, item: MemoryItem) -> None: ...

    async def delete(self, user_id: str, memory_id: str) -> None: ...

