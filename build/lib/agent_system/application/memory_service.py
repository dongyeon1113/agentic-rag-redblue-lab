from __future__ import annotations

from uuid import uuid4

from agent_system.application.memory_extraction import (
    MemoryCandidate,
    contains_sensitive_content,
)
from agent_system.ports.memory import (
    ConversationTurn,
    LongTermMemoryContextProvider,
    LongTermMemoryRepository,
    MemoryItem,
    OrchestrationContext,
    SessionMemoryRepository,
)


class MemoryService:
    def __init__(
        self,
        sessions: SessionMemoryRepository,
        long_term_contexts: LongTermMemoryContextProvider,
    ) -> None:
        self._sessions = sessions
        self._long_term_contexts = long_term_contexts

    def long_term(self, memory_context: str) -> LongTermMemoryRepository:
        return self._long_term_contexts.get(memory_context)

    def list_contexts(self) -> list[str]:
        return self._long_term_contexts.list_contexts()

    async def load_context(
        self,
        *,
        user_id: str,
        session_id: str,
        query: str,
        memory_context: str,
    ) -> OrchestrationContext:
        session_turns = await self._sessions.load(user_id, session_id)
        memories = await self.long_term(memory_context).search(user_id, query, limit=5)
        return OrchestrationContext(
            session_turns=self._safe_session_turns(session_turns),
            memories=[
                item for item in memories
                if not contains_sensitive_content(item.content)
            ],
        )

    async def remember(
        self,
        *,
        user_id: str,
        memory_context: str,
        candidates: list[MemoryCandidate],
    ) -> list[MemoryItem]:
        repository = self.long_term(memory_context)
        existing = await repository.list(user_id)
        normalized = {self._normalize(item.content) for item in existing}
        stored: list[MemoryItem] = []
        for candidate in candidates:
            if contains_sensitive_content(candidate.content):
                continue
            key = self._normalize(candidate.content)
            if not key or key in normalized:
                continue
            item = MemoryItem(
                memory_id=f"memory-{uuid4()}",
                content=candidate.content.strip(),
                metadata={
                    "category": candidate.category,
                    "confidence": candidate.confidence,
                    "source": "llm_conversation_extraction",
                    "trust": "user_conversation",
                },
            )
            await repository.save(user_id, item)
            normalized.add(key)
            stored.append(item)
        return stored

    async def record_turn(
        self,
        *,
        user_id: str,
        session_id: str,
        query: str,
        answer: str,
        sensitive: bool = False,
    ) -> None:
        if (
            sensitive
            or contains_sensitive_content(query)
            or contains_sensitive_content(answer)
        ):
            return
        await self._sessions.append(
            user_id, session_id, ConversationTurn(role="user", content=query)
        )
        await self._sessions.append(
            user_id, session_id, ConversationTurn(role="assistant", content=answer)
        )

    @staticmethod
    def _safe_session_turns(
        turns: list[ConversationTurn],
    ) -> list[ConversationTurn]:
        safe: list[ConversationTurn] = []
        sensitive_exchange = False
        for turn in turns:
            sensitive_turn = (
                bool(turn.metadata.get("sensitive"))
                or contains_sensitive_content(turn.content)
            )
            if turn.role == "user":
                sensitive_exchange = sensitive_turn
                if not sensitive_turn:
                    safe.append(turn)
                continue
            if turn.role == "assistant":
                if not sensitive_exchange and not sensitive_turn:
                    safe.append(turn)
                sensitive_exchange = False
                continue
            if not sensitive_turn:
                safe.append(turn)
        return safe

    @staticmethod
    def _normalize(content: str) -> str:
        return " ".join(content.casefold().split())
