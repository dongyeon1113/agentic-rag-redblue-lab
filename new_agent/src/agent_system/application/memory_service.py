from __future__ import annotations

from agent_system.ports.memory import (
    ConversationTurn,
    LongTermMemoryRepository,
    OrchestrationContext,
    SessionMemoryRepository,
)


class MemoryService:
    def __init__(
        self,
        sessions: SessionMemoryRepository,
        long_term: LongTermMemoryRepository,
    ) -> None:
        self._sessions = sessions
        self._long_term = long_term

    async def load_context(
        self,
        *,
        user_id: str,
        session_id: str,
        query: str,
    ) -> OrchestrationContext:
        session_turns = await self._sessions.load(user_id, session_id)
        memories = await self._long_term.search(user_id, query, limit=5)
        return OrchestrationContext(session_turns=session_turns, memories=memories)

    async def record_turn(
        self,
        *,
        user_id: str,
        session_id: str,
        query: str,
        answer: str,
    ) -> None:
        await self._sessions.append(
            user_id, session_id, ConversationTurn(role="user", content=query)
        )
        await self._sessions.append(
            user_id, session_id, ConversationTurn(role="assistant", content=answer)
        )

