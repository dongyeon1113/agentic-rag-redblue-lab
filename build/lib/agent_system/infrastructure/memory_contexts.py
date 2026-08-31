from __future__ import annotations

import re
from pathlib import Path

from agent_system.infrastructure.embeddings import EmbeddingClient
from agent_system.infrastructure.vector_memory import VectorLongTermMemoryRepository
from agent_system.ports.memory import LongTermMemoryRepository


MEMORY_CONTEXT_PATTERN = r"^context[1-9][0-9]{0,5}$"
_CONTEXT_RE = re.compile(MEMORY_CONTEXT_PATTERN)


def validate_memory_context(context_id: str) -> str:
    if not _CONTEXT_RE.fullmatch(context_id):
        raise ValueError(
            "memory_context must use the numbered form context1, context2, ..."
        )
    return context_id


class VectorMemoryContextRegistry:
    """Lazily attaches named JSON files to independent Chroma collections."""

    def __init__(
        self,
        *,
        memory_directory: Path,
        embedding: EmbeddingClient,
        collection_prefix: str = "long-term-memory",
        batch_size: int = 500,
        default_context: str = "context1",
    ) -> None:
        self._memory_directory = memory_directory
        self._contexts_directory = memory_directory / "contexts"
        self._persist_directory = memory_directory / "chroma"
        self._embedding = embedding
        self._collection_prefix = collection_prefix.rstrip("-._")
        self._batch_size = batch_size
        self._default_context = validate_memory_context(default_context)
        self._repositories: dict[str, VectorLongTermMemoryRepository] = {}

    def get(self, context_id: str) -> LongTermMemoryRepository:
        context_id = validate_memory_context(context_id)
        repository = self._repositories.get(context_id)
        if repository is not None:
            return repository
        data_file = (
            self._memory_directory / "long_term.json"
            if context_id == self._default_context
            else self._contexts_directory / f"{context_id}.json"
        )
        repository = VectorLongTermMemoryRepository(
            data_file=data_file,
            persist_directory=self._persist_directory,
            embedding=self._embedding,
            collection_name=f"{self._collection_prefix}-{context_id}",
            batch_size=self._batch_size,
        )
        self._repositories[context_id] = repository
        return repository

    def list_contexts(self) -> list[str]:
        names = {self._default_context, *self._repositories}
        if self._contexts_directory.exists():
            names.update(path.stem for path in self._contexts_directory.glob("*.json"))
        return sorted(name for name in names if _CONTEXT_RE.fullmatch(name))
