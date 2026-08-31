from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chromadb

from agent_system.infrastructure.embeddings import EmbeddingClient


@dataclass(frozen=True)
class VectorRecord:
    record_id: str
    text: str
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)


@dataclass(frozen=True)
class VectorHit:
    record_id: str
    text: str
    metadata: dict[str, Any]
    score: float


class ChromaVectorIndex:
    def __init__(
        self,
        *,
        persist_directory: Path,
        collection_name: str,
        embedding: EmbeddingClient,
        batch_size: int = 500,
    ) -> None:
        persist_directory.mkdir(parents=True, exist_ok=True)
        self._embedding = embedding
        self._batch_size = max(1, batch_size)
        self._client = chromadb.PersistentClient(path=str(persist_directory))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={
                "embedding_model": embedding.model,
                "hnsw:space": "cosine",
            },
        )

    def count(self) -> int:
        return self._collection.count()

    def index_missing(self, records: list[VectorRecord]) -> int:
        existing = set(self._collection.get(include=[]).get("ids", []))
        pending = [record for record in records if record.record_id not in existing]
        for start in range(0, len(pending), self._batch_size):
            self.upsert(pending[start : start + self._batch_size])
        return len(pending)

    def upsert(self, records: list[VectorRecord]) -> None:
        if not records:
            return
        embeddings = self._embedding.embed([record.text for record in records])
        self._collection.upsert(
            ids=[record.record_id for record in records],
            embeddings=embeddings,
            documents=[record.text for record in records],
            metadatas=[record.metadata for record in records],
        )

    def delete(self, record_id: str) -> None:
        self._collection.delete(ids=[record_id])

    def search(
        self,
        query: str,
        *,
        limit: int,
        where: dict[str, Any] | None = None,
    ) -> list[VectorHit]:
        if self.count() == 0:
            return []
        query_embedding = self._embedding.embed([query])[0]
        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(limit, self.count()),
            "include": ["documents", "metadatas", "distances"],
        }
        if where is not None:
            kwargs["where"] = where
        response = self._collection.query(**kwargs)
        ids = (response.get("ids") or [[]])[0]
        documents = (response.get("documents") or [[]])[0]
        metadatas = (response.get("metadatas") or [[]])[0]
        distances = (response.get("distances") or [[]])[0]
        return [
            VectorHit(
                record_id=str(record_id),
                text=str(text or ""),
                metadata=dict(metadata or {}),
                score=round(1.0 / (1.0 + max(float(distance), 0.0)), 6),
            )
            for record_id, text, metadata, distance in zip(
                ids,
                documents,
                metadatas,
                distances,
                strict=True,
            )
        ]

