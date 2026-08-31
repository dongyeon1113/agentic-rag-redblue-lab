from __future__ import annotations

import asyncio
from pathlib import Path

from agent_system.infrastructure.chroma_index import ChromaVectorIndex, VectorRecord
from agent_system.infrastructure.embeddings import EmbeddingClient
from agent_system.services.local_db.domain import Document
from agent_system.services.local_db.json_repository import JsonDocumentRepository


class VectorDocumentRepository(JsonDocumentRepository):
    def __init__(
        self,
        *,
        data_file: Path,
        knowledge_seed_file: Path,
        secret_seed_file: Path,
        persist_directory: Path,
        embedding: EmbeddingClient,
        collection_name: str = "local-db",
        batch_size: int = 500,
    ) -> None:
        super().__init__(
            data_file=data_file,
            knowledge_seed_file=knowledge_seed_file,
            secret_seed_file=secret_seed_file,
        )
        self._index = ChromaVectorIndex(
            persist_directory=persist_directory,
            collection_name=collection_name,
            embedding=embedding,
            batch_size=batch_size,
        )
        self._index.index_missing([
            self._record(document) for document in self._documents.values()
        ])

    @staticmethod
    def _record(document: Document) -> VectorRecord:
        return VectorRecord(
            record_id=document.document_id,
            text=(
                document.content
                if document.content.splitlines()
                and document.content.splitlines()[0].strip() == document.title.strip()
                else f"{document.title}\n\n{document.content}"
            ),
            metadata={
                "namespace": document.namespace,
                "title": document.title,
            },
        )

    async def search(
        self,
        query: str,
        namespace: str,
        limit: int,
    ) -> list[Document]:
        hits = await asyncio.to_thread(
            self._index.search,
            query,
            limit=limit,
            where={"namespace": namespace},
        )
        return [
            self._documents[hit.record_id]
            for hit in hits
            if hit.record_id in self._documents
        ]

    async def create(self, document: Document) -> Document:
        created = await super().create(document)
        await asyncio.to_thread(self._index.upsert, [self._record(created)])
        return created

    async def update(
        self,
        document_id: str,
        changes: dict[str, object],
    ) -> Document:
        updated = await super().update(document_id, changes)
        await asyncio.to_thread(self._index.upsert, [self._record(updated)])
        return updated

    async def delete(self, document_id: str) -> bool:
        deleted = await super().delete(document_id)
        if deleted:
            await asyncio.to_thread(self._index.delete, document_id)
        return deleted

