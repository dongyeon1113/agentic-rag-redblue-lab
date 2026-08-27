from __future__ import annotations

import asyncio
import json
from pathlib import Path

from agent_system.infrastructure.json_files import read_json, write_json_atomic
from agent_system.services.local_db.domain import Document


class JsonDocumentRepository:
    """Mutable JSON working copy initialized from NQ and secret seed files."""

    def __init__(
        self,
        *,
        data_file: Path,
        knowledge_seed_file: Path,
        secret_seed_file: Path,
    ) -> None:
        self._data_file = data_file
        self._lock = asyncio.Lock()
        if not data_file.exists():
            documents = self._documents_from_seeds(
                knowledge_seed_file,
                secret_seed_file,
            )
            write_json_atomic(
                data_file,
                [document.model_dump(mode="json") for document in documents],
            )
        self._documents = {
            document.document_id: document
            for document in (
                Document.model_validate(item) for item in read_json(data_file)
            )
        }

    @staticmethod
    def _documents_from_seeds(
        knowledge_seed_file: Path,
        secret_seed_file: Path,
    ) -> list[Document]:
        knowledge = []
        for item in read_json(knowledge_seed_file):
            text = str(item["text"])
            title = text.splitlines()[0].strip() or str(item["id"])
            knowledge.append(
                Document(
                    document_id=str(item["id"]),
                    namespace="knowledge",
                    title=title,
                    content=text,
                    metadata={
                        "source": str(item.get("source", "nq")),
                        "trust": str(item.get("trust", "trusted")),
                        "tags": json.dumps(item.get("tags", []), ensure_ascii=False),
                    },
                )
            )

        secret_payload = read_json(secret_seed_file)
        secrets = [
            Document(
                document_id=f"secret-{name}",
                namespace="secret",
                title=name,
                content=str(value),
                metadata={"source": "mock-secret", "trust": "secret"},
            )
            for name, value in secret_payload.get("secrets", {}).items()
        ]
        return knowledge + secrets

    async def _persist(self) -> None:
        write_json_atomic(
            self._data_file,
            [document.model_dump(mode="json") for document in self._documents.values()],
        )

    async def get(self, document_id: str) -> Document | None:
        return self._documents.get(document_id)

    async def search(
        self,
        query: str,
        namespace: str,
        limit: int,
    ) -> list[Document]:
        needle = query.casefold()
        matches = [
            document
            for document in self._documents.values()
            if document.namespace == namespace
            and needle in f"{document.title}\n{document.content}".casefold()
        ]
        return matches[:limit]

    async def create(self, document: Document) -> Document:
        async with self._lock:
            if document.document_id in self._documents:
                raise ValueError(f"Document already exists: {document.document_id}")
            self._documents[document.document_id] = document
            await self._persist()
        return document

    async def update(
        self,
        document_id: str,
        changes: dict[str, object],
    ) -> Document:
        async with self._lock:
            current = self._documents.get(document_id)
            if current is None:
                raise LookupError(f"Document not found: {document_id}")
            updated = current.model_copy(update=changes)
            self._documents[document_id] = updated
            await self._persist()
        return updated

    async def delete(self, document_id: str) -> bool:
        async with self._lock:
            deleted = self._documents.pop(document_id, None) is not None
            if deleted:
                await self._persist()
        return deleted

