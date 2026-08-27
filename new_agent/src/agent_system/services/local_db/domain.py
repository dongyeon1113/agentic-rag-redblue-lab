from __future__ import annotations

from typing import Literal, Protocol

from pydantic import BaseModel, Field


class Document(BaseModel):
    document_id: str
    namespace: Literal["knowledge", "secret"] = "knowledge"
    title: str
    content: str
    metadata: dict[str, str] = Field(default_factory=dict)


class DocumentRepository(Protocol):
    async def get(self, document_id: str) -> Document | None: ...

    async def search(
        self,
        query: str,
        namespace: str,
        limit: int,
    ) -> list[Document]: ...

    async def create(self, document: Document) -> Document: ...

    async def update(
        self,
        document_id: str,
        changes: dict[str, object],
    ) -> Document: ...

    async def delete(self, document_id: str) -> bool: ...


class InMemoryDocumentRepository:
    def __init__(self, documents: list[Document] | None = None) -> None:
        self._documents = {
            document.document_id: document
            for document in (documents or [])
        }

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
        if document.document_id in self._documents:
            raise ValueError(f"Document already exists: {document.document_id}")
        self._documents[document.document_id] = document
        return document

    async def update(
        self,
        document_id: str,
        changes: dict[str, object],
    ) -> Document:
        current = self._documents.get(document_id)
        if current is None:
            raise LookupError(f"Document not found: {document_id}")
        updated = current.model_copy(update=changes)
        self._documents[document_id] = updated
        return updated

    async def delete(self, document_id: str) -> bool:
        return self._documents.pop(document_id, None) is not None

