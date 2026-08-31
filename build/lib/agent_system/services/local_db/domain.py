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
