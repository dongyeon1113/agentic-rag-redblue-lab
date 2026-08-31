from __future__ import annotations

from typing import Literal, Protocol
from uuid import uuid4

from pydantic import BaseModel, Field


class DriveItem(BaseModel):
    item_id: str
    item_type: Literal["file", "folder"]
    name: str
    parent_id: str | None = None
    content: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class DriveGateway(Protocol):
    async def get(self, item_id: str) -> DriveItem | None: ...

    async def search(
        self,
        query: str,
        parent_id: str | None,
        limit: int,
    ) -> list[DriveItem]: ...

    async def create(
        self,
        item_type: str,
        name: str,
        parent_id: str | None,
        content: str | None,
    ) -> DriveItem: ...

    async def move(
        self,
        item_id: str,
        expected_parent_id: str | None,
        destination_parent_id: str,
    ) -> DriveItem: ...

    async def delete(self, item_id: str) -> bool: ...
