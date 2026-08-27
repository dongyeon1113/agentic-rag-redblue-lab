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


class InMemoryDriveGateway:
    def __init__(self, items: list[DriveItem] | None = None) -> None:
        self._items = {item.item_id: item for item in (items or [])}

    async def get(self, item_id: str) -> DriveItem | None:
        return self._items.get(item_id)

    async def search(
        self,
        query: str,
        parent_id: str | None,
        limit: int,
    ) -> list[DriveItem]:
        needle = query.casefold()
        return [
            item for item in self._items.values()
            if (parent_id is None or item.parent_id == parent_id)
            and needle in f"{item.name} {item.content or ''}".casefold()
        ][:limit]

    async def create(
        self,
        item_type: str,
        name: str,
        parent_id: str | None,
        content: str | None,
    ) -> DriveItem:
        item = DriveItem(
            item_id=str(uuid4()),
            item_type=item_type,
            name=name,
            parent_id=parent_id,
            content=content,
        )
        self._items[item.item_id] = item
        return item

    async def move(
        self,
        item_id: str,
        expected_parent_id: str | None,
        destination_parent_id: str,
    ) -> DriveItem:
        item = self._items.get(item_id)
        if item is None:
            raise LookupError(f"Drive item not found: {item_id}")
        if item.parent_id != expected_parent_id:
            raise ValueError("Drive item parent changed before the move")
        updated = item.model_copy(update={"parent_id": destination_parent_id})
        self._items[item_id] = updated
        return updated

    async def delete(self, item_id: str) -> bool:
        return self._items.pop(item_id, None) is not None

