from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import uuid4

from agent_system.infrastructure.json_files import read_json, write_json_atomic
from agent_system.services.drive.domain import DriveItem


class JsonDriveGateway:
    def __init__(self, *, data_file: Path, seed_file: Path) -> None:
        self._data_file = data_file
        self._lock = asyncio.Lock()
        if not data_file.exists():
            write_json_atomic(data_file, read_json(seed_file))
        self._items = {
            item.item_id: item
            for item in (DriveItem.model_validate(value) for value in read_json(data_file))
        }

    async def _persist(self) -> None:
        write_json_atomic(
            self._data_file,
            [item.model_dump(mode="json") for item in self._items.values()],
        )

    async def get(self, item_id: str) -> DriveItem | None:
        return self._items.get(item_id)

    async def create(
        self,
        item_type: str,
        name: str,
        parent_id: str | None,
        content: str | None,
    ) -> DriveItem:
        item = DriveItem(
            item_id=f"drive-{uuid4()}",
            item_type=item_type,
            name=name,
            parent_id=parent_id,
            content=content,
        )
        async with self._lock:
            self._items[item.item_id] = item
            await self._persist()
        return item

    async def move(
        self,
        item_id: str,
        expected_parent_id: str | None,
        destination_parent_id: str,
    ) -> DriveItem:
        async with self._lock:
            item = self._items.get(item_id)
            if item is None:
                raise LookupError(f"Drive item not found: {item_id}")
            if item.parent_id != expected_parent_id:
                raise ValueError("Drive item parent changed before the move")
            updated = item.model_copy(update={"parent_id": destination_parent_id})
            self._items[item_id] = updated
            await self._persist()
        return updated

    async def delete(self, item_id: str) -> bool:
        async with self._lock:
            deleted = self._items.pop(item_id, None) is not None
            if deleted:
                await self._persist()
        return deleted

