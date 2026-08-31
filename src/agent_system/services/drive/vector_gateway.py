from __future__ import annotations

import asyncio
from pathlib import Path

from agent_system.infrastructure.chroma_index import ChromaVectorIndex, VectorRecord
from agent_system.infrastructure.embeddings import EmbeddingClient
from agent_system.services.drive.domain import DriveItem
from agent_system.services.drive.json_gateway import JsonDriveGateway


class VectorDriveGateway(JsonDriveGateway):
    def __init__(
        self,
        *,
        data_file: Path,
        seed_file: Path,
        persist_directory: Path,
        embedding: EmbeddingClient,
        collection_name: str = "drive",
    ) -> None:
        super().__init__(data_file=data_file, seed_file=seed_file)
        self._index = ChromaVectorIndex(
            persist_directory=persist_directory,
            collection_name=collection_name,
            embedding=embedding,
        )
        self._index.index_missing([
            self._record(item) for item in self._items.values()
        ])

    @staticmethod
    def _record(item: DriveItem) -> VectorRecord:
        return VectorRecord(
            record_id=item.item_id,
            text=f"Name: {item.name}\nType: {item.item_type}\n\n{item.content or ''}",
            metadata={
                "item_type": item.item_type,
                "parent_id": item.parent_id or "__root__",
            },
        )

    async def search(
        self,
        query: str,
        parent_id: str | None,
        limit: int,
    ) -> list[DriveItem]:
        where = None
        if parent_id is not None:
            where = {"parent_id": parent_id}
        hits = await asyncio.to_thread(
            self._index.search,
            query,
            limit=limit,
            where=where,
        )
        return [
            self._items[hit.record_id]
            for hit in hits
            if hit.record_id in self._items
        ]

    async def create(
        self,
        item_type: str,
        name: str,
        parent_id: str | None,
        content: str | None,
    ) -> DriveItem:
        item = await super().create(item_type, name, parent_id, content)
        await asyncio.to_thread(self._index.upsert, [self._record(item)])
        return item

    async def move(
        self,
        item_id: str,
        expected_parent_id: str | None,
        destination_parent_id: str,
    ) -> DriveItem:
        item = await super().move(
            item_id,
            expected_parent_id,
            destination_parent_id,
        )
        await asyncio.to_thread(self._index.upsert, [self._record(item)])
        return item

    async def delete(self, item_id: str) -> bool:
        deleted = await super().delete(item_id)
        if deleted:
            await asyncio.to_thread(self._index.delete, item_id)
        return deleted

