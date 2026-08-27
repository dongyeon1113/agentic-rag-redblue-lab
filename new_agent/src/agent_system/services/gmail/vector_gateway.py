from __future__ import annotations

import asyncio
from pathlib import Path

from agent_system.infrastructure.chroma_index import ChromaVectorIndex, VectorRecord
from agent_system.infrastructure.embeddings import EmbeddingClient
from agent_system.services.gmail.domain import EmailMessage
from agent_system.services.gmail.json_gateway import JsonGmailGateway


class VectorGmailGateway(JsonGmailGateway):
    def __init__(
        self,
        *,
        inbox_file: Path,
        sent_file: Path,
        inbox_seed_file: Path,
        sent_seed_file: Path,
        persist_directory: Path,
        embedding: EmbeddingClient,
        collection_name: str = "gmail",
    ) -> None:
        super().__init__(
            inbox_file=inbox_file,
            sent_file=sent_file,
            inbox_seed_file=inbox_seed_file,
            sent_seed_file=sent_seed_file,
        )
        self._index = ChromaVectorIndex(
            persist_directory=persist_directory,
            collection_name=collection_name,
            embedding=embedding,
        )
        self._index.index_missing([
            self._record(message)
            for message in [*self._inbox.values(), *self._sent.values()]
        ])

    @staticmethod
    def _record(message: EmailMessage) -> VectorRecord:
        return VectorRecord(
            record_id=message.message_id,
            text=(
                f"From: {message.sender}\n"
                f"To: {', '.join(message.recipients)}\n"
                f"Subject: {message.subject}\n\n{message.body}"
            ),
            metadata={"mailbox": message.mailbox},
        )

    async def search(self, mailbox: str, query: str, limit: int) -> list[EmailMessage]:
        hits = await asyncio.to_thread(
            self._index.search,
            query,
            limit=limit,
            where={"mailbox": mailbox},
        )
        source = self._inbox if mailbox == "inbox" else self._sent
        return [source[hit.record_id] for hit in hits if hit.record_id in source]

    async def send(
        self,
        sender: str,
        recipients: list[str],
        subject: str,
        body: str,
    ) -> EmailMessage:
        message = await super().send(sender, recipients, subject, body)
        await asyncio.to_thread(self._index.upsert, [self._record(message)])
        return message

    async def delete(self, message_id: str) -> bool:
        deleted = await super().delete(message_id)
        if deleted:
            await asyncio.to_thread(self._index.delete, message_id)
        return deleted

