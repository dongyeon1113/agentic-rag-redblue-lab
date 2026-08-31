from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import uuid4

from agent_system.infrastructure.json_files import read_json, write_json_atomic
from agent_system.services.gmail.domain import EmailMessage


class JsonGmailGateway:
    def __init__(
        self,
        *,
        inbox_file: Path,
        sent_file: Path,
        inbox_seed_file: Path,
        sent_seed_file: Path,
    ) -> None:
        self._inbox_file = inbox_file
        self._sent_file = sent_file
        self._lock = asyncio.Lock()
        self._initialize(inbox_file, inbox_seed_file)
        self._initialize(sent_file, sent_seed_file)
        self._inbox = self._load(inbox_file, "inbox")
        self._sent = self._load(sent_file, "sent")

    @staticmethod
    def _initialize(data_file: Path, seed_file: Path) -> None:
        if not data_file.exists():
            write_json_atomic(data_file, read_json(seed_file))

    @staticmethod
    def _load(path: Path, expected_mailbox: str) -> dict[str, EmailMessage]:
        messages = [EmailMessage.model_validate(item) for item in read_json(path)]
        if any(message.mailbox != expected_mailbox for message in messages):
            raise ValueError(f"Unexpected mailbox value in {path}")
        return {message.message_id: message for message in messages}

    async def _persist_inbox(self) -> None:
        write_json_atomic(
            self._inbox_file,
            [message.model_dump(mode="json") for message in self._inbox.values()],
        )

    async def _persist_sent(self) -> None:
        write_json_atomic(
            self._sent_file,
            [message.model_dump(mode="json") for message in self._sent.values()],
        )

    async def get(self, message_id: str) -> EmailMessage | None:
        return self._inbox.get(message_id) or self._sent.get(message_id)

    async def list_messages(self, mailbox: str, limit: int) -> list[EmailMessage]:
        source = self._inbox if mailbox == "inbox" else self._sent
        return list(source.values())[:limit]

    async def send(
        self,
        sender: str,
        recipients: list[str],
        subject: str,
        body: str,
    ) -> EmailMessage:
        message = EmailMessage(
            message_id=f"sent-{uuid4()}",
            mailbox="sent",
            sender=sender,
            recipients=recipients,
            subject=subject,
            body=body,
        )
        async with self._lock:
            self._sent[message.message_id] = message
            await self._persist_sent()
        return message

    async def delete(self, message_id: str) -> bool:
        async with self._lock:
            deleted = self._inbox.pop(message_id, None) is not None
            if deleted:
                await self._persist_inbox()
                return True
            deleted = self._sent.pop(message_id, None) is not None
            if deleted:
                await self._persist_sent()
            return deleted

