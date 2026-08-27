from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Protocol
from uuid import uuid4

from pydantic import BaseModel, Field


class EmailMessage(BaseModel):
    message_id: str
    mailbox: Literal["inbox", "sent"]
    sender: str
    recipients: list[str]
    subject: str
    body: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class GmailGateway(Protocol):
    async def get(self, message_id: str) -> EmailMessage | None: ...

    async def list_messages(self, mailbox: str, limit: int) -> list[EmailMessage]: ...

    async def search(self, mailbox: str, query: str, limit: int) -> list[EmailMessage]: ...

    async def send(
        self,
        sender: str,
        recipients: list[str],
        subject: str,
        body: str,
    ) -> EmailMessage: ...

    async def delete(self, message_id: str) -> bool: ...


class InMemoryGmailGateway:
    def __init__(self, messages: list[EmailMessage] | None = None) -> None:
        self._messages = {message.message_id: message for message in (messages or [])}

    async def get(self, message_id: str) -> EmailMessage | None:
        return self._messages.get(message_id)

    async def list_messages(self, mailbox: str, limit: int) -> list[EmailMessage]:
        return [
            message for message in self._messages.values()
            if message.mailbox == mailbox
        ][:limit]

    async def search(self, mailbox: str, query: str, limit: int) -> list[EmailMessage]:
        needle = query.casefold()
        return [
            message for message in self._messages.values()
            if message.mailbox == mailbox
            and needle in (
                f"{message.sender} {' '.join(message.recipients)} "
                f"{message.subject} {message.body}"
            ).casefold()
        ][:limit]

    async def send(
        self,
        sender: str,
        recipients: list[str],
        subject: str,
        body: str,
    ) -> EmailMessage:
        message = EmailMessage(
            message_id=str(uuid4()),
            mailbox="sent",
            sender=sender,
            recipients=recipients,
            subject=subject,
            body=body,
        )
        self._messages[message.message_id] = message
        return message

    async def delete(self, message_id: str) -> bool:
        return self._messages.pop(message_id, None) is not None

