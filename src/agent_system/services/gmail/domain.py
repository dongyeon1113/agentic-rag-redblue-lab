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
