from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from agent_system.contracts import Capability, Principal, RiskLevel
from agent_system.services.gmail.domain import GmailGateway
from agent_system.tool_runtime.handler import ToolHandler


def _capability(
    action: str,
    description: str,
    permission: str,
    risk: RiskLevel,
    model: type[BaseModel],
) -> Capability:
    return Capability(
        executor="gmail",
        action=action,
        description=description,
        permission=permission,
        risk=risk,
        approval_required=risk in {RiskLevel.EXTERNAL_WRITE, RiskLevel.DELETE},
        input_schema=model.model_json_schema(),
    )


class GetMessageInput(BaseModel):
    message_id: str


class GetMessageTool(ToolHandler[GetMessageInput]):
    request_model = GetMessageInput
    capability = _capability(
        "message_get", "Get a message by ID", "gmail:read", RiskLevel.READ,
        GetMessageInput,
    )

    def __init__(self, gateway: GmailGateway) -> None:
        self._gateway = gateway

    async def execute(self, request: GetMessageInput, principal: Principal) -> dict:
        del principal
        message = await self._gateway.get(request.message_id)
        if message is None:
            raise LookupError(f"Message not found: {request.message_id}")
        return {"message": message.model_dump(mode="json")}


class ListMessagesInput(BaseModel):
    mailbox: Literal["inbox", "sent"]
    limit: int = Field(default=20, ge=1, le=100)


class ListMessagesTool(ToolHandler[ListMessagesInput]):
    request_model = ListMessagesInput
    capability = _capability(
        "message_list", "List inbox or sent messages", "gmail:read", RiskLevel.READ,
        ListMessagesInput,
    )

    def __init__(self, gateway: GmailGateway) -> None:
        self._gateway = gateway

    async def execute(self, request: ListMessagesInput, principal: Principal) -> dict:
        del principal
        messages = await self._gateway.list_messages(request.mailbox, request.limit)
        return {"messages": [message.model_dump(mode="json") for message in messages]}


class SearchMessagesInput(BaseModel):
    mailbox: Literal["inbox", "sent"]
    query: str
    limit: int = Field(default=3, ge=1, le=100)


class SearchMessagesTool(ToolHandler[SearchMessagesInput]):
    request_model = SearchMessagesInput
    capability = _capability(
        "message_search", "Search inbox or sent messages", "gmail:read",
        RiskLevel.READ, SearchMessagesInput,
    )

    def __init__(self, gateway: GmailGateway) -> None:
        self._gateway = gateway

    async def execute(self, request: SearchMessagesInput, principal: Principal) -> dict:
        del principal
        messages = await self._gateway.search(
            request.mailbox, request.query, request.limit
        )
        return {"messages": [message.model_dump(mode="json") for message in messages]}


class SendMessageInput(BaseModel):
    sender: str
    recipients: list[str] = Field(min_length=1)
    subject: str
    body: str


class SendMessageTool(ToolHandler[SendMessageInput]):
    request_model = SendMessageInput
    capability = _capability(
        "message_send", "Send an email", "gmail:send", RiskLevel.EXTERNAL_WRITE,
        SendMessageInput,
    )

    def __init__(self, gateway: GmailGateway) -> None:
        self._gateway = gateway

    async def execute(self, request: SendMessageInput, principal: Principal) -> dict:
        del principal
        message = await self._gateway.send(
            request.sender, request.recipients, request.subject, request.body
        )
        return {"message": message.model_dump(mode="json")}


class DeleteMessageInput(BaseModel):
    message_id: str


class DeleteMessageTool(ToolHandler[DeleteMessageInput]):
    request_model = DeleteMessageInput
    capability = _capability(
        "message_delete", "Delete a received message", "gmail:delete",
        RiskLevel.DELETE, DeleteMessageInput,
    )

    def __init__(self, gateway: GmailGateway) -> None:
        self._gateway = gateway

    async def execute(self, request: DeleteMessageInput, principal: Principal) -> dict:
        del principal
        message = await self._gateway.get(request.message_id)
        if message is not None and message.mailbox != "inbox":
            raise ValueError("Only received messages may be deleted by this action")
        return {"deleted": await self._gateway.delete(request.message_id)}


def create_handlers(gateway: GmailGateway) -> list[ToolHandler]:
    return [
        GetMessageTool(gateway),
        ListMessagesTool(gateway),
        SearchMessagesTool(gateway),
        SendMessageTool(gateway),
        DeleteMessageTool(gateway),
    ]

