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
    message_id: str = Field(
        description=(
            "Exact message_id returned by a previous Gmail search or list operation. "
            "Do not use a subject or sender here."
        )
    )


class GetMessageTool(ToolHandler[GetMessageInput]):
    request_model = GetMessageInput
    capability = _capability(
        "message_get",
        (
            "Fetch one Gmail message only when its exact message_id is already known. "
            "Use message_search when the user provides a subject, sender, recipient, "
            "keyword, or description instead of an ID."
        ),
        "gmail:read",
        RiskLevel.READ,
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
    mailbox: Literal["inbox", "sent"] = Field(
        description="Mailbox to list: inbox for received mail or sent for sent mail."
    )
    limit: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum number of newest messages to return.",
    )


class ListMessagesTool(ToolHandler[ListMessagesInput]):
    request_model = ListMessagesInput
    capability = _capability(
        "message_list",
        (
            "List recent Gmail messages when the user asks to show the inbox or sent "
            "mail without a search topic. Use message_search for any sender, subject, "
            "recipient, keyword, or event criteria."
        ),
        "gmail:read",
        RiskLevel.READ,
        ListMessagesInput,
    )

    def __init__(self, gateway: GmailGateway) -> None:
        self._gateway = gateway

    async def execute(self, request: ListMessagesInput, principal: Principal) -> dict:
        del principal
        messages = await self._gateway.list_messages(request.mailbox, request.limit)
        return {"messages": [message.model_dump(mode="json") for message in messages]}


class SearchMessagesInput(BaseModel):
    mailbox: Literal["inbox", "sent"] = Field(
        description="Search inbox for received messages or sent for sent messages."
    )
    query: str = Field(
        description=(
            "Semantic search text preserving identifying terms such as sender, subject, "
            "recipient, project name, or event."
        )
    )
    limit: int = Field(
        default=3,
        ge=1,
        le=100,
        description="Number of candidates to inspect; request at least 3.",
    )


class SearchMessagesTool(ToolHandler[SearchMessagesInput]):
    request_model = SearchMessagesInput
    capability = _capability(
        "message_search",
        (
            "Search Gmail when the user asks to find, check, summarize, or identify "
            "messages using a sender, subject, recipient, keyword, project, or event. "
            "Use message_get only when an exact message_id is already known."
        ),
        "gmail:read",
        RiskLevel.READ,
        SearchMessagesInput,
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
    sender: str = Field(description="Sender email address, normally the current user.")
    recipients: list[str] = Field(
        min_length=1, description="One or more destination email addresses."
    )
    subject: str = Field(description="Email subject requested by the user.")
    body: str = Field(description="Complete email body requested by the user.")


class SendMessageTool(ToolHandler[SendMessageInput]):
    request_model = SendMessageInput
    capability = _capability(
        "message_send",
        (
            "Send a Gmail message only when the user explicitly asks to send email. "
            "This external write requires gmail:send and pauses for user approval."
        ),
        "gmail:send",
        RiskLevel.EXTERNAL_WRITE,
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
    message_id: str = Field(
        description="Exact ID of a received inbox message to delete."
    )


class DeleteMessageTool(ToolHandler[DeleteMessageInput]):
    request_model = DeleteMessageInput
    capability = _capability(
        "message_delete",
        (
            "Delete a received Gmail inbox message by exact message_id. Search first "
            "when the ID is unknown. Requires gmail:delete and user approval."
        ),
        "gmail:delete",
        RiskLevel.DELETE,
        DeleteMessageInput,
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
