from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from agent_system.contracts import Capability, Principal, RiskLevel
from agent_system.services.drive.domain import DriveGateway
from agent_system.tool_runtime.handler import ToolHandler


def _capability(
    action: str,
    description: str,
    permission: str,
    risk: RiskLevel,
    model: type[BaseModel],
    *,
    approval_required: bool = False,
) -> Capability:
    return Capability(
        executor="drive",
        action=action,
        description=description,
        permission=permission,
        risk=risk,
        approval_required=approval_required,
        input_schema=model.model_json_schema(),
    )


class GetItemInput(BaseModel):
    item_id: str = Field(
        description=(
            "Exact item_id returned by a previous Drive search. Do not use a file or "
            "folder name here."
        )
    )


class GetItemTool(ToolHandler[GetItemInput]):
    request_model = GetItemInput
    capability = _capability(
        "item_get",
        (
            "Fetch one Drive file or folder only when its exact item_id is already "
            "known. Use item_search when the user provides a name, topic, or description."
        ),
        "drive:read",
        RiskLevel.READ,
        GetItemInput,
    )

    def __init__(self, gateway: DriveGateway) -> None:
        self._gateway = gateway

    async def execute(self, request: GetItemInput, principal: Principal) -> dict:
        del principal
        item = await self._gateway.get(request.item_id)
        if item is None:
            raise LookupError(f"Drive item not found: {request.item_id}")
        return {"item": item.model_dump()}


class SearchItemsInput(BaseModel):
    query: str = Field(
        description=(
            "Semantic search text preserving the requested file name, folder name, "
            "topic, or identifying terms."
        )
    )
    parent_id: str | None = Field(
        default=None,
        description=(
            "Optional exact parent folder ID. Leave null unless the user specifies a "
            "known folder ID or a previous result established it."
        ),
    )
    limit: int = Field(
        default=3,
        ge=1,
        le=100,
        description="Number of candidates to inspect; request at least 3.",
    )


class SearchItemsTool(ToolHandler[SearchItemsInput]):
    request_model = SearchItemsInput
    capability = _capability(
        "item_search",
        (
            "Search Drive files and folders by name, topic, or description when an "
            "exact item_id is not known. Use item_get only after a search returned an ID."
        ),
        "drive:read",
        RiskLevel.READ,
        SearchItemsInput,
    )

    def __init__(self, gateway: DriveGateway) -> None:
        self._gateway = gateway

    async def execute(self, request: SearchItemsInput, principal: Principal) -> dict:
        del principal
        items = await self._gateway.search(request.query, request.parent_id, request.limit)
        return {"items": [item.model_dump() for item in items]}


class CreateItemInput(BaseModel):
    item_type: Literal["file", "folder"] = Field(
        description="Create a file for content or a folder as a container."
    )
    name: str = Field(description="Name of the new file or folder.")
    parent_id: str | None = Field(
        default=None, description="Exact destination parent folder ID, or null for root."
    )
    content: str | None = Field(
        default=None, description="File content; normally null when creating a folder."
    )


class CreateItemTool(ToolHandler[CreateItemInput]):
    request_model = CreateItemInput
    capability = _capability(
        "item_create",
        (
            "Create a new Drive file or folder only when the user explicitly asks to "
            "create one. Requires drive:write."
        ),
        "drive:write",
        RiskLevel.INTERNAL_WRITE,
        CreateItemInput,
    )

    def __init__(self, gateway: DriveGateway) -> None:
        self._gateway = gateway

    async def execute(self, request: CreateItemInput, principal: Principal) -> dict:
        del principal
        item = await self._gateway.create(
            request.item_type, request.name, request.parent_id, request.content
        )
        return {"item": item.model_dump()}


class MoveItemInput(BaseModel):
    item_id: str = Field(description="Exact ID of the file or folder to move.")
    expected_parent_id: str | None = Field(
        default=None,
        description="Current parent ID when known, used as a concurrency precondition.",
    )
    destination_parent_id: str = Field(
        description="Exact destination folder ID returned by a previous search."
    )


class MoveItemTool(ToolHandler[MoveItemInput]):
    request_model = MoveItemInput
    capability = _capability(
        "item_move",
        (
            "Move a Drive file or folder using exact source and destination IDs. Search "
            "for unknown IDs first. Requires drive:write and user approval."
        ),
        "drive:write",
        RiskLevel.INTERNAL_WRITE,
        MoveItemInput,
        approval_required=True,
    )

    def __init__(self, gateway: DriveGateway) -> None:
        self._gateway = gateway

    async def execute(self, request: MoveItemInput, principal: Principal) -> dict:
        del principal
        item = await self._gateway.move(
            request.item_id,
            request.expected_parent_id,
            request.destination_parent_id,
        )
        return {"item": item.model_dump()}


class DeleteItemInput(BaseModel):
    item_id: str = Field(description="Exact ID of the Drive item to delete.")


class DeleteItemTool(ToolHandler[DeleteItemInput]):
    request_model = DeleteItemInput
    capability = _capability(
        "item_delete",
        (
            "Delete a Drive file or folder by exact item_id. Search first when the ID "
            "is unknown. Requires drive:delete and user approval."
        ),
        "drive:delete",
        RiskLevel.DELETE,
        DeleteItemInput,
        approval_required=True,
    )

    def __init__(self, gateway: DriveGateway) -> None:
        self._gateway = gateway

    async def execute(self, request: DeleteItemInput, principal: Principal) -> dict:
        del principal
        return {"deleted": await self._gateway.delete(request.item_id)}


def create_handlers(gateway: DriveGateway) -> list[ToolHandler]:
    return [
        GetItemTool(gateway),
        SearchItemsTool(gateway),
        CreateItemTool(gateway),
        MoveItemTool(gateway),
        DeleteItemTool(gateway),
    ]
