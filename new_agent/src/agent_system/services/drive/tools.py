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
    item_id: str


class GetItemTool(ToolHandler[GetItemInput]):
    request_model = GetItemInput
    capability = _capability(
        "item_get", "Get a Drive item by ID", "drive:read", RiskLevel.READ,
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
    query: str
    parent_id: str | None = None
    limit: int = Field(default=3, ge=1, le=100)


class SearchItemsTool(ToolHandler[SearchItemsInput]):
    request_model = SearchItemsInput
    capability = _capability(
        "item_search", "Search files and folders", "drive:read", RiskLevel.READ,
        SearchItemsInput,
    )

    def __init__(self, gateway: DriveGateway) -> None:
        self._gateway = gateway

    async def execute(self, request: SearchItemsInput, principal: Principal) -> dict:
        del principal
        items = await self._gateway.search(request.query, request.parent_id, request.limit)
        return {"items": [item.model_dump() for item in items]}


class CreateItemInput(BaseModel):
    item_type: Literal["file", "folder"]
    name: str
    parent_id: str | None = None
    content: str | None = None


class CreateItemTool(ToolHandler[CreateItemInput]):
    request_model = CreateItemInput
    capability = _capability(
        "item_create", "Create a file or folder", "drive:write",
        RiskLevel.INTERNAL_WRITE, CreateItemInput,
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
    item_id: str
    expected_parent_id: str | None = None
    destination_parent_id: str


class MoveItemTool(ToolHandler[MoveItemInput]):
    request_model = MoveItemInput
    capability = _capability(
        "item_move", "Move a file or folder", "drive:write",
        RiskLevel.INTERNAL_WRITE, MoveItemInput,
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
    item_id: str


class DeleteItemTool(ToolHandler[DeleteItemInput]):
    request_model = DeleteItemInput
    capability = _capability(
        "item_delete", "Delete a file or folder", "drive:delete",
        RiskLevel.DELETE, DeleteItemInput,
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

