from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from agent_system.contracts import Capability, Principal, RiskLevel
from agent_system.services.local_db.domain import Document, DocumentRepository
from agent_system.tool_runtime.handler import ToolHandler
from agent_system.tool_runtime.policies import AuthorizationError


def _capability(
    action: str,
    description: str,
    permission: str,
    risk: RiskLevel,
    model: type[BaseModel],
) -> Capability:
    return Capability(
        executor="local_db",
        action=action,
        description=description,
        permission=permission,
        risk=risk,
        approval_required=risk == RiskLevel.DELETE,
        input_schema=model.model_json_schema(),
    )


def _require_secret_permission(principal: Principal, namespace: str, verb: str) -> None:
    permission = f"secret:{verb}"
    if namespace == "secret" and permission not in principal.permissions:
        raise AuthorizationError(f"Missing permission: {permission}")


class GetDocumentInput(BaseModel):
    document_id: str


class GetDocumentTool(ToolHandler[GetDocumentInput]):
    request_model = GetDocumentInput
    capability = _capability(
        "document_get", "Get a document by ID", "document:read", RiskLevel.READ,
        GetDocumentInput,
    )

    def __init__(self, repository: DocumentRepository) -> None:
        self._repository = repository

    async def execute(self, request: GetDocumentInput, principal: Principal) -> dict:
        document = await self._repository.get(request.document_id)
        if document is None:
            raise LookupError(f"Document not found: {request.document_id}")
        _require_secret_permission(principal, document.namespace, "read")
        return {"document": document.model_dump()}


class SearchDocumentsInput(BaseModel):
    query: str
    namespace: Literal["knowledge", "secret"] = "knowledge"
    limit: int = Field(default=3, ge=1, le=100)


class SearchDocumentsTool(ToolHandler[SearchDocumentsInput]):
    request_model = SearchDocumentsInput
    capability = _capability(
        "document_search", "Search documents", "document:read", RiskLevel.READ,
        SearchDocumentsInput,
    )

    def __init__(self, repository: DocumentRepository) -> None:
        self._repository = repository

    async def execute(self, request: SearchDocumentsInput, principal: Principal) -> dict:
        _require_secret_permission(principal, request.namespace, "read")
        documents = await self._repository.search(
            request.query, request.namespace, request.limit
        )
        return {"documents": [document.model_dump() for document in documents]}


class CreateDocumentInput(Document):
    pass


class CreateDocumentTool(ToolHandler[CreateDocumentInput]):
    request_model = CreateDocumentInput
    capability = _capability(
        "document_create", "Create a document", "document:write",
        RiskLevel.INTERNAL_WRITE, CreateDocumentInput,
    )

    def __init__(self, repository: DocumentRepository) -> None:
        self._repository = repository

    async def execute(self, request: CreateDocumentInput, principal: Principal) -> dict:
        _require_secret_permission(principal, request.namespace, "write")
        document = await self._repository.create(Document.model_validate(request))
        return {"document": document.model_dump()}


class UpdateDocumentInput(BaseModel):
    document_id: str
    title: str | None = None
    content: str | None = None
    metadata: dict[str, str] | None = None


class UpdateDocumentTool(ToolHandler[UpdateDocumentInput]):
    request_model = UpdateDocumentInput
    capability = _capability(
        "document_update", "Update a document", "document:write",
        RiskLevel.INTERNAL_WRITE, UpdateDocumentInput,
    )

    def __init__(self, repository: DocumentRepository) -> None:
        self._repository = repository

    async def execute(self, request: UpdateDocumentInput, principal: Principal) -> dict:
        current = await self._repository.get(request.document_id)
        if current is None:
            raise LookupError(f"Document not found: {request.document_id}")
        _require_secret_permission(principal, current.namespace, "write")
        changes = request.model_dump(exclude={"document_id"}, exclude_none=True)
        updated = await self._repository.update(request.document_id, changes)
        return {"document": updated.model_dump()}


class DeleteDocumentInput(BaseModel):
    document_id: str


class DeleteDocumentTool(ToolHandler[DeleteDocumentInput]):
    request_model = DeleteDocumentInput
    capability = _capability(
        "document_delete", "Delete a document", "document:delete",
        RiskLevel.DELETE, DeleteDocumentInput,
    )

    def __init__(self, repository: DocumentRepository) -> None:
        self._repository = repository

    async def execute(self, request: DeleteDocumentInput, principal: Principal) -> dict:
        current = await self._repository.get(request.document_id)
        if current is not None:
            _require_secret_permission(principal, current.namespace, "delete")
        return {"deleted": await self._repository.delete(request.document_id)}


def create_handlers(repository: DocumentRepository) -> list[ToolHandler]:
    return [
        GetDocumentTool(repository),
        SearchDocumentsTool(repository),
        CreateDocumentTool(repository),
        UpdateDocumentTool(repository),
        DeleteDocumentTool(repository),
    ]

