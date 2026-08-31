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
    document_id: str = Field(
        description=(
            "Exact document_id returned by a previous search, such as doc-123 or "
            "secret-test_api_key. Do not use a title or keyword here."
        )
    )


class GetDocumentTool(ToolHandler[GetDocumentInput]):
    request_model = GetDocumentInput
    capability = _capability(
        "document_get",
        (
            "Fetch one Local DB document only when its exact document_id is already "
            "known. For an unknown ID, title, key name, or topic use document_search. "
            "A secret-* document additionally requires secret:read."
        ),
        "document:read",
        RiskLevel.READ,
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
    query: str = Field(
        description=(
            "Semantic search text preserving the user's identifying terms, such as "
            "Chicago Fire season 4 or database_password."
        )
    )
    namespace: Literal["knowledge", "secret"] = Field(
        default="knowledge",
        description=(
            "Use knowledge for ordinary documents. Use secret when the user explicitly "
            "requests stored credentials, passwords, tokens, API keys, or the secret "
            "namespace. secret additionally requires secret:read."
        ),
    )
    limit: int = Field(
        default=3,
        ge=1,
        le=100,
        description="Number of candidates to inspect; request at least 3.",
    )


class SearchDocumentsTool(ToolHandler[SearchDocumentsInput]):
    request_model = SearchDocumentsInput
    capability = _capability(
        "document_search",
        (
            "Search Local DB by title, key name, keyword, or topic when an exact "
            "document_id is not known. Use namespace=secret for credential names such "
            "as database_password or test_api_key; this requires both document:read "
            "and secret:read. Never claim a stored value without calling this tool."
        ),
        "document:read",
        RiskLevel.READ,
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
        "document_create",
        (
            "Create a new Local DB document. Use only when the user asks to store new "
            "content, not for lookup. namespace=secret additionally requires secret:write."
        ),
        "document:write",
        RiskLevel.INTERNAL_WRITE,
        CreateDocumentInput,
    )

    def __init__(self, repository: DocumentRepository) -> None:
        self._repository = repository

    async def execute(self, request: CreateDocumentInput, principal: Principal) -> dict:
        _require_secret_permission(principal, request.namespace, "write")
        document = await self._repository.create(Document.model_validate(request))
        return {"document": document.model_dump()}


class UpdateDocumentInput(BaseModel):
    document_id: str = Field(description="Exact ID of the document to update.")
    title: str | None = Field(default=None, description="Replacement title, if requested.")
    content: str | None = Field(
        default=None, description="Replacement document content, if requested."
    )
    metadata: dict[str, str] | None = Field(
        default=None, description="Replacement metadata, if requested."
    )


class UpdateDocumentTool(ToolHandler[UpdateDocumentInput]):
    request_model = UpdateDocumentInput
    capability = _capability(
        "document_update",
        (
            "Update an existing Local DB document by exact document_id. Search first "
            "when only a title or key name is known. Secret documents additionally "
            "require secret:write."
        ),
        "document:write",
        RiskLevel.INTERNAL_WRITE,
        UpdateDocumentInput,
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
    document_id: str = Field(description="Exact ID of the document to delete.")


class DeleteDocumentTool(ToolHandler[DeleteDocumentInput]):
    request_model = DeleteDocumentInput
    capability = _capability(
        "document_delete",
        (
            "Delete a Local DB document by exact document_id after user approval. "
            "Search first when the ID is unknown. Secret documents additionally require "
            "secret:delete."
        ),
        "document:delete",
        RiskLevel.DELETE,
        DeleteDocumentInput,
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
