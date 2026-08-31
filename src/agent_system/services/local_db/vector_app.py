from __future__ import annotations

import hmac
import os
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from agent_system.infrastructure.embeddings import OllamaEmbeddingClient
from agent_system.services.local_db.domain import Document
from agent_system.services.local_db.tools import create_handlers
from agent_system.services.local_db.vector_repository import VectorDocumentRepository
from agent_system.tool_runtime.api import create_tool_service_app
from agent_system.tool_runtime.executor import DomainExecutor
from agent_system.tool_runtime.policies import (
    DigestApprovalVerifier,
    PermissionAuthorizationService,
)
from attack.models import AttackDocument


class DeleteExperimentDocumentsRequest(BaseModel):
    document_ids: list[str]


def create_app() -> FastAPI:
    data_directory = Path(os.getenv("LOCAL_DB_DATA_DIR", "data/local_db"))
    repository = VectorDocumentRepository(
        data_file=data_directory / "documents.json",
        knowledge_seed_file=Path(
            os.getenv("KNOWLEDGE_SEED_FILE", "mock_data/local_db/nq_10000.json")
        ),
        secret_seed_file=Path(
            os.getenv("SECRET_SEED_FILE", "mock_data/local_db/secrets.json")
        ),
        persist_directory=Path(
            os.getenv("CHROMA_PERSIST_DIRECTORY", str(data_directory / "chroma"))
        ),
        collection_name=os.getenv("CHROMA_COLLECTION", "local-db-nomic-v2"),
        embedding=OllamaEmbeddingClient(
            model=os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
            base_url=os.getenv(
                "OLLAMA_EMBEDDING_BASE_URL", "http://localhost:11434"
            ),
            timeout_seconds=float(os.getenv("EMBEDDING_TIMEOUT_SECONDS", "180")),
        ),
        batch_size=int(os.getenv("CHROMA_INDEX_BATCH_SIZE", "500")),
    )
    executor = DomainExecutor(
        name="local_db",
        handlers=create_handlers(repository),
        authorization=PermissionAuthorizationService(),
        approval_verifier=DigestApprovalVerifier(),
    )
    app = create_tool_service_app(executor)
    expected_token = os.getenv("EXPERIMENT_API_TOKEN", "local-experiment-token")

    def authorize(token: str | None) -> None:
        if token is None or not hmac.compare_digest(token, expected_token):
            raise HTTPException(status_code=403, detail="Invalid experiment token")

    @app.post("/v1/experiments/documents", response_model=list[AttackDocument])
    async def inject_documents(
        documents: list[AttackDocument],
        x_experiment_token: str | None = Header(default=None),
    ) -> list[AttackDocument]:
        authorize(x_experiment_token)
        created: list[AttackDocument] = []
        try:
            for item in documents:
                if not item.document_id.startswith("experiment-"):
                    raise HTTPException(
                        status_code=422,
                        detail="Experiment document IDs must start with experiment-",
                    )
                metadata = {
                    **item.metadata,
                    "trust": "untrusted",
                    "source": "experiment",
                }
                document = Document(
                    document_id=item.document_id,
                    namespace="knowledge",
                    title=item.title,
                    content=item.content,
                    metadata=metadata,
                )
                await repository.create(document)
                created.append(item.model_copy(update={"metadata": metadata}))
        except Exception:
            for item in created:
                await repository.delete(item.document_id)
            raise
        return created

    @app.delete("/v1/experiments/documents")
    async def delete_documents(
        request: DeleteExperimentDocumentsRequest,
        x_experiment_token: str | None = Header(default=None),
    ) -> dict[str, list[str]]:
        authorize(x_experiment_token)
        deleted: list[str] = []
        for document_id in request.document_ids:
            if not document_id.startswith("experiment-"):
                continue
            if await repository.delete(document_id):
                deleted.append(document_id)
        return {"deleted": deleted}

    return app


app = create_app()
