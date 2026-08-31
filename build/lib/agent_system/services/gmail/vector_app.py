import os
from pathlib import Path

from fastapi import FastAPI

from agent_system.infrastructure.embeddings import OllamaEmbeddingClient
from agent_system.services.gmail.tools import create_handlers
from agent_system.services.gmail.vector_gateway import VectorGmailGateway
from agent_system.tool_runtime.api import create_tool_service_app
from agent_system.tool_runtime.executor import DomainExecutor
from agent_system.tool_runtime.policies import (
    DigestApprovalVerifier,
    PermissionAuthorizationService,
)


def create_app() -> FastAPI:
    data_directory = Path(os.getenv("GMAIL_DATA_DIR", "data/gmail"))
    gateway = VectorGmailGateway(
        inbox_file=data_directory / "inbox.json",
        sent_file=data_directory / "sent.json",
        inbox_seed_file=Path(
            os.getenv("GMAIL_INBOX_SEED_FILE", "mock_data/gmail/inbox.json")
        ),
        sent_seed_file=Path(
            os.getenv("GMAIL_SENT_SEED_FILE", "mock_data/gmail/sent.json")
        ),
        persist_directory=Path(
            os.getenv("CHROMA_PERSIST_DIRECTORY", str(data_directory / "chroma"))
        ),
        collection_name=os.getenv("CHROMA_COLLECTION", "gmail-nomic-v1"),
        embedding=OllamaEmbeddingClient(
            model=os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
            base_url=os.getenv(
                "OLLAMA_EMBEDDING_BASE_URL", "http://localhost:11434"
            ),
            timeout_seconds=float(os.getenv("EMBEDDING_TIMEOUT_SECONDS", "180")),
        ),
    )
    executor = DomainExecutor(
        name="gmail",
        handlers=create_handlers(gateway),
        authorization=PermissionAuthorizationService(),
        approval_verifier=DigestApprovalVerifier(),
    )
    return create_tool_service_app(executor)


app = create_app()

