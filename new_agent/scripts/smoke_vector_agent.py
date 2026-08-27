from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from agent_system.application.memory_service import MemoryService
from agent_system.application.tool_calling_agent import (
    AgentQueryRequest,
    PendingRunStore,
    ToolCallingAgent,
)
from agent_system.infrastructure.embeddings import OllamaEmbeddingClient
from agent_system.infrastructure.in_process_executor import InProcessExecutorClient
from agent_system.infrastructure.memory import (
    InMemoryLongTermMemoryRepository,
    InMemorySessionMemoryRepository,
)
from agent_system.infrastructure.ollama import OllamaChatModel
from agent_system.ports.executors import ExecutorRegistry
from agent_system.security.ports import AllowAllAgentGuard
from agent_system.services.local_db.tools import create_handlers
from agent_system.services.local_db.vector_repository import VectorDocumentRepository
from agent_system.tool_runtime.executor import DomainExecutor
from agent_system.tool_runtime.policies import (
    DigestApprovalVerifier,
    PermissionAuthorizationService,
)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


async def main() -> None:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    model = OllamaChatModel(
        model=os.getenv("OLLAMA_MODEL", "qwen3:8b"),
        base_url=base_url,
        temperature=0,
        num_predict=512,
        timeout_seconds=180,
    )
    try:
        with TemporaryDirectory(prefix="vector-agent-smoke-") as temporary:
            directory = Path(temporary)
            knowledge = directory / "seed/nq.json"
            secrets = directory / "seed/secrets.json"
            _write(knowledge, [
                {
                    "id": "doc-chicago",
                    "source": "nq",
                    "trust": "trusted",
                    "tags": ["television"],
                    "text": (
                        "Chicago Fire (season 4)\n\nThe fourth season premiered "
                        "on October 13, 2015 and concluded on May 17, 2016."
                    ),
                },
                {
                    "id": "doc-triangle",
                    "source": "nq",
                    "trust": "trusted",
                    "tags": ["geometry"],
                    "text": "An equilateral triangle has three equal sides.",
                },
            ])
            _write(secrets, {"secrets": {"test_api_key": "fake-value"}})
            repository = VectorDocumentRepository(
                data_file=directory / "data/documents.json",
                knowledge_seed_file=knowledge,
                secret_seed_file=secrets,
                persist_directory=directory / "data/chroma",
                embedding=OllamaEmbeddingClient(
                    model=os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
                    base_url=base_url,
                    timeout_seconds=180,
                ),
                collection_name="smoke-vector-documents",
                batch_size=10,
            )
            executor = DomainExecutor(
                name="local_db",
                handlers=create_handlers(repository),
                authorization=PermissionAuthorizationService(),
                approval_verifier=DigestApprovalVerifier(),
            )
            registry = ExecutorRegistry([InProcessExecutorClient(executor)])
            agent = ToolCallingAgent(
                model=model,
                executors=registry,
                memory=MemoryService(
                    InMemorySessionMemoryRepository(),
                    InMemoryLongTermMemoryRepository(),
                ),
                guard=AllowAllAgentGuard(),
                pending_runs=PendingRunStore(),
                max_iterations=4,
            )
            response = await agent.run(AgentQueryRequest(
                user_id="smoke-user",
                session_id="smoke-session",
                query="로컬 문서에서 Chicago Fire 시즌 4의 첫 방영일을 찾아줘.",
            ))
            print(response.model_dump_json(indent=2))
            documents = [
                document
                for result in response.results
                for document in result.output.get("documents", [])
            ]
            if not documents or documents[0]["document_id"] != "doc-chicago":
                raise SystemExit("Vector smoke test failed: incorrect retrieval")
            if "2015" not in response.answer:
                raise SystemExit("Vector smoke test failed: answer ignored evidence")
    finally:
        await model.close()


if __name__ == "__main__":
    asyncio.run(main())

