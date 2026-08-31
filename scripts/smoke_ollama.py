from __future__ import annotations

import asyncio
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from agent_system.application.memory_service import MemoryService
from agent_system.application.tool_calling_agent import (
    AgentQueryRequest,
    PendingRunStore,
    ToolCallingAgent,
)
from agent_system.infrastructure.in_process_executor import InProcessExecutorClient
from agent_system.infrastructure.memory import (
    InMemoryLongTermMemoryContextProvider,
    InMemorySessionMemoryRepository,
)
from agent_system.infrastructure.ollama import OllamaChatModel
from agent_system.ports.executors import ExecutorRegistry
from agent_system.security.ports import AllowAllAgentGuard
from agent_system.services.local_db.json_repository import JsonDocumentRepository
from agent_system.services.local_db.tools import create_handlers
from agent_system.tool_runtime.executor import DomainExecutor
from agent_system.tool_runtime.policies import (
    DigestApprovalVerifier,
    PermissionAuthorizationService,
)


async def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    model = OllamaChatModel(
        model=os.getenv("OLLAMA_MODEL", "qwen3:8b"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        temperature=0,
        num_predict=512,
        timeout_seconds=180,
    )
    try:
        with TemporaryDirectory(prefix="agent-system-smoke-") as temporary:
            repository = JsonDocumentRepository(
                data_file=Path(temporary) / "documents.json",
                knowledge_seed_file=project_root / "mock_data/local_db/nq_10000.json",
                secret_seed_file=project_root / "mock_data/local_db/secrets.json",
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
                    InMemoryLongTermMemoryContextProvider(),
                ),
                guard=AllowAllAgentGuard(),
                pending_runs=PendingRunStore(),
                max_iterations=4,
            )
            response = await agent.run(AgentQueryRequest(
                user_id="smoke-user",
                session_id="smoke-session",
                query=(
                    "로컬 문서 데이터베이스에서 Chicago Fire 시즌 4가 "
                    "언제 처음 방영되었는지 찾아서 답해줘."
                ),
            ))
            print(response.model_dump_json(indent=2))
            if not response.results:
                raise SystemExit("Smoke test failed: the model did not call a tool")
            if response.results[0].status != "succeeded":
                raise SystemExit("Smoke test failed: tool execution failed")
    finally:
        await model.close()


if __name__ == "__main__":
    asyncio.run(main())

