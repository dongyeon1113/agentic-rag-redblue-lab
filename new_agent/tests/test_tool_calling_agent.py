from __future__ import annotations

import asyncio
from collections import deque

from agent_system.application.memory_service import MemoryService
from agent_system.application.tool_calling_agent import (
    AgentQueryRequest,
    AgentRunStatus,
    ApproveWorkflowRequest,
    PendingRunStore,
    ToolCallingAgent,
)
from agent_system.infrastructure.in_process_executor import InProcessExecutorClient
from agent_system.infrastructure.memory import (
    InMemoryLongTermMemoryRepository,
    InMemorySessionMemoryRepository,
)
from agent_system.infrastructure.ollama import (
    OllamaFunctionCall,
    OllamaMessage,
    OllamaToolCall,
)
from agent_system.ports.executors import ExecutorRegistry
from agent_system.security.ports import AllowAllAgentGuard
from agent_system.services.gmail.domain import InMemoryGmailGateway
from agent_system.services.gmail.tools import create_handlers as create_gmail_handlers
from agent_system.services.local_db.domain import Document, InMemoryDocumentRepository
from agent_system.services.local_db.tools import create_handlers as create_db_handlers
from agent_system.tool_runtime.executor import DomainExecutor
from agent_system.tool_runtime.policies import (
    DigestApprovalVerifier,
    PermissionAuthorizationService,
)


class ScriptedToolModel:
    model = "scripted-test-model"

    def __init__(self, messages: list[OllamaMessage]) -> None:
        self._messages = deque(messages)
        self.requests: list[list[dict]] = []

    async def chat(self, messages, *, tools=None, response_format=None):
        del tools, response_format
        self.requests.append(list(messages))
        return self._messages.popleft()


def _executor(name: str, handlers) -> DomainExecutor:
    return DomainExecutor(
        name=name,
        handlers=handlers,
        authorization=PermissionAuthorizationService(),
        approval_verifier=DigestApprovalVerifier(),
    )


def _agent(model, executors: list[DomainExecutor]) -> ToolCallingAgent:
    registry = ExecutorRegistry([
        InProcessExecutorClient(executor) for executor in executors
    ])
    return ToolCallingAgent(
        model=model,
        executors=registry,
        memory=MemoryService(
            InMemorySessionMemoryRepository(),
            InMemoryLongTermMemoryRepository(),
        ),
        guard=AllowAllAgentGuard(),
        pending_runs=PendingRunStore(),
    )


def test_natural_language_tool_loop_returns_answer_from_tool_result() -> None:
    repository = InMemoryDocumentRepository([
        Document(
            document_id="doc-1",
            title="Chicago Fire",
            content="Chicago Fire season four premiered in October 2015.",
        )
    ])
    model = ScriptedToolModel([
        OllamaMessage(tool_calls=[
            OllamaToolCall(function=OllamaFunctionCall(
                name="local_db__document_search",
                arguments={
                    "query": "Chicago Fire",
                    "namespace": "knowledge",
                    "limit": 3,
                },
            ))
        ]),
        OllamaMessage(content="Chicago Fire 시즌 4는 2015년 10월에 방영되었습니다. [doc-1]"),
    ])
    agent = _agent(model, [_executor("local_db", create_db_handlers(repository))])

    response = asyncio.run(agent.run(AgentQueryRequest(
        user_id="user-1",
        session_id="session-1",
        query="Chicago Fire 시즌 4의 방영 시점을 찾아줘",
    )))

    assert response.status == AgentRunStatus.COMPLETED
    assert response.results[0].output["documents"][0]["document_id"] == "doc-1"
    assert "2015년" in response.answer
    assert any(
        message["role"] == "tool"
        for message in model.requests[1]
    )


def test_external_write_pauses_and_resumes_after_approval() -> None:
    gateway = InMemoryGmailGateway()
    model = ScriptedToolModel([
        OllamaMessage(tool_calls=[
            OllamaToolCall(function=OllamaFunctionCall(
                name="gmail__message_send",
                arguments={
                    "sender": "me@example.com",
                    "recipients": ["recipient@example.com"],
                    "subject": "Test",
                    "body": "Hello",
                },
            ))
        ]),
        OllamaMessage(content="승인된 메일을 발송했습니다."),
    ])
    agent = _agent(model, [_executor("gmail", create_gmail_handlers(gateway))])
    request = AgentQueryRequest(
        user_id="user-1",
        session_id="session-1",
        query="recipient@example.com으로 테스트 메일을 보내줘",
        permissions={"gmail:send"},
    )

    pending = asyncio.run(agent.run(request))
    assert pending.status == AgentRunStatus.AWAITING_APPROVAL
    task_id = pending.approval_requests[0].task_id

    completed = asyncio.run(agent.approve(
        pending.workflow_id,
        ApproveWorkflowRequest(
            user_id="user-1",
            session_id="session-1",
            approved_task_ids={task_id},
        ),
    ))
    assert completed.status == AgentRunStatus.COMPLETED
    assert completed.results[0].status == "succeeded"
    assert "발송" in completed.answer

