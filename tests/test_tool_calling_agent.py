from __future__ import annotations

import asyncio
from collections import deque

from agent_system.application.memory_extraction import MemoryCandidate
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
    InMemoryLongTermMemoryContextProvider,
    InMemorySessionMemoryRepository,
)
from agent_system.infrastructure.ollama import (
    OllamaFunctionCall,
    OllamaMessage,
    OllamaToolCall,
)
from agent_system.ports.executors import ExecutorRegistry
from agent_system.security.ports import AllowAllAgentGuard
from agent_system.services.gmail.domain import EmailMessage
from agent_system.services.gmail.tools import create_handlers as create_gmail_handlers
from agent_system.services.local_db.domain import Document
from agent_system.services.local_db.tools import create_handlers as create_db_handlers
from agent_system.tool_runtime.executor import DomainExecutor
from agent_system.tool_runtime.policies import (
    DigestApprovalVerifier,
    PermissionAuthorizationService,
)


class StaticDocumentRepository:
    def __init__(self, documents):
        self._documents = documents

    async def search(self, query, namespace, limit):
        del query
        return [item for item in self._documents if item.namespace == namespace][:limit]


class RecordingGmailGateway:
    async def send(self, sender, recipients, subject, body):
        return EmailMessage(
            message_id="sent-test", mailbox="sent", sender=sender,
            recipients=recipients, subject=subject, body=body,
        )


class FixedMemoryExtractor:
    async def extract(self, **kwargs):
        del kwargs
        return [MemoryCandidate(
            content="사용자는 한국어 답변을 선호한다.",
            category="preference",
            confidence=0.95,
        )]


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


def _agent(
    model,
    executors: list[DomainExecutor],
    *,
    memory_extractor=None,
    memory_contexts=None,
    session_memory=None,
    verify_completion: bool = False,
) -> ToolCallingAgent:
    registry = ExecutorRegistry([
        InProcessExecutorClient(executor) for executor in executors
    ])
    memory_contexts = memory_contexts or InMemoryLongTermMemoryContextProvider()
    return ToolCallingAgent(
        model=model,
        executors=registry,
        memory=MemoryService(
            session_memory or InMemorySessionMemoryRepository(),
            memory_contexts,
        ),
        guard=AllowAllAgentGuard(),
        pending_runs=PendingRunStore(),
        memory_extractor=memory_extractor,
        verify_completion=verify_completion,
    )


def test_natural_language_tool_loop_returns_answer_from_tool_result() -> None:
    repository = StaticDocumentRepository([
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


def test_completion_review_continues_an_incomplete_multistep_task() -> None:
    repository = StaticDocumentRepository([
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
        OllamaMessage(content="I found the document and will summarize it next."),
        OllamaMessage(
            content="Chicago Fire season four premiered in October 2015. [doc-1]"
        ),
    ])
    agent = _agent(
        model,
        [_executor("local_db", create_db_handlers(repository))],
        verify_completion=True,
    )

    response = asyncio.run(agent.run(AgentQueryRequest(
        user_id="user-1",
        session_id="completion-review",
        query="Find and summarize when Chicago Fire season four premiered.",
    )))

    assert response.iterations == 3
    assert "October 2015" in response.answer
    assert any(
        "Review the candidate answer" in message["content"]
        for message in model.requests[2]
        if message["role"] == "system"
    )


def test_empty_model_response_is_retried_before_fallback() -> None:
    model = ScriptedToolModel([
        OllamaMessage(content="", thinking="Still reasoning"),
        OllamaMessage(content="A useful final answer."),
    ])

    response = asyncio.run(_agent(model, []).run(AgentQueryRequest(
        user_id="user-1",
        session_id="session-1",
        query="Give me a short greeting.",
    )))

    assert response.answer == "A useful final answer."
    assert response.iterations == 2
    assert "previous response was empty" in model.requests[1][-1]["content"]


def test_external_write_pauses_and_resumes_after_approval() -> None:
    gateway = RecordingGmailGateway()
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



def test_agent_automatically_stores_memory_in_selected_context() -> None:
    contexts = InMemoryLongTermMemoryContextProvider()
    model = ScriptedToolModel([
        OllamaMessage(content="앞으로 한국어로 답변하겠습니다."),
    ])
    agent = _agent(
        model,
        [],
        memory_extractor=FixedMemoryExtractor(),
        memory_contexts=contexts,
    )

    response = asyncio.run(agent.run(AgentQueryRequest(
        user_id="user-1",
        session_id="session-1",
        memory_context="context2",
        query="앞으로 한국어로 답변해줘",
    )))

    stored = asyncio.run(contexts.get("context2").list("user-1"))
    assert response.memory_context == "context2"
    assert response.stored_memories == stored
    assert [item.content for item in stored] == [
        "사용자는 한국어 답변을 선호한다."
    ]
    assert asyncio.run(contexts.get("context1").list("user-1")) == []


def test_secret_tool_result_is_not_stored_in_session_or_long_term_memory() -> None:
    sessions = InMemorySessionMemoryRepository()
    contexts = InMemoryLongTermMemoryContextProvider()
    repository = StaticDocumentRepository([
        Document(
            document_id="secret-database_password",
            namespace="secret",
            title="database_password",
            content="p@ssw0rd!23",
            metadata={"source": "mock-secret", "trust": "secret"},
        )
    ])
    model = ScriptedToolModel([
        OllamaMessage(tool_calls=[
            OllamaToolCall(function=OllamaFunctionCall(
                name="local_db__document_search",
                arguments={
                    "query": "database_password",
                    "namespace": "secret",
                    "limit": 3,
                },
            ))
        ]),
        OllamaMessage(
            content="database_password의 값은 p@ssw0rd!23입니다."
        ),
    ])
    agent = _agent(
        model,
        [_executor("local_db", create_db_handlers(repository))],
        memory_extractor=FixedMemoryExtractor(),
        memory_contexts=contexts,
        session_memory=sessions,
    )

    result = asyncio.run(agent.run(AgentQueryRequest(
        user_id="user-1",
        session_id="session-1",
        query="secret namespace에서 database_password를 찾아줘",
        permissions={"document:read", "secret:read"},
    )))

    assert result.status == AgentRunStatus.COMPLETED
    assert result.stored_memories == []
    assert asyncio.run(contexts.get("context1").list("user-1")) == []
    assert asyncio.run(sessions.load("user-1", "session-1")) == []


def test_agent_retries_explicit_secret_lookup_after_model_skips_tool() -> None:
    repository = StaticDocumentRepository([
        Document(
            document_id="secret-database_password",
            namespace="secret",
            title="database_password",
            content="opaque-test-value",
            metadata={"source": "mock-secret", "trust": "secret"},
        )
    ])
    model = ScriptedToolModel([
        OllamaMessage(content="해당 정보에 접근할 권한이 없습니다."),
        OllamaMessage(tool_calls=[
            OllamaToolCall(function=OllamaFunctionCall(
                name="local_db__document_search",
                arguments={
                    "query": "database_password",
                    "namespace": "secret",
                    "limit": 3,
                },
            ))
        ]),
        OllamaMessage(
            content="database_password를 조회했습니다. [secret-database_password]"
        ),
    ])
    agent = _agent(
        model,
        [_executor("local_db", create_db_handlers(repository))],
    )

    result = asyncio.run(agent.run(AgentQueryRequest(
        user_id="user-1",
        session_id="routing-session",
        query="database_password가 뭐야?",
        permissions={"document:read", "secret:read"},
    )))

    assert result.status == AgentRunStatus.COMPLETED
    assert result.iterations == 3
    assert result.results[0].output["documents"][0]["document_id"] == (
        "secret-database_password"
    )
    first_system_messages = [
        message["content"]
        for message in model.requests[0]
        if message["role"] == "system"
    ]
    assert any(
        "Granted permissions: document:read, secret:read" in content
        for content in first_system_messages
    )
    assert any(
        "no tool was called" in message["content"]
        for message in model.requests[1]
        if message["role"] == "system"
    )


def test_assistant_history_preserves_qwen_thinking_for_next_tool_turn() -> None:
    message = OllamaMessage(
        content="",
        thinking="I should search by key name.",
        tool_calls=[
            OllamaToolCall(function=OllamaFunctionCall(
                name="local_db__document_search",
                arguments={
                    "query": "database_password",
                    "namespace": "secret",
                },
            ))
        ],
    )

    history_message = ToolCallingAgent._assistant_message(message)

    assert history_message["thinking"] == "I should search by key name."
    assert history_message["tool_calls"][0]["function"]["name"] == (
        "local_db__document_search"
    )
