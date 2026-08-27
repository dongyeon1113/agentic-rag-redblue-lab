from agent_system.application.tool_calling_agent import (
    AgentQueryRequest,
    AgentRunState,
    ToolCallingAgent,
)
from agent_system.contracts import Capability, Principal, RiskLevel
from agent_system.infrastructure.ollama import (
    OllamaFunctionCall,
    OllamaMessage,
    OllamaToolCall,
)
from agent_system.services.drive.tools import SearchItemsInput
from agent_system.services.gmail.tools import SearchMessagesInput
from agent_system.services.local_db.domain import Document
from agent_system.services.local_db.tools import SearchDocumentsInput
from agent_system.services.local_db.vector_repository import VectorDocumentRepository


def test_search_tools_use_the_legacy_top_k_default() -> None:
    assert SearchDocumentsInput(query="query").limit == 3
    assert SearchMessagesInput(mailbox="inbox", query="query").limit == 3
    assert SearchItemsInput(query="query").limit == 3


def test_tool_agent_enforces_three_retrieval_candidates() -> None:
    capability = Capability(
        executor="local_db",
        action="document_search",
        description="Search documents",
        permission="document:read",
        risk=RiskLevel.READ,
        approval_required=False,
        input_schema=SearchDocumentsInput.model_json_schema(),
    )
    state = AgentRunState(
        workflow_id="workflow-1",
        request=AgentQueryRequest(
            user_id="user-1",
            session_id="session-1",
            query="Chicago Fire season four premiere date",
        ),
        principal=Principal(
            user_id="user-1",
            session_id="session-1",
            permissions={"document:read"},
        ),
        messages=[],
        tools=[],
        capabilities={"local_db__document_search": capability},
        iterations=1,
    )
    message = OllamaMessage(tool_calls=[
        OllamaToolCall(function=OllamaFunctionCall(
            name="local_db__document_search",
            arguments={"query": "Chicago Fire", "limit": 1},
        ))
    ])

    agent = object.__new__(ToolCallingAgent)
    tasks = agent._tasks_from_calls(state, message)

    assert tasks[0][1].parameters["limit"] == 3


def test_nq_title_is_not_duplicated_in_embedding_text() -> None:
    document = Document(
        document_id="doc-1",
        title="Chicago Fire (season 4)",
        content=(
            "Chicago Fire (season 4)\n\n"
            "The fourth season premiered on October 13, 2015."
        ),
    )

    record = VectorDocumentRepository._record(document)

    assert record.text == document.content

