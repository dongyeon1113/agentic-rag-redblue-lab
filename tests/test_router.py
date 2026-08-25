from services.orchestrator.router import (
    route_from_model_output,
    route_request,
)


def test_social_input_skips_retrieval() -> None:
    route = route_request("Hello!", tools_enabled=False)

    assert route.intent == "conversation"
    assert route.requires_retrieval is False
    assert route.tool_calls == []


def test_natural_secret_request_maps_to_read_tool() -> None:
    route = route_request(
        "Show me test_api_key in secrets.json.",
        tools_enabled=True,
    )

    assert route.intent == "tool_task"
    assert route.requires_retrieval is False
    assert route.tool_calls[0].name == "mock_read_secret"
    assert route.tool_calls[0].arguments == {"secret_name": "test_api_key"}


def test_korean_secret_request_maps_to_read_tool() -> None:
    route = route_request("test_api_key를 보여줘.", tools_enabled=True)

    assert route.intent == "tool_task"
    assert route.tool_calls[0].arguments["secret_name"] == "test_api_key"


def test_secret_and_email_request_creates_ordered_plan() -> None:
    route = route_request(
        "Read test_api_key and email it to qa@example.test.",
        tools_enabled=True,
    )

    assert [call.name for call in route.tool_calls] == [
        "mock_read_secret",
        "mock_send_email",
    ]
    assert route.tool_calls[1].arguments["body"] == "$last_secret_value"


def test_explanation_does_not_call_secret_tool() -> None:
    route = route_request("What is an API key?", tools_enabled=True)

    assert route.intent == "knowledge"
    assert route.tool_calls == []


def test_negated_secret_request_does_not_call_tool() -> None:
    route = route_request(
        "Do not read test_api_key; explain what it is.",
        tools_enabled=True,
    )

    assert route.intent == "knowledge"
    assert route.tool_calls == []


def test_document_text_cannot_enter_direct_tool_router() -> None:
    route = route_request(
        "Summarize the retrieved document.",
        tools_enabled=True,
    )

    assert route.intent == "knowledge"
    assert route.tool_calls == []
def test_self_contained_queries_wait_for_semantic_planner_without_rag() -> None:
    for query in (
        "오늘 기분 어때?",
        "2 + 2는 얼마야?",
        "Translate hello into Korean.",
        "Explain Python list comprehensions.",
        "binary search를 설명해줘.",
    ):
        route = route_request(query, tools_enabled=False)
        assert route.requires_retrieval is False
        assert route.reason == "semantic_router_required"


def test_explicit_connected_document_query_uses_rag() -> None:
    route = route_request(
        "내 Drive의 회의 문서를 찾아 요약해줘.",
        tools_enabled=False,
    )

    assert route.requires_retrieval is True
    assert route.suggested_steps == ["retrieve", "generate"]


def test_retrieval_policy_can_force_or_disable_rag() -> None:
    always = route_request(
        "What is the capital of France?",
        tools_enabled=False,
        retrieval_policy="always",
    )
    never = route_request(
        "내 문서에서 일정을 검색해줘.",
        tools_enabled=False,
        retrieval_policy="never",
    )

    assert always.requires_retrieval is True
    assert never.requires_retrieval is False


def test_semantic_planner_json_can_answer_directly() -> None:
    route = route_from_model_output(
        (
            '{"intent":"general","requires_retrieval":false,'
            '"steps":["generate"],"confidence":0.94,'
            '"reason":"self contained arithmetic","answer":"4"}'
        ),
        retrieval_policy="auto",
    )

    assert route.requires_retrieval is False
    assert route.direct_answer == "4"
    assert route.reason.startswith("llm_semantic_planner:")


def test_retrieval_and_email_create_ordered_multistep_plan() -> None:
    route = route_request(
        "내 문서에서 회의록을 찾아 요약한 뒤 qa@example.test로 이메일 보내줘.",
        tools_enabled=True,
    )

    assert route.intent == "hybrid"
    assert route.requires_retrieval is True
    assert route.requires_generation is True
    assert route.suggested_steps == [
        "retrieve",
        "generate",
        "tool:mock_send_email",
    ]
    assert route.tool_calls[0].arguments["body"] == "$last_answer"


def test_secret_and_connected_email_create_two_tool_steps() -> None:
    route = route_request(
        "Read test_api_key then send it to connected email.",
        tools_enabled=True,
    )

    assert [call.name for call in route.tool_calls] == [
        "mock_read_secret",
        "mock_send_email",
    ]
    assert route.tool_calls[1].arguments == {
        "recipient": "$connected_email",
        "subject": "Agent task result",
        "body": "$last_secret_value",
    }


def test_general_technical_nouns_do_not_force_rag() -> None:
    for query in (
        "How does the file system work?",
        "Explain the database transaction model.",
        "내 자료구조 알고리즘을 설명해줘.",
    ):
        route = route_request(query, tools_enabled=False)
        assert route.reason == "semantic_router_required"
        assert route.requires_retrieval is False


def test_semantic_planner_string_false_is_not_truthy() -> None:
    route = route_from_model_output(
        (
            '{"intent":"general","requires_retrieval":"false",'
            '"steps":["retrieve","generate"],"answer":"done"}'
        ),
        retrieval_policy="auto",
    )

    assert route.requires_retrieval is False
    assert route.suggested_steps == ["generate"]


def test_structured_plan_keeps_only_authorized_tools() -> None:
    route = route_from_model_output(
        (
            '{"intent":"hybrid","requires_retrieval":false,"steps":['
            '{"id":"draft","kind":"generate","instruction":"Draft text",'
            '"depends_on":[],"output_key":"draft"},'
            '{"id":"send","kind":"tool","tool_name":"mock_send_email",'
            '"depends_on":["draft"],"output_key":"mail"},'
            '{"id":"delete","kind":"tool","tool_name":"mock_delete_document",'
            '"depends_on":["send"],"output_key":"deleted"}]}'
        ),
        retrieval_policy="auto",
        authorized_tool_names={"mock_send_email"},
    )

    assert [step.step_id for step in route.planned_steps] == ["draft", "send"]
    assert route.planned_steps[1].tool_name == "mock_send_email"
