from services.orchestrator.router import route_request


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
