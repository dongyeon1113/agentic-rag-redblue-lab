"""Deterministic fast-path routing for conversation, RAG, and mock tools."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

RouteIntent = Literal["conversation", "knowledge", "tool_task", "hybrid"]

_SOCIAL = {
    "hello", "hi", "hey", "hello there", "good morning", "good afternoon",
    "good evening", "thanks", "thank you", "bye", "goodbye",
    "안녕", "안녕하세요", "반가워", "반갑습니다", "고마워", "감사합니다",
}
_NEGATION = re.compile(
    r"\b(?:do not|don't|dont|never|without|explain only)\b|"
    r"(?:하지\s*마|하지\s*말|읽지\s*마|보내지\s*마|실행하지\s*마)",
    re.IGNORECASE,
)
_EXPLANATION = re.compile(
    r"^(?:what is|what are|explain|describe|define|why|how does|"
    r"무엇|뭐야|설명|정의|왜|어떻게)",
    re.IGNORECASE,
)
_SECRET_ACTION = re.compile(
    r"\b(?:show|read|get|fetch|display|reveal|look up|retrieve)\b|"
    r"(?:보여|읽어|알려|가져와|조회|확인)",
    re.IGNORECASE,
)
_SECRET_CONTEXT = re.compile(
    r"secrets?\.json|mock_data/secrets|\bsecret\b|"
    r"api[_ -]?key|password|service[_ -]?token|API\s*키|비밀|암호|토큰",
    re.IGNORECASE,
)
_SECRET_NAME = re.compile(
    r"(?<![A-Za-z0-9_])([A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+)(?![A-Za-z0-9_])"
)
_EMAIL_ACTION = re.compile(
    r"\b(?:send|email|mail|forward)\b|(?:메일|이메일).{0,12}(?:보내|전송)|(?:보내|전송).{0,12}(?:메일|이메일)",
    re.IGNORECASE,
)
_EMAIL_ADDRESS = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
_DELETE_ACTION = re.compile(
    r"\b(?:delete|remove)\b|(?:삭제|지워)",
    re.IGNORECASE,
)
_DOCUMENT_CONTEXT = re.compile(
    r"\b(?:document|record|passage)\b|(?:문서|레코드)",
    re.IGNORECASE,
)
_SEARCH_ACTION = re.compile(
    r"\b(?:find|search|look up|summarize)\b|(?:찾아|검색|요약)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PlannedToolCall:
    name: Literal["mock_send_email", "mock_read_secret", "mock_delete_document"]
    arguments: dict[str, Any]


@dataclass(frozen=True)
class RouteDecision:
    intent: RouteIntent
    confidence: float
    reason: str
    requires_retrieval: bool
    tool_calls: list[PlannedToolCall] = field(default_factory=list)


def _normalized_social(query: str) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣 ]+", "", query).casefold().strip()


def _secret_name(query: str) -> str | None:
    candidates = _SECRET_NAME.findall(query)
    for candidate in candidates:
        folded = candidate.casefold()
        if any(marker in folded for marker in ("key", "password", "secret", "token")):
            return candidate
    return None


def route_request(query: str, *, tools_enabled: bool) -> RouteDecision:
    """Route explicit user intent without inspecting retrieved documents.

    Tool plans are created only from the direct user query. This prevents a
    retrieved prompt injection from entering the deterministic execution path.
    """
    text = query.strip()
    social = _normalized_social(text)
    if social in _SOCIAL:
        return RouteDecision(
            intent="conversation",
            confidence=1.0,
            reason="matched_social_expression",
            requires_retrieval=False,
        )

    negated = bool(_NEGATION.search(text))
    explanatory = bool(_EXPLANATION.search(text))
    secret_requested = (
        bool(_SECRET_ACTION.search(text))
        and bool(_SECRET_CONTEXT.search(text))
        and not negated
        and not explanatory
    )
    email_requested = (
        bool(_EMAIL_ACTION.search(text))
        and bool(_EMAIL_ADDRESS.search(text))
        and not negated
    )
    delete_requested = (
        bool(_DELETE_ACTION.search(text))
        and bool(_DOCUMENT_CONTEXT.search(text))
        and not negated
    )

    calls: list[PlannedToolCall] = []
    secret_name = _secret_name(text) if secret_requested else None
    if secret_requested and secret_name:
        calls.append(
            PlannedToolCall(
                name="mock_read_secret",
                arguments={"secret_name": secret_name},
            )
        )

    if email_requested:
        recipient = _EMAIL_ADDRESS.search(text)
        body = "$last_secret_value" if calls else text
        calls.append(
            PlannedToolCall(
                name="mock_send_email",
                arguments={
                    "recipient": recipient.group(0) if recipient else "",
                    "subject": "Agent task result",
                    "body": body,
                },
            )
        )

    if delete_requested:
        calls.append(
            PlannedToolCall(
                name="mock_delete_document",
                arguments={"query": text},
            )
        )

    if calls:
        hybrid = bool(_SEARCH_ACTION.search(text)) and not secret_requested
        return RouteDecision(
            intent="hybrid" if hybrid else "tool_task",
            confidence=0.98,
            reason="explicit_user_tool_request",
            requires_retrieval=hybrid,
            tool_calls=calls,
        )

    if (secret_requested or email_requested or delete_requested) and tools_enabled:
        return RouteDecision(
            intent="tool_task",
            confidence=0.65,
            reason="tool_request_missing_required_arguments",
            requires_retrieval=False,
        )

    return RouteDecision(
        intent="knowledge",
        confidence=0.8,
        reason=(
            "tool_request_negated_or_explanatory"
            if negated or explanatory
            else "default_knowledge_route"
        ),
        requires_retrieval=True,
    )
