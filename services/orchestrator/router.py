"""Safe fast paths and structured semantic routing for agent requests."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from services.orchestrator.memory import extract_user_facts, is_personal_memory_query

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
_CONNECTED_EMAIL_TARGET = re.compile(
    r"\b(?:connected|linked)\s+(?:email|mailbox|gmail)\b|"
    r"(?:연결된|연동된)\s*(?:이메일|메일|지메일)",
    re.IGNORECASE,
)
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
_RETRIEVAL_SCOPE = re.compile(
    r"\b(?:document|file|email|mail|inbox|drive|database|knowledge base|corpus|"
    r"repository|report|passage|record|gmail|local db|retrieved context)\b|"
    r"(?:문서|파일|메일|이메일|받은편지함|드라이브|데이터베이스|지식\s*베이스|"
    r"코퍼스|저장소|보고서|자료|레코드|검색\s*결과)",
    re.IGNORECASE,
)
_DEICTIC_SCOPE = re.compile(
    r"\b(?:my|our|this|that|these|those|uploaded|connected)\s+"
    r"(?:document|file(?!\s+system)|email|mail|inbox|drive|database|report|record)s?\b|"
    r"(?:내|제|우리|이|그|해당|업로드한|연결된)\s*"
    r"(?:문서|파일|메일|이메일|받은편지함|드라이브|데이터베이스|보고서)",
    re.IGNORECASE,
)
_RESULT_DEPENDENCY = re.compile(
    r"\b(?:it|that|the result|the answer|the summary|the translation)\b|"
    r"(?:그걸|그것을|결과|답변|요약|번역).{0,16}(?:보내|전송)",
    re.IGNORECASE,
)
_MODEL_STEP_KINDS = {
    "retrieve", "generate", "tool", "respond", "memory_read", "memory_write"
}


@dataclass(frozen=True)
class PlannedToolCall:
    name: Literal["mock_send_email", "mock_read_secret", "mock_delete_document"]
    arguments: dict[str, Any]


@dataclass(frozen=True)
class PlannedStep:
    step_id: str
    kind: Literal[
        "retrieve", "generate", "tool", "respond", "memory_read", "memory_write"
    ]
    instruction: str = ""
    depends_on: list[str] = field(default_factory=list)
    output_key: str = ""
    tool_name: str | None = None


@dataclass(frozen=True)
class RouteDecision:
    intent: RouteIntent
    confidence: float
    reason: str
    requires_retrieval: bool
    requires_generation: bool = False
    suggested_steps: list[str] = field(default_factory=list)
    planned_steps: list[PlannedStep] = field(default_factory=list)
    direct_answer: str | None = None
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
def is_explicit_retrieval_request(query: str) -> bool:
    text = query.strip()
    return (
        text.casefold().startswith(("@rag ", "rag:"))
        or (
            bool(_DEICTIC_SCOPE.search(text))
            and bool(_SEARCH_ACTION.search(text))
        )
        or (
            bool(_RETRIEVAL_SCOPE.search(text))
            and bool(_SEARCH_ACTION.search(text))
        )
    )


def _json_object(raw: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, character in enumerate(raw):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(raw[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("router model did not return a JSON object")


def _json_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    return False


def route_from_model_output(
    raw: str,
    *,
    retrieval_policy: RetrievalPolicy,
    authorized_tool_names: set[str] | None = None,
) -> RouteDecision:
    """Validate the semantic planner output before the executor uses it."""
    payload = _json_object(raw)
    raw_intent = str(payload.get("intent", "general")).casefold()
    intent_map: dict[str, RouteIntent] = {
        "conversation": "conversation",
        "memory": "conversation",
        "general": "knowledge",
        "knowledge": "knowledge",
        "retrieval": "knowledge",
        "tool": "tool_task",
        "hybrid": "hybrid",
    }
    intent = intent_map.get(raw_intent, "knowledge")
    requires_retrieval = _json_bool(payload.get("requires_retrieval", False))
    if retrieval_policy == "always" and intent not in {"conversation", "tool_task"}:
        requires_retrieval = True
    elif retrieval_policy == "never":
        requires_retrieval = False

    raw_steps = payload.get("steps", [])
    if not isinstance(raw_steps, list):
        raw_steps = []
    steps: list[str] = []
    planned_steps: list[PlannedStep] = []
    known_ids: set[str] = set()
    for index, raw_step in enumerate(raw_steps[:8], start=1):
        if isinstance(raw_step, str):
            kind = raw_step if raw_step in _MODEL_STEP_KINDS else ""
            raw_item: dict[str, Any] = {}
        elif isinstance(raw_step, dict):
            kind = str(raw_step.get("kind", ""))
            raw_item = raw_step
        else:
            continue
        if kind not in _MODEL_STEP_KINDS:
            continue
        tool_name = (
            str(raw_item.get("tool_name", "")).strip() or None
            if kind == "tool"
            else None
        )
        if kind == "tool" and (
            not tool_name
            or tool_name not in (authorized_tool_names or set())
        ):
            continue
        candidate_id = str(raw_item.get("id", f"step-{index}"))
        step_id = (
            candidate_id
            if re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,39}", candidate_id)
            and candidate_id not in known_ids
            else f"step-{index}"
        )
        dependencies = [
            str(item)
            for item in raw_item.get("depends_on", [])
            if isinstance(item, str) and item in known_ids
        ][:4]
        if not dependencies and planned_steps:
            dependencies = [planned_steps[-1].step_id]
        instruction = " ".join(
            str(raw_item.get("instruction", "")).split()
        )[:300]
        output_key = str(
            raw_item.get("output_key", f"result_{index}")
        )[:40]
        planned_steps.append(
            PlannedStep(
                step_id=step_id,
                kind=kind,
                instruction=instruction,
                depends_on=dependencies,
                output_key=output_key,
                tool_name=tool_name,
            )
        )
        known_ids.add(step_id)
        steps.append(
            f"tool:{tool_name}" if kind == "tool" and tool_name else kind
        )
    if not requires_retrieval:
        steps = [step for step in steps if step != "retrieve"]
        planned_steps = [
            step for step in planned_steps if step.kind != "retrieve"
        ]
    if not steps:
        steps = ["retrieve", "generate"] if requires_retrieval else ["generate"]
    if requires_retrieval and "retrieve" not in steps:
        steps.insert(0, "retrieve")

    direct_answer = payload.get("answer")
    if not isinstance(direct_answer, str) or not direct_answer.strip():
        direct_answer = None
    if requires_retrieval:
        direct_answer = None

    try:
        confidence = min(1.0, max(0.0, float(payload.get("confidence", 0.7))))
    except (TypeError, ValueError):
        confidence = 0.7
    model_reason = " ".join(str(payload.get("reason", "semantic decision")).split())[:120]
    return RouteDecision(
        intent=intent,
        confidence=confidence,
        reason=f"llm_semantic_planner:{model_reason}",
        requires_retrieval=requires_retrieval,
        requires_generation=not requires_retrieval and direct_answer is None,
        suggested_steps=steps,
        planned_steps=planned_steps,
        direct_answer=direct_answer,
    )


def route_request(
    query: str,
    *,
    tools_enabled: bool,
    retrieval_policy: RetrievalPolicy = "auto",
) -> RouteDecision:
    """Route explicit user intent without inspecting retrieved documents.

    Tool plans are created only from the direct user query. This prevents a
    retrieved prompt injection from entering the deterministic execution path.
    """
    text = query.strip()
    if is_personal_memory_query(text):
        fact_write = bool(extract_user_facts(text))
        return RouteDecision(
            intent="conversation",
            confidence=1.0,
            reason="personal_fact_write" if fact_write else "personal_memory_recall",
            requires_retrieval=False,
            suggested_steps=(
                ["memory_write", "respond"]
                if fact_write
                else ["memory_read", "respond"]
            ),
        )
    social = _normalized_social(text)
    if social in _SOCIAL:
        return RouteDecision(
            intent="conversation",
            confidence=1.0,
            reason="matched_social_expression",
            requires_retrieval=False,
            suggested_steps=["respond"],
        )

    negated = bool(_NEGATION.search(text))
    explanatory = bool(_EXPLANATION.search(text))
    retrieval_requested = is_explicit_retrieval_request(text)
    secret_requested = (
        bool(_SECRET_ACTION.search(text))
        and bool(_SECRET_CONTEXT.search(text))
        and not negated
        and not explanatory
    )
    email_requested = (
        bool(_EMAIL_ACTION.search(text))
        and bool(
            _EMAIL_ADDRESS.search(text)
            or _CONNECTED_EMAIL_TARGET.search(text)
        )
        and not negated
    )
    delete_requested = (
        bool(_DELETE_ACTION.search(text))
        and bool(_DOCUMENT_CONTEXT.search(text))
        and not negated
    )

    calls: list[PlannedToolCall] = []
    generation_requested = False
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
        generation_requested = retrieval_requested or (
            not calls and bool(_RESULT_DEPENDENCY.search(text))
        )
        body = (
            "$last_secret_value" if calls
            else "$last_answer" if generation_requested
            else text
        )
        calls.append(
            PlannedToolCall(
                name="mock_send_email",
                arguments={
                    "recipient": (
                        recipient.group(0) if recipient else "$connected_email"
                    ),
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
        requires_retrieval = (
            retrieval_policy != "never" and retrieval_requested
        )
        hybrid = requires_retrieval or generation_requested
        steps: list[str] = []
        if requires_retrieval:
            steps.append("retrieve")
        if generation_requested:
            steps.append("generate")
        steps.extend(f"tool:{call.name}" for call in calls)
        return RouteDecision(
            intent="hybrid" if hybrid else "tool_task",
            confidence=0.98,
            reason="explicit_user_multistep_request" if hybrid else "explicit_user_tool_request",
            requires_retrieval=requires_retrieval,
            requires_generation=generation_requested,
            suggested_steps=steps,
            tool_calls=calls,
        )

    if (secret_requested or email_requested or delete_requested) and tools_enabled:
        return RouteDecision(
            intent="tool_task",
            confidence=0.65,
            reason="tool_request_missing_required_arguments",
            requires_retrieval=False,
        )

    if retrieval_policy == "always":
        requires_retrieval = True
        reason = "retrieval_policy_always"
        confidence = 1.0
    elif retrieval_policy == "never":
        requires_retrieval = False
        reason = "retrieval_policy_never"
        confidence = 1.0
    elif retrieval_requested:
        requires_retrieval = True
        reason = "explicit_retrieval_scope"
        confidence = 0.98
    else:
        requires_retrieval = False
        reason = "semantic_router_required"
        confidence = 0.5
    return RouteDecision(
        intent="knowledge",
        confidence=confidence,
        reason=reason,
        requires_retrieval=requires_retrieval,
        requires_generation=not requires_retrieval,
        suggested_steps=["retrieve", "generate"] if requires_retrieval else ["generate"],
    )


def needs_semantic_planner(route: RouteDecision) -> bool:
    return (
        route.reason in {"semantic_router_required", "retrieval_policy_never"}
        or route.requires_retrieval
        or route.requires_generation
        or bool(route.tool_calls)
    )
