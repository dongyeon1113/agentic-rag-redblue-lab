#!/usr/bin/env python3
"""Interactive CLI client for the agentic RAG orchestrator.

The client talks only to the orchestrator HTTP API. With --tools, the server may
run its allowlisted local mock tools; it never grants access to arbitrary shell
commands or external tools.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from dataclasses import dataclass, field
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen
from uuid import uuid4

SOURCE_CHOICES = ("local_db", "gmail", "drive")
MODEL_CHOICES = ("qwen3:8b", "llama3.2:3b", "llama3.2:1b")
MODE_CHOICES = ("vulnerable", "defended")
DEFENSE_CHOICES = ("none", "ragpart")
RETRIEVAL_POLICY_CHOICES = ("auto", "always", "never")


class AgentCLIError(RuntimeError):
    """A readable API or transport error."""


@dataclass
class CLIConfig:
    base_url: str = "http://127.0.0.1:8000"
    session_id: str = field(default_factory=lambda: f"cli-{uuid4().hex[:12]}")
    sources: list[str] = field(default_factory=lambda: list(SOURCE_CHOICES))
    model: str = "qwen3:8b"
    mode: str = "vulnerable"
    retrieval_defense: str = "none"
    retrieval_policy: str = "auto"
    limit: int = 6
    use_memory: bool = True
    enable_tools: bool = False
    regex_filter: bool = False
    prompt_guard: bool = False
    context_capacity: int | None = None
    spotlighting: list[str] = field(default_factory=list)
    timeout: float = 120.0
    show_documents: bool = False
    show_memory: bool = True
    show_tool_calls: bool = True
    raw_json: bool = False

    def answer_payload(self, query: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "query": query,
            "sources": self.sources,
            "limit": self.limit,
            "answer_model": self.model,
            "mode": self.mode,
            "retrieval_defense": self.retrieval_defense,
            "retrieval_policy": self.retrieval_policy,
            "session_id": self.session_id,
            "use_memory": self.use_memory,
            "enable_mock_tools": self.enable_tools,
            "regex_filter": self.regex_filter,
            "prompt_guard": self.prompt_guard,
        }
        if self.context_capacity is not None:
            payload["context_capacity"] = self.context_capacity
        return payload


def api_request(
    config: CLIConfig,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    url = f"{config.base_url.rstrip('/')}{path}"
    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(url, data=body, headers=headers, method=method)
    try:
        with urlopen(request, timeout=config.timeout) as response:
            content = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(detail)
            detail = json.dumps(parsed, ensure_ascii=False)
        except json.JSONDecodeError:
            pass
        raise AgentCLIError(f"HTTP {exc.code} {exc.reason}: {detail}") from exc
    except URLError as exc:
        raise AgentCLIError(
            f"오케스트레이터에 연결할 수 없습니다: {exc.reason}. "
            "docker compose ps와 --base-url을 확인하세요."
        ) from exc
    except TimeoutError as exc:
        raise AgentCLIError(
            f"{config.timeout:g}초 안에 응답하지 않았습니다. "
            "--timeout 값을 늘리거나 서버 로그를 확인하세요."
        ) from exc

    if not content:
        return {}
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise AgentCLIError(f"서버가 JSON이 아닌 응답을 반환했습니다: {content[:300]}") from exc
    if not isinstance(parsed, dict):
        raise AgentCLIError("서버 응답의 최상위 값이 JSON object가 아닙니다.")
    return parsed


def answer_path(config: CLIConfig) -> str:
    params = {method: "true" for method in config.spotlighting}
    return f"/answer?{urlencode(params)}" if params else "/answer"


def print_rule(title: str) -> None:
    print(f"\n--- {title} ---")


def compact_text(value: Any, width: int = 180) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= width else f"{text[: width - 1]}…"


def print_response(config: CLIConfig, response: dict[str, Any]) -> None:
    if config.raw_json:
        print(json.dumps(response, ensure_ascii=False, indent=2))
        return

    intent = response.get("intent")
    if intent:
        retrieval = "skipped" if response.get("retrieval_skipped") else "used"
        print(f"\nroute={intent} confidence={response.get('route_confidence')} retrieval={retrieval}")
    print_rule("에이전트 답변")
    print(response.get("answer", "(답변 없음)"))
    execution_plan = response.get("execution_plan") or []
    step_results = {
        item.get("step_id"): item
        for item in (response.get("step_results") or [])
    }
    if execution_plan:
        print_rule(f"실행 계획 ({len(execution_plan)}단계)")
        for step in execution_plan:
            result = step_results.get(step.get("id"), {})
            label = step.get("name") or step.get("kind", "?")
            status = result.get("status", "planned")
            dependencies = ",".join(step.get("depends_on") or []) or "-"
            print(f"[{step.get('id')}] {label} status={status} depends_on={dependencies}")
            if result.get("output"):
                print(f"    {compact_text(result['output'], 240)}")


    tool_calls = response.get("tool_calls") or []
    if config.show_tool_calls and tool_calls:
        print_rule(f"도구 실행 감사 로그 ({len(tool_calls)})")
        for index, call in enumerate(tool_calls, start=1):
            print(
                f"[{index}] {call.get('name', '?')} "
                f"status={call.get('status', '?')}"
            )
            print(f"    arguments: {json.dumps(call.get('arguments', {}), ensure_ascii=False)}")
            print(f"    result: {compact_text(call.get('result'), 300)}")

    memory = response.get("memory") or []
    if config.show_memory:
        print_rule(f"회상된 장기 메모리 ({len(memory)})")
        if not memory:
            print("(없음)")
        for index, item in enumerate(memory, start=1):
            print(
                f"[{index}] {item.get('document_id')} "
                f"trust={item.get('trust')} score={item.get('score')}"
            )
            print(f"    {compact_text(item.get('text'))}")

    blocked = response.get("blocked_documents") or []
    if blocked:
        print_rule(f"방어 필터 차단 문서 ({len(blocked)})")
        for index, item in enumerate(blocked, start=1):
            print(f"[{index}] {json.dumps(item, ensure_ascii=False)}")

    if config.show_documents:
        documents = response.get("documents") or []
        print_rule(f"최종 컨텍스트 문서 ({len(documents)})")
        for index, item in enumerate(documents, start=1):
            print(
                f"[{index}] {item.get('document_id')} source={item.get('source')} "
                f"trust={item.get('trust')} score={item.get('score')}"
            )
            print(f"    {compact_text(item.get('text'), 240)}")


def ask(config: CLIConfig, query: str) -> bool:
    try:
        response = api_request(
            config,
            "POST",
            answer_path(config),
            config.answer_payload(query),
        )
    except (AgentCLIError, ValueError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return False
    print_response(config, response)
    return True


def print_config(config: CLIConfig) -> None:
    print(
        json.dumps(
            {
                "base_url": config.base_url,
                "session_id": config.session_id,
                "sources": config.sources,
                "model": config.model,
                "mode": config.mode,
                "retrieval_defense": config.retrieval_defense,
                "retrieval_policy": config.retrieval_policy,
                "limit": config.limit,
                "use_memory": config.use_memory,
                "enable_mock_tools": config.enable_tools,
                "regex_filter": config.regex_filter,
                "prompt_guard": config.prompt_guard,
                "context_capacity": config.context_capacity,
                "spotlighting": config.spotlighting,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def show_memory(config: CLIConfig) -> None:
    path = f"/memory?{urlencode({'session_id': config.session_id, 'limit': 50})}"
    try:
        response = api_request(config, "GET", path)
    except AgentCLIError as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return
    print(json.dumps(response, ensure_ascii=False, indent=2))


def clear_memory(config: CLIConfig) -> None:
    confirmation = input(
        f"세션 {config.session_id!r}의 메모리를 삭제할까요? [y/N] "
    ).strip().casefold()
    if confirmation not in {"y", "yes"}:
        print("취소했습니다.")
        return
    path = f"/memory?{urlencode({'session_id': config.session_id})}"
    try:
        response = api_request(config, "DELETE", path)
    except AgentCLIError as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return
    print(json.dumps(response, ensure_ascii=False, indent=2))


def base_url_with_port(url: str, port: int | None) -> str:
    candidate = url if "://" in url else f"http://{url}"
    parsed = urlsplit(candidate)
    if not parsed.hostname:
        raise ValueError("URL에 host가 필요합니다.")
    selected_port = port if port is not None else parsed.port or 8000
    if not 1 <= selected_port <= 65535:
        raise ValueError("port는 1..65535여야 합니다.")
    host = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
    netloc = f"{host}:{selected_port}"
    return urlunsplit((parsed.scheme or "http", netloc, parsed.path.rstrip("/"), "", ""))


def set_option(config: CLIConfig, arguments: list[str]) -> None:
    if len(arguments) < 2:
        print("사용법: /set OPTION VALUE")
        return
    name, values = arguments[0], arguments[1:]
    value = " ".join(values)
    try:
        if name == "url":
            config.base_url = base_url_with_port(value, None)
        elif name == "port":
            config.base_url = base_url_with_port(config.base_url, int(value))
        elif name == "mode" and value in MODE_CHOICES:
            config.mode = value
        elif name == "model" and value in MODEL_CHOICES:
            config.model = value
        elif name == "defense" and value in DEFENSE_CHOICES:
            config.retrieval_defense = value
        elif name == "retrieval" and value in RETRIEVAL_POLICY_CHOICES:
            config.retrieval_policy = value
        elif name == "sources":
            sources = [item.strip() for item in value.split(",") if item.strip()]
            invalid = set(sources) - set(SOURCE_CHOICES)
            if not sources or invalid:
                raise ValueError(f"허용되지 않은 source: {sorted(invalid)}")
            config.sources = sources
        elif name == "limit":
            limit = int(value)
            if not 1 <= limit <= 20:
                raise ValueError("limit은 1..20이어야 합니다.")
            config.limit = limit
        elif name in {"memory", "tools", "regex", "guard", "documents"}:
            enabled = value.casefold() in {"1", "true", "on", "yes"}
            attribute = {
                "memory": "use_memory",
                "tools": "enable_tools",
                "regex": "regex_filter",
                "guard": "prompt_guard",
                "documents": "show_documents",
            }[name]
            setattr(config, attribute, enabled)
        else:
            raise ValueError(f"알 수 없거나 잘못된 옵션: {name}={value}")
    except ValueError as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return
    print_config(config)


HELP = """
일반 문장을 입력하면 에이전트에 전달합니다.

REPL 명령:
  /help                   도움말
  /config                 현재 설정
  /set url URL            오케스트레이터 URL 또는 host
  /set port PORT          오케스트레이터 port
  /set mode MODE          vulnerable 또는 defended
  /set model MODEL        qwen3:8b, llama3.2:3b, llama3.2:1b
  /set defense DEFENSE    none 또는 ragpart
  /set retrieval POLICY  auto, always 또는 never
  /set sources LIST       local_db,gmail,drive 형식
  /set limit N            검색 Top-K
  /set memory on|off      장기 메모리
  /set tools on|off       로컬 mock 도구
  /set regex on|off       regex prompt-injection 필터
  /set guard on|off       Meta Prompt Guard
  /set documents on|off   최종 컨텍스트 출력
  /memory                 현재 세션 메모리 조회
  /clear-memory           현재 세션 메모리 삭제
  /new-session [ID]       새 대화 세션
  /quit                   종료

태스크 예:
  fake test_api_key를 읽어서 알려줘
  보안 보고서를 검색하고 요약해줘
  테스트 이메일을 qa@example.test로 보내줘
""".strip()


def handle_command(config: CLIConfig, line: str) -> bool:
    try:
        parts = shlex.split(line)
    except ValueError as exc:
        print(f"명령 구문 오류: {exc}", file=sys.stderr)
        return True
    command, *arguments = parts
    if command in {"/quit", "/exit"}:
        return False
    if command == "/help":
        print(HELP)
    elif command == "/config":
        print_config(config)
    elif command == "/set":
        set_option(config, arguments)
    elif command == "/memory":
        show_memory(config)
    elif command == "/clear-memory":
        clear_memory(config)
    elif command == "/new-session":
        config.session_id = arguments[0] if arguments else f"cli-{uuid4().hex[:12]}"
        print(f"새 session_id: {config.session_id}")
    else:
        print(f"알 수 없는 명령: {command}. /help를 확인하세요.", file=sys.stderr)
    return True


def interactive(config: CLIConfig) -> int:
    print("Agentic RAG CLI. 일반 문장을 입력하거나 /help를 입력하세요.")
    print(f"session_id={config.session_id} tools={config.enable_tools} mode={config.mode}")
    while True:
        try:
            line = input("\nuser> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n종료합니다.")
            return 0
        if not line:
            continue
        if line.startswith("/"):
            if not handle_command(config, line):
                return 0
            continue
        ask(config, line)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Agentic RAG orchestrator용 단발/대화형 CLI"
    )
    parser.add_argument("query", nargs="?", help="한 번 실행할 질의 또는 태스크")
    parser.add_argument("-i", "--interactive", action="store_true", help="대화형 REPL")
    parser.add_argument(
        "--url",
        default="http://127.0.0.1",
        help="오케스트레이터 scheme과 host; 기본 http://127.0.0.1",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="오케스트레이터 port; URL에 없으면 기본 8000",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="호환 옵션: 완전한 URL. --url보다 우선",
    )
    parser.add_argument("--session-id", default=None)
    parser.add_argument(
        "--sources",
        default="local_db,gmail,drive",
        help="쉼표로 구분: local_db,gmail,drive",
    )
    parser.add_argument("--model", choices=MODEL_CHOICES, default="qwen3:8b")
    parser.add_argument("--mode", choices=MODE_CHOICES, default="vulnerable")
    parser.add_argument("--defense", choices=DEFENSE_CHOICES, default="none")
    parser.add_argument(
        "--retrieval-policy",
        choices=RETRIEVAL_POLICY_CHOICES,
        default="auto",
    )
    parser.add_argument("--limit", type=int, choices=range(1, 21), default=6)
    parser.add_argument("--context-capacity", type=int, choices=range(1, 21))
    parser.add_argument("--no-memory", action="store_true")
    parser.add_argument("--tools", action="store_true", help="로컬 mock 도구 허용")
    parser.add_argument("--regex-filter", action="store_true")
    parser.add_argument("--prompt-guard", action="store_true")
    parser.add_argument(
        "--spotlighting",
        choices=("delimiting", "datamarking", "encoding"),
        action="append",
        default=[],
    )
    parser.add_argument("--show-documents", action="store_true")
    parser.add_argument("--hide-memory", action="store_true")
    parser.add_argument("--hide-tool-calls", action="store_true")
    parser.add_argument("--json", action="store_true", help="원본 JSON 출력")
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> CLIConfig:
    sources = [item.strip() for item in args.sources.split(",") if item.strip()]
    invalid = set(sources) - set(SOURCE_CHOICES)
    if not sources or invalid:
        raise AgentCLIError(f"유효하지 않은 --sources 값: {sorted(invalid)}")
    if args.timeout <= 0:
        raise AgentCLIError("--timeout은 0보다 커야 합니다.")
    return CLIConfig(
        base_url=base_url_with_port(args.base_url or args.url, args.port),
        session_id=args.session_id or f"cli-{uuid4().hex[:12]}",
        sources=sources,
        model=args.model,
        mode=args.mode,
        retrieval_defense=args.defense,
        limit=args.limit,
        retrieval_policy=args.retrieval_policy,
        use_memory=not args.no_memory,
        enable_tools=args.tools,
        regex_filter=args.regex_filter,
        prompt_guard=args.prompt_guard,
        context_capacity=args.context_capacity,
        spotlighting=args.spotlighting,
        timeout=args.timeout,
        show_documents=args.show_documents,
        show_memory=not args.hide_memory,
        show_tool_calls=not args.hide_tool_calls,
        raw_json=args.json,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = config_from_args(args)
    except (AgentCLIError, ValueError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 2

    if args.query:
        success = ask(config, args.query)
        if not success:
            return 1
    if args.interactive or not args.query:
        return interactive(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
