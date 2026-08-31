from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import sys
from collections.abc import Callable, Sequence
from typing import Any
from uuid import uuid4

import httpx
from pydantic import ValidationError

from agent_system.application.tool_calling_agent import (
    AgentRunResponse,
    AgentRunStatus,
)
MEMORY_CONTEXT_RE = re.compile(r"^context[1-9][0-9]{0,5}$")


def validate_memory_context(context_id: str) -> str:
    if not MEMORY_CONTEXT_RE.fullmatch(context_id):
        raise ValueError("memory_context must use context1, context2, ...")
    return context_id


InputFunction = Callable[[str], str]
OutputFunction = Callable[[str], None]


class AgentApiError(RuntimeError):
    """A user-facing error returned while communicating with the agent API."""


class AgentApiClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 180.0,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._owns_client = http_client is None
        self._http = http_client or httpx.Client(timeout=timeout_seconds)

    def close(self) -> None:
        if self._owns_client:
            self._http.close()

    def ready(self) -> dict[str, Any]:
        return self._request("GET", "/ready")

    def query(
        self,
        *,
        user_id: str,
        session_id: str,
        memory_context: str,
        query: str,
    ) -> AgentRunResponse:
        payload = self._request(
            "POST",
            "/v1/agent/query",
            json={
                "user_id": user_id,
                "session_id": session_id,
                "memory_context": memory_context,
                "query": query,
            },
        )
        return self._agent_response(payload, memory_context=memory_context)

    def approve(
        self,
        workflow_id: str,
        *,
        user_id: str,
        session_id: str,
        memory_context: str,
        task_ids: set[str],
    ) -> AgentRunResponse:
        payload = self._request(
            "POST",
            f"/v1/agent/workflows/{workflow_id}/approve",
            json={
                "user_id": user_id,
                "session_id": session_id,
                "approved_task_ids": sorted(task_ids),
            },
        )
        return self._agent_response(payload, memory_context=memory_context)

    @staticmethod
    def _agent_response(
        payload: Any,
        *,
        memory_context: str,
    ) -> AgentRunResponse:
        if not isinstance(payload, dict):
            raise AgentApiError("서버가 잘못된 에이전트 응답을 반환했습니다.")
        compatible_payload = dict(payload)
        compatible_payload.setdefault("memory_context", memory_context)
        compatible_payload.setdefault("stored_memories", [])
        try:
            return AgentRunResponse.model_validate(compatible_payload)
        except ValidationError as exc:
            raise AgentApiError(
                f"서버의 에이전트 응답 스키마가 호환되지 않습니다: {exc}"
            ) from exc

    def cancel(
        self,
        workflow_id: str,
        *,
        user_id: str,
        session_id: str,
    ) -> None:
        self._request(
            "DELETE",
            f"/v1/agent/workflows/{workflow_id}",
            params={"user_id": user_id, "session_id": session_id},
        )

    def permissions(self) -> list[str]:
        payload = self._request("GET", "/v1/permissions")
        if not isinstance(payload, list):
            raise AgentApiError("서버가 잘못된 권한 응답을 반환했습니다.")
        return [str(item) for item in payload]

    def memory_contexts(self) -> list[str]:
        payload = self._request("GET", "/v1/memory-contexts")
        if not isinstance(payload, list):
            raise AgentApiError("서버가 잘못된 메모리 컨텍스트 응답을 반환했습니다.")
        return [str(item) for item in payload]

    def memories(self, user_id: str, memory_context: str) -> list[dict[str, Any]]:
        payload = self._request(
            "GET",
            f"/v1/memories/{user_id}",
            params={"memory_context": memory_context},
        )
        if not isinstance(payload, list):
            raise AgentApiError("서버가 잘못된 장기 메모리 응답을 반환했습니다.")
        return payload

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            response = self._http.request(method, f"{self.base_url}{path}", **kwargs)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise AgentApiError(self._error_detail(exc.response)) from exc
        except httpx.RequestError as exc:
            raise AgentApiError(
                f"에이전트 서버에 연결할 수 없습니다: {exc}"
            ) from exc

        if response.status_code == 204 or not response.content:
            return None
        try:
            return response.json()
        except ValueError as exc:
            raise AgentApiError("서버가 JSON이 아닌 응답을 반환했습니다.") from exc

    @staticmethod
    def _error_detail(response: httpx.Response) -> str:
        try:
            payload = response.json()
        except ValueError:
            payload = None
        if isinstance(payload, dict) and "detail" in payload:
            detail = payload["detail"]
            if not isinstance(detail, str):
                detail = json.dumps(detail, ensure_ascii=False)
            return f"에이전트 API 오류 ({response.status_code}): {detail}"
        text = response.text.strip()
        suffix = f": {text[:500]}" if text else ""
        return f"에이전트 API 오류 ({response.status_code}){suffix}"


class AgentShell:
    def __init__(
        self,
        client: AgentApiClient,
        *,
        user_id: str,
        session_id: str,
        memory_context: str,
        show_results: bool = False,
        input_fn: InputFunction = input,
        output_fn: OutputFunction = print,
    ) -> None:
        self.client = client
        self.user_id = user_id
        self.session_id = session_id
        self.memory_context = validate_memory_context(memory_context)
        self.show_results = show_results
        self._input = input_fn
        self._output = output_fn

    def run(self) -> int:
        self._output("Agent CLI에 연결되었습니다. /help에서 명령을 확인할 수 있습니다.")
        self._output(self._status_line())
        while True:
            try:
                line = self._input("you> ").strip()
            except EOFError:
                self._output("")
                return 0
            except KeyboardInterrupt:
                self._output("\n종료합니다.")
                return 130

            if not line:
                continue
            try:
                if line.startswith("/"):
                    if not self._command(line):
                        return 0
                else:
                    self.ask(line)
            except AgentApiError as exc:
                self._output(f"error> {exc}")
            except ValueError as exc:
                self._output(f"error> {exc}")
            except KeyboardInterrupt:
                self._output("\n요청 입력으로 돌아갑니다.")

    def ask(self, query: str) -> AgentRunResponse | None:
        self._output("agent> 처리 중...")
        response = self.client.query(
            user_id=self.user_id,
            session_id=self.session_id,
            memory_context=self.memory_context,
            query=query,
        )
        while response.status == AgentRunStatus.AWAITING_APPROVAL:
            self._print_approval(response)
            if not self._confirm("위 작업을 모두 승인할까요? [y/N] "):
                self.client.cancel(
                    response.workflow_id,
                    user_id=self.user_id,
                    session_id=self.session_id,
                )
                self._output("agent> 작업을 승인하지 않아 취소했습니다.")
                return None
            response = self.client.approve(
                response.workflow_id,
                user_id=self.user_id,
                session_id=self.session_id,
                memory_context=self.memory_context,
                task_ids={item.task_id for item in response.approval_requests},
            )

        self._output(f"agent> {response.answer}")
        self._print_details(response)
        return response

    def _command(self, line: str) -> bool:
        try:
            parts = shlex.split(line)
        except ValueError as exc:
            raise ValueError(f"명령을 해석할 수 없습니다: {exc}") from exc
        command = parts[0].casefold()
        arguments = parts[1:]

        if command in {"/exit", "/quit"}:
            return False
        if command == "/help":
            self._output(HELP_TEXT)
        elif command == "/status":
            self._output(self._status_line())
        elif command == "/new":
            self.session_id = arguments[0] if arguments else _new_session_id()
            self._output(f"새 세션을 시작했습니다: {self.session_id}")
        elif command == "/context":
            self._set_context(arguments)
        elif command == "/contexts":
            contexts = self.client.memory_contexts()
            self._output("메모리 컨텍스트: " + ", ".join(contexts))
        elif command == "/permissions":
            self._set_permissions(arguments)
        elif command == "/memories":
            self._print_memories()
        else:
            self._output(f"알 수 없는 명령입니다: {parts[0]} (/help 참고)")
        return True

    def _set_context(self, arguments: list[str]) -> None:
        if not arguments:
            self._output(f"현재 메모리 컨텍스트: {self.memory_context}")
            return
        if len(arguments) != 1:
            raise ValueError("사용법: /context contextN")
        self.memory_context = validate_memory_context(arguments[0])
        self._output(f"메모리 컨텍스트를 변경했습니다: {self.memory_context}")

    def _set_permissions(self, arguments: list[str]) -> None:
        if arguments:
            raise ValueError(
                "권한은 서버 관리 항목입니다. /permissions는 조회만 지원합니다."
            )
        permissions = self.client.permissions()
        self._output("서버 권한: " + ", ".join(permissions))

    def _print_approval(self, response: AgentRunResponse) -> None:
        self._output("approval> 사용자 승인이 필요한 작업입니다.")
        for index, item in enumerate(response.approval_requests, start=1):
            parameters = json.dumps(
                item.parameters, ensure_ascii=False, indent=2, sort_keys=True
            )
            self._output(
                f"  [{index}] {item.executor}.{item.action}\n"
                f"      task_id: {item.task_id}\n"
                f"      risk: {item.risk}\n"
                f"      parameters: {parameters}"
            )

    def _print_details(self, response: AgentRunResponse) -> None:
        if self.show_results and response.results:
            self._output("tool results>")
            for result in response.results:
                self._output(
                    json.dumps(
                        result.model_dump(mode="json"),
                        ensure_ascii=False,
                        indent=2,
                    )
                )
        if response.stored_memories:
            contents = "; ".join(item.content for item in response.stored_memories)
            self._output(f"memory> 장기 메모리에 저장: {contents}")

    def _print_memories(self) -> None:
        memories = self.client.memories(self.user_id, self.memory_context)
        if not memories:
            self._output("저장된 장기 메모리가 없습니다.")
            return
        self._output(f"장기 메모리 ({self.memory_context}):")
        for item in memories:
            category = item.get("metadata", {}).get("category")
            label = f" [{category}]" if category else ""
            self._output(f"  - {item.get('content', '')}{label}")

    def _confirm(self, prompt: str) -> bool:
        try:
            answer = self._input(prompt).strip().casefold()
        except (EOFError, KeyboardInterrupt):
            return False
        return answer in {"y", "yes", "예", "승인"}

    def _status_line(self) -> str:
        return (
            f"user={self.user_id} session={self.session_id} "
            f"memory={self.memory_context}"
        )


HELP_TEXT = """사용 가능한 명령:
  /help                          명령 목록
  /status                        현재 사용자·세션·메모리 컨텍스트
  /new [session-id]              새 대화 세션 시작
  /context [contextN]            장기 메모리 컨텍스트 조회·변경
  /contexts                      서버의 메모리 컨텍스트 목록
  /permissions                   서버가 부여한 권한 조회
  /memories                      현재 컨텍스트의 장기 메모리 조회
  /exit                          종료"""


def _new_session_id() -> str:
    return f"cli-{uuid4()}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-cli",
        description="대화·도구 실행·승인 처리를 지원하는 Agent API CLI",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("AGENT_API_URL", "http://localhost:19000"),
        help="Agent API 주소 (기본값: %(default)s)",
    )
    parser.add_argument(
        "--user-id",
        default=os.getenv("AGENT_USER_ID", "user-1"),
        help="대화 및 메모리를 구분할 사용자 ID",
    )
    parser.add_argument(
        "--session-id",
        default=os.getenv("AGENT_SESSION_ID") or _new_session_id(),
        help="기존 대화를 이어갈 세션 ID",
    )
    parser.add_argument(
        "--memory-context",
        default=os.getenv("AGENT_MEMORY_CONTEXT", "context1"),
        help="연결할 장기 메모리 컨텍스트",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.getenv("AGENT_CLI_TIMEOUT_SECONDS", "180")),
        help="API 요청 제한 시간(초)",
    )
    parser.add_argument(
        "--show-results",
        action="store_true",
        help="최종 답변과 함께 원본 도구 실행 결과 표시",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        memory_context = validate_memory_context(args.memory_context)
    except ValueError as exc:
        print(f"error> {exc}", file=sys.stderr)
        return 2

    client = AgentApiClient(args.base_url, timeout_seconds=args.timeout)
    shell = AgentShell(
        client,
        user_id=args.user_id,
        session_id=args.session_id,
        memory_context=memory_context,
        show_results=args.show_results,
    )
    try:
        client.ready()
        return shell.run()
    except AgentApiError as exc:
        print(f"error> {exc}", file=sys.stderr)
        return 1
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
