from __future__ import annotations

import json
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, Field

T = TypeVar("T", bound=BaseModel)


class OllamaFunctionCall(BaseModel):
    name: str
    arguments: dict[str, Any] | str = Field(default_factory=dict)

    def parsed_arguments(self) -> dict[str, Any]:
        if isinstance(self.arguments, dict):
            return self.arguments
        value = json.loads(self.arguments)
        if not isinstance(value, dict):
            raise ValueError("Tool arguments must be a JSON object")
        return value


class OllamaToolCall(BaseModel):
    function: OllamaFunctionCall


class OllamaMessage(BaseModel):
    role: str = "assistant"
    content: str = ""
    thinking: str | None = None
    tool_calls: list[OllamaToolCall] = Field(default_factory=list)


class OllamaUnavailableError(RuntimeError):
    pass


class OllamaChatModel:
    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        temperature: float = 0.0,
        num_predict: int = 1024,
        think: bool = True,
        timeout_seconds: float = 120.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._temperature = temperature
        self._num_predict = num_predict
        self._think = think
        self._owns_client = http_client is None
        self._http = http_client or httpx.AsyncClient(timeout=timeout_seconds)

    async def close(self) -> None:
        if self._owns_client:
            await self._http.aclose()

    async def installed_models(self) -> list[str]:
        try:
            response = await self._http.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise OllamaUnavailableError(str(exc)) from exc
        return [str(item["name"]) for item in response.json().get("models", [])]

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | str | None = None,
        think: bool | None = None,
    ) -> OllamaMessage:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": self._think if think is None else think,
            "options": {
                "temperature": self._temperature,
                "num_predict": self._num_predict,
            },
        }
        if tools:
            payload["tools"] = tools
        if response_format is not None:
            payload["format"] = response_format

        try:
            response = await self._http.post(
                f"{self.base_url}/api/chat",
                json=payload,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:500]
            raise OllamaUnavailableError(
                f"Ollama returned {exc.response.status_code}: {detail}"
            ) from exc
        except httpx.HTTPError as exc:
            raise OllamaUnavailableError(str(exc)) from exc

        raw_message = response.json().get("message")
        if not isinstance(raw_message, dict):
            raise OllamaUnavailableError("Ollama response has no message")
        return OllamaMessage.model_validate(raw_message)

    async def generate_structured(
        self,
        messages: list[dict[str, str]],
        output_schema: type[T],
    ) -> T:
        response = await self.chat(
            messages,
            response_format=output_schema.model_json_schema(),
            think=False,
        )
        return output_schema.model_validate_json(response.content)

