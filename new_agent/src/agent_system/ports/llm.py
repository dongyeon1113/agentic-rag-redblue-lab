from __future__ import annotations

from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LanguageModel(Protocol):
    async def generate_text(self, messages: list[dict[str, str]]) -> str: ...

    async def generate_structured(
        self,
        messages: list[dict[str, str]],
        output_schema: type[T],
    ) -> T: ...


class LanguageModelNotConfigured:
    """Safe placeholder until an Ollama/OpenAI adapter is selected."""

    async def generate_text(self, messages: list[dict[str, str]]) -> str:
        raise RuntimeError("No language model adapter is configured")

    async def generate_structured(
        self,
        messages: list[dict[str, str]],
        output_schema: type[T],
    ) -> T:
        del messages, output_schema
        raise RuntimeError("No language model adapter is configured")

