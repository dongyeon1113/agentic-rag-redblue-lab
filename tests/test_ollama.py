from __future__ import annotations

import asyncio
import json

import httpx
from pydantic import BaseModel

from agent_system.infrastructure.ollama import OllamaChatModel


class StructuredResult(BaseModel):
    value: str


def test_agent_chat_enables_thinking_but_structured_extraction_disables_it() -> None:
    payloads: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        payloads.append(payload)
        content = '{"value":"ok"}' if "format" in payload else "done"
        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": content,
                    "thinking": "reasoning" if payload["think"] else None,
                }
            },
        )

    async def scenario() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http:
            model = OllamaChatModel(
                model="qwen3:8b",
                base_url="http://ollama.test",
                think=True,
                http_client=http,
            )
            message = await model.chat([
                {"role": "user", "content": "Drive에서 문서를 찾아줘"}
            ])
            structured = await model.generate_structured(
                [{"role": "user", "content": "extract"}],
                StructuredResult,
            )
            assert message.thinking == "reasoning"
            assert structured.value == "ok"

    asyncio.run(scenario())

    assert payloads[0]["think"] is True
    assert payloads[1]["think"] is False
    assert payloads[0]["stream"] is False
