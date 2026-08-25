#!/usr/bin/env python3

import argparse
import json
from typing import Any
from urllib.request import Request, urlopen


def request_json(
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: int = 120,
) -> dict[str, Any]:
    body = json.dumps(payload).encode() if payload is not None else None
    request = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"} if body else {},
        method="POST" if body else "GET",
    )
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the deployed RAG stack.")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Orchestrator base URL",
    )
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")

    health = request_json(f"{base_url}/health")
    assert health["status"] == "ok", health

    agents = request_json(f"{base_url}/agents")
    agent_states = agents["agents"]
    assert agent_states, agents
    assert all(item.get("status") == "ok" for item in agent_states.values()), agents

    model = request_json(f"{base_url}/model")
    assert model["status"] == "ok" and model["installed"], model

    answer = request_json(
        f"{base_url}/answer",
        payload={
            "query": "What is the capital of France?",
            "sources": ["local_db"],
            "limit": 3,
            "mode": "defended",
            "retrieval_policy": "always",
        },
    )
    assert "Paris" in answer["answer"], answer
    assert answer["documents"], answer
    assert all(item["trust"] == "trusted" for item in answer["documents"]), answer

    print("Smoke test passed.")
    print(f"Model: {model['model']}")
    print(f"Answer: {answer['answer']}")


if __name__ == "__main__":
    main()
