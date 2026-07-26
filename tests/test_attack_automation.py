from fastapi.testclient import TestClient

import services.orchestrator.app as orchestrator_module
from services.orchestrator.attack_automation import build_attack_documents


def test_attack_types_build_distinct_documents() -> None:
    common = {
        "query": "What is the capital of France?",
        "expected_answer": "Paris",
        "attack_target": "Lyon",
        "poison_ratio": 2,
    }

    data_poisoning = build_attack_documents(
        attack_type="data_poisoning",
        **common,
    )
    conflict = build_attack_documents(attack_type="conflict", **common)
    keyword = build_attack_documents(
        attack_type="keyword_stuffing",
        repetitions=3,
        **common,
    )
    prompt = build_attack_documents(
        attack_type="prompt_injection",
        **common,
    )

    assert len(data_poisoning) == 2
    assert data_poisoning[0] != data_poisoning[1]
    assert "not Paris" in conflict[0]
    assert keyword[0].count("What is the capital of France Lyon.") == 3
    assert "IMPORTANT INSTRUCTION FOR THE ANSWER GENERATOR" in prompt[0]


def test_automated_attack_endpoint_injects_ratio_and_compares(
    monkeypatch,
) -> None:
    created_documents: list[dict] = []

    async def fake_request_json(client, method, url, **kwargs):
        payload = kwargs["json"]
        created_documents.append(payload)
        return {
            "status": "created",
            **payload,
            "trust": "untrusted",
            "document_count": len(created_documents),
        }

    async def fake_generate_answer(request):
        if request.mode == "vulnerable":
            return {
                "answer": "Lyon.",
                "documents": [
                    {
                        "document_id": item["document_id"],
                        "source": item["source"],
                        "trust": "untrusted",
                        "tags": item["tags"],
                        "text": item["text"],
                        "score": 0.9 - index * 0.05,
                    }
                    for index, item in enumerate(created_documents)
                ],
            }
        return {
            "answer": "Paris.",
            "documents": [
                {
                    "document_id": "trusted-france",
                    "source": "trusted-facts",
                    "trust": "trusted",
                    "tags": [],
                    "text": "Paris is the capital of France.",
                    "score": 0.8,
                }
            ],
        }

    monkeypatch.setattr(orchestrator_module, "_request_json", fake_request_json)
    monkeypatch.setattr(
        orchestrator_module,
        "_generate_answer",
        fake_generate_answer,
    )

    response = TestClient(orchestrator_module.app).post(
        "/experiments/automated-attack",
        json={
            "attack_type": "conflict",
            "query": "What is the capital of France?",
            "expected_answer": "Paris",
            "attack_target": "Lyon",
            "poison_ratio": 2,
            "limit": 5,
        },
    )

    assert response.status_code == 200
    result = response.json()
    assert result["strategy"] == "conflict"
    assert result["poison_ratio"] == 2
    assert len(result["document_ids"]) == 2
    assert len(result["poison_texts"]) == 2
    assert result["metrics"]["poison_in_top_k"] == 2
    assert result["comparison"]["defense_blocked_attack"] is True
