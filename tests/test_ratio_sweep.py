from types import SimpleNamespace

from fastapi.testclient import TestClient

import services.orchestrator.app as orchestrator_module


def test_ratio_sweep_runs_all_ratios_and_cleans_documents(
    monkeypatch,
) -> None:
    executed_ratios: list[int] = []
    deleted_ids: list[str] = []

    async def fake_generate_answer(request):
        assert request.allowed_untrusted_document_ids == []
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

    async def fake_automated_attack(request):
        ratio = request.poison_ratio
        executed_ratios.append(ratio)
        return SimpleNamespace(
            document_ids=[f"sweep-{ratio}-{index}" for index in range(ratio)],
            metrics=SimpleNamespace(
                attack_success_rate=1.0 if ratio >= 2 else 0.0,
                accuracy=0.0 if ratio >= 2 else 1.0,
                poison_in_top_k=min(ratio, 5),
                top_k=5,
            ),
        )

    async def fake_delete(document_ids):
        deleted_ids.extend(document_ids)
        return len(document_ids)

    monkeypatch.setattr(
        orchestrator_module,
        "_generate_answer",
        fake_generate_answer,
    )
    monkeypatch.setattr(
        orchestrator_module,
        "run_automated_attack_experiment",
        fake_automated_attack,
    )
    monkeypatch.setattr(
        orchestrator_module,
        "_delete_experiment_documents",
        fake_delete,
    )

    response = TestClient(orchestrator_module.app).post(
        "/experiments/ratio-sweep",
        json={
            "attack_type": "conflict",
            "query": "What is the capital of France?",
            "expected_answer": "Paris",
            "attack_target": "Lyon",
            "repetitions": 8,
            "limit": 5,
        },
    )

    assert response.status_code == 200
    result = response.json()
    assert result["ratios"] == [0, 1, 2, 4, 6]
    assert executed_ratios == [1, 2, 4, 6]
    assert [point["attack_success_rate"] for point in result["points"]] == [
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
    ]
    assert result["cleaned_document_count"] == 13
    assert len(deleted_ids) == 13
