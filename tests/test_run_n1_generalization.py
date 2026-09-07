import json

from scripts.run_n1_generalization import balanced_sample, load_scenarios


def test_balanced_sample_includes_each_question_type() -> None:
    records = [
        {"id": "person-1", "question_type": "person"},
        {"id": "person-2", "question_type": "person"},
        {"id": "time-1", "question_type": "time"},
        {"id": "place-1", "question_type": "place"},
    ]

    selected = balanced_sample(records, count=3, seed=9)

    assert {item["question_type"] for item in selected} == {"person", "time", "place"}


def test_load_scenarios_excludes_invalid_baselines(tmp_path) -> None:
    path = tmp_path / "scenarios.jsonl"
    values = [
        {"id": "valid", "status": "accepted", "baseline_valid": True},
        {"id": "invalid", "status": "accepted", "baseline_valid": False},
        {"id": "rejected", "status": "rejected", "baseline_valid": True},
    ]
    path.write_text("".join(json.dumps(item) + "\n" for item in values), encoding="utf-8")

    assert [item["id"] for item in load_scenarios(path)] == ["valid"]
