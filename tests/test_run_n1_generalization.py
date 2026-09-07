from scripts.run_n1_generalization import balanced_sample


def test_balanced_sample_includes_each_question_type() -> None:
    records = [
        {"id": "person-1", "question_type": "person"},
        {"id": "person-2", "question_type": "person"},
        {"id": "time-1", "question_type": "time"},
        {"id": "place-1", "question_type": "place"},
    ]

    selected = balanced_sample(records, count=3, seed=9)

    assert {item["question_type"] for item in selected} == {"person", "time", "place"}
