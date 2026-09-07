from scripts.build_nq_scenario_expansion import classify_question, select_balanced


def test_classify_question_covers_distinct_answer_shapes() -> None:
    assert classify_question("How many episodes are there?") == "quantity"
    assert classify_question("Does the character survive?") == "yes_no"
    assert classify_question("Why does the reaction occur?") == "reason_method"
    assert classify_question("Who wrote the book?") == "person"
    assert classify_question("Where is the station?") == "place"
    assert classify_question("When did the war end?") == "time"
    assert classify_question("What is a balance sheet?") == "choice_definition"
    assert classify_question("Converting stereo to mono is called") == "other"


def test_balanced_selection_reaches_every_available_type_first() -> None:
    records = [
        {"id": f"person-{index}", "question_type": "person"} for index in range(10)
    ] + [
        {"id": "yes-no-1", "question_type": "yes_no"},
        {"id": "time-1", "question_type": "time"},
    ]

    selected = select_balanced(records, count=3, seed=7)

    assert {item["question_type"] for item in selected} == {"person", "yes_no", "time"}
