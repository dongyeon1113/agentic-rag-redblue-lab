import pytest

from services.common.embeddings import DeterministicHashEmbeddings
from services.orchestrator.agent_poison import optimize_trigger, rank_memory, score_trigger


def test_trigger_optimizer_is_deterministic_and_reports_paper_losses() -> None:
    embedding = DeterministicHashEmbeddings(64)
    kwargs = dict(
        queries=["alpha question", "beta question"],
        benign_texts=["ordinary alpha reference", "ordinary beta reference"],
        seed_trigger="please answer",
        candidate_tokens=["signal", "quietly", "parallel"],
        iterations=4,
    )
    first, history = optimize_trigger(embedding, **kwargs)
    second, _ = optimize_trigger(embedding, **kwargs)

    assert first == second
    assert len(history) == 4
    assert first.objective == pytest.approx(first.uniqueness - 0.1 * first.compactness)


def test_triggered_query_retrieves_matching_poison_memory() -> None:
    embedding = DeterministicHashEmbeddings(128)
    memories = [
        ("ordinary unrelated record", "normal", False),
        ("question unique signal", "target action", True),
    ]

    ranked = rank_memory(
        embedding,
        query="question unique signal",
        memories=memories,
        top_k=1,
    )

    assert ranked[0][2] is True
    assert ranked[0][1] == "target action"


def test_empty_inputs_are_rejected() -> None:
    embedding = DeterministicHashEmbeddings()
    try:
        score_trigger(
            embedding,
            queries=[],
            benign_texts=["record"],
            trigger="signal",
        )
    except ValueError as exc:
        assert "must not be empty" in str(exc)
    else:
        raise AssertionError("empty queries should fail")
