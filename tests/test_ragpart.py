import math

import pytest

from services.common.embeddings import DeterministicHashEmbeddings
from services.common.ragpart import (
    RagPartConfig,
    combination_count,
    combination_vectors,
    majority_vote,
    partition_text,
)


def test_partition_splits_into_contiguous_balanced_fragments() -> None:
    text = " ".join(f"w{index}" for index in range(11))

    pieces = partition_text(text, 5)

    assert len(pieces) == 5
    assert [len(piece.split()) for piece in pieces] == [3, 2, 2, 2, 2]
    assert " ".join(pieces) == text


def test_partition_always_returns_exactly_fragments_pieces_for_short_text() -> None:
    # Always returning `fragments` pieces (repeating words when the text is
    # too short) keeps combo_index numbering consistent with every other
    # document -- see partition_text's docstring for why a shorter return
    # value used to silently under-count short documents in majority_vote.
    assert partition_text("two words", 5) == ["two", "words", "two", "words", "two"]
    assert partition_text("single", 5) == ["single"] * 5
    assert partition_text("", 5) == [""] * 5
    assert len(partition_text("two words", 5)) == 5


def test_short_documents_produce_the_same_combo_count_as_long_ones() -> None:
    embedding = DeterministicHashEmbeddings(16)
    short_vectors = embedding.embed_documents(partition_text("two words", 5))
    long_text = " ".join(f"w{index}" for index in range(40))
    long_vectors = embedding.embed_documents(partition_text(long_text, 5))

    short_combos = combination_vectors(short_vectors, 3)
    long_combos = combination_vectors(long_vectors, 3)

    assert len(short_combos) == len(long_combos) == combination_count(5, 3)


def test_combination_vectors_are_mean_pooled_and_normalised() -> None:
    fragments = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]

    vectors = combination_vectors(fragments, 2)

    assert len(vectors) == combination_count(3, 2) == 3
    for vector in vectors:
        assert math.isclose(math.sqrt(sum(v * v for v in vector)), 1.0)
    # Mean of [1,0] and [0,1], normalised.
    assert vectors[0] == pytest.approx([0.7071067, 0.7071067], abs=1e-6)


def test_combination_size_is_clamped_to_available_fragments() -> None:
    assert len(combination_vectors([[1.0, 0.0]], 3)) == 1


def test_majority_vote_ranks_by_votes_then_best_rank() -> None:
    ranked_sets = [
        [("a", 0.9), ("b", 0.5)],
        [("b", 0.6), ("a", 0.4)],
        [("c", 0.8), ("b", 0.7)],
    ]

    aggregated = majority_vote(ranked_sets, limit=2)

    assert [document_id for document_id, _, _ in aggregated] == ["b", "a"]
    assert aggregated[0][2] == 3
    assert aggregated[0][1] == pytest.approx((0.5 + 0.6 + 0.7) / 3)


def test_config_rejects_combination_size_above_fragments() -> None:
    RagPartConfig(fragments=5, combination_size=3)
    with pytest.raises(ValueError):
        RagPartConfig(fragments=3, combination_size=5)


def test_query_as_poison_covers_fewer_combinations_than_a_golden_document():
    """The paper's mechanism, measured on this lab's embedding and attack.

    A `query || instruction` poison carries the query text in only one or two
    fragments, so it scores zero in every combination that excludes them. A
    genuinely relevant document spreads its query terms across the text and
    survives more combinations, which is what majority vote ranks on.

    Note this is *coverage*, not score suppression: with the lab's hash
    embedding the poison's mean combination score does not drop, because
    per-fragment L2 normalisation amplifies short fragments that do contain
    query terms. See docs/ragpart-ragmask.ko.md.
    """
    embedding = DeterministicHashEmbeddings()
    query = "What is the capital of France?"
    query_vector = embedding.embed_query(query)
    golden = "Paris is the capital and most populous city of France."
    poison = f"{query} " + " ".join(f"filler{index}" for index in range(30))

    def covered_combinations(text: str) -> int:
        fragment_vectors = embedding.embed_documents(partition_text(text, 5))
        return sum(
            sum(a * b for a, b in zip(query_vector, vector, strict=True))
            > 1e-9
            for vector in combination_vectors(fragment_vectors, 3)
        )

    assert covered_combinations(golden) == 9
    assert covered_combinations(poison) == 6
