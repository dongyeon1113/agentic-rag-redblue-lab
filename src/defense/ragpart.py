"""RAGPart: retrieval-stage defense from arXiv:2512.24268.

A document is split into ``fragments`` contiguous pieces that are embedded
individually. The embeddings of every size-``combination_size`` subset are mean
pooled, giving ``C(N, k)`` representations per document. Each combination is
searched separately and the resulting top-p lists are aggregated by majority
vote.

Mean pooling is what dilutes a poisoned fragment: an adversarial passage built
as ``query || instruction`` only carries the query text in one or two
fragments, so every combination that excludes those fragments scores the
document low, and combinations that include them weight them by ``1 / k``.
"""

import math
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations


@dataclass(frozen=True)
class RagPartConfig:
    """Paper defaults are ``N=5``, ``k=3`` (Appendix Table 9/10).

    ``enabled`` gates building the side index. It is off by default because
    indexing costs ``fragments`` embedding calls per document, which is
    prohibitive on the full NQ corpus.
    """

    fragments: int = 5
    combination_size: int = 3
    enabled: bool = False

    def __post_init__(self) -> None:
        if self.fragments < 1:
            raise ValueError("fragments must be positive")
        if not 1 <= self.combination_size <= self.fragments:
            raise ValueError("combination_size must be within 1..fragments")


def partition_text(text: str, fragments: int) -> list[str]:
    """Split text into contiguous, near-equal fragments.

    Always returns exactly ``fragments`` pieces. A short document that has
    fewer than ``fragments`` words repeats words across fragments rather than
    returning fewer, non-empty pieces than that: ``_index_ragpart`` stores
    ``combination_vectors(fragment_vectors, combination_size)`` under
    ``combo_index`` 0..len(fragment_vectors)-1, and ``search_ragpart`` queries
    a *fixed* ``combo_index`` range computed from the configured
    ``fragments``/``combination_size``, the same range for every document.
    Returning fewer fragments for short text used to make
    ``combination_vectors`` derive its combination size from
    ``len(fragment_vectors)`` instead of the configured ``combination_size``,
    so a short document's ``combo_index`` values did not line up with what
    every other document's ``combo_index`` values mean, and any
    ``combo_index`` beyond its own (smaller) vector count silently excluded
    it from that round's majority vote -- an accidental under-count in favor
    of longer documents, not a defensive property.
    """
    words = text.split() or [text]
    if len(words) >= fragments:
        size, remainder = divmod(len(words), fragments)
        pieces: list[str] = []
        start = 0
        for index in range(fragments):
            stop = start + size + (1 if index < remainder else 0)
            pieces.append(" ".join(words[start:stop]))
            start = stop
        return pieces
    return [words[index % len(words)] for index in range(fragments)]


def _mean_pool(vectors: list[list[float]]) -> list[float]:
    count = len(vectors)
    pooled = [sum(values) / count for values in zip(*vectors, strict=True)]
    magnitude = math.sqrt(sum(value * value for value in pooled))
    return [value / magnitude for value in pooled] if magnitude else pooled


def combination_vectors(
    fragment_vectors: list[list[float]],
    combination_size: int,
) -> list[list[float]]:
    """Mean pool every size-k subset of the fragment embeddings."""
    size = max(1, min(combination_size, len(fragment_vectors)))
    return [
        _mean_pool(list(subset))
        for subset in combinations(fragment_vectors, size)
    ]


def combination_count(fragments: int, combination_size: int) -> int:
    return math.comb(fragments, max(1, min(combination_size, fragments)))


def majority_vote(
    ranked_sets: list[list[tuple[str, float]]],
    limit: int,
) -> list[tuple[str, float, int]]:
    """Aggregate per-combination top-p lists into a single ranking.

    Returns ``(document_id, mean score over appearances, vote count)`` for the
    ``limit`` most frequently retrieved documents. Ties are broken by the best
    rank the document reached, then by id so results stay deterministic.
    """
    votes: dict[str, int] = defaultdict(int)
    scores: dict[str, list[float]] = defaultdict(list)
    best_rank: dict[str, int] = {}

    for ranked in ranked_sets:
        for rank, (document_id, score) in enumerate(ranked):
            votes[document_id] += 1
            scores[document_id].append(score)
            if rank < best_rank.get(document_id, len(ranked)):
                best_rank[document_id] = rank

    ordered = sorted(
        votes,
        key=lambda document_id: (
            -votes[document_id],
            best_rank[document_id],
            document_id,
        ),
    )
    return [
        (
            document_id,
            round(
                sum(scores[document_id]) / len(scores[document_id]),
                6,
            ),
            votes[document_id],
        )
        for document_id in ordered[:limit]
    ]
