from pathlib import Path

from services.common.search import JsonDocumentStore, lexical_score, tokenize


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_tokenize_supports_english_and_korean() -> None:
    assert tokenize("France 수도 2026") == ["france", "수도", "2026"]


def test_lexical_score_ignores_unrelated_document() -> None:
    assert lexical_score("capital of France", "Mount Everest is a mountain") == 0


def test_local_store_returns_paris_document_first() -> None:
    store = JsonDocumentStore(PROJECT_ROOT / "datasets/sample/nq_sample.json")

    hits = store.search("capital of France", limit=3)

    assert hits
    assert hits[0].document_id == "nq-sample-001"
    assert "Paris" in hits[0].text
