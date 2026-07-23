from pathlib import Path

from services.common.chroma_store import ChromaDocumentStore

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_chroma_store_returns_paris_document_first(tmp_path: Path) -> None:
    store = ChromaDocumentStore(
        PROJECT_ROOT / "datasets/sample/nq_sample.json",
        collection_name="test-local-db",
        persist_directory=tmp_path / "chroma",
    )

    hits = store.search("What is the capital of France?", limit=3)

    assert hits
    assert hits[0].document_id == "nq-sample-001"
    assert hits[0].source == "beir-nq-sample"
    assert hits[0].trust == "trusted"
    assert hits[0].tags == []
    assert "Paris" in hits[0].text
    assert hits[0].score > hits[1].score


def test_chroma_store_persists_without_duplicate_documents(tmp_path: Path) -> None:
    persist_directory = tmp_path / "chroma"
    data_file = PROJECT_ROOT / "datasets/sample/nq_sample.json"

    first_store = ChromaDocumentStore(
        data_file,
        collection_name="test-persistence",
        persist_directory=persist_directory,
    )
    second_store = ChromaDocumentStore(
        data_file,
        collection_name="test-persistence",
        persist_directory=persist_directory,
    )

    assert first_store.count() == 3
    assert second_store.count() == 3
