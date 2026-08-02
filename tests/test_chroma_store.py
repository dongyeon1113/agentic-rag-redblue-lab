import json
from pathlib import Path

from services.common.chroma_store import ChromaDocumentStore

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sample_documents() -> list[dict[str, object]]:
    data_file = PROJECT_ROOT / "datasets/sample/nq_sample.json"
    return json.loads(data_file.read_text(encoding="utf-8"))


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

    expected_count = len(_sample_documents())
    assert first_store.count() == expected_count
    assert second_store.count() == expected_count


def test_chroma_store_sync_removes_documents_missing_from_json(
    tmp_path: Path,
) -> None:
    persist_directory = tmp_path / "chroma"
    data_file = PROJECT_ROOT / "datasets/sample/nq_sample.json"
    store = ChromaDocumentStore(
        data_file,
        collection_name="test-sync",
        persist_directory=persist_directory,
    )
    store.add_document(
        document_id="stale-document",
        source="old-json",
        trust="untrusted",
        tags=[],
        text="This document no longer exists in the JSON file.",
    )

    synchronized_store = ChromaDocumentStore(
        data_file,
        collection_name="test-sync",
        persist_directory=persist_directory,
        sync_data_file=True,
    )

    assert not synchronized_store.contains("stale-document")
    assert synchronized_store.count() == len(_sample_documents())
    assert synchronized_store.contains("nq-sample-001")


def test_chroma_store_deletes_only_untrusted_documents(tmp_path: Path) -> None:
    store = ChromaDocumentStore(
        PROJECT_ROOT / "datasets/sample/nq_sample.json",
        collection_name="test-reset",
        persist_directory=tmp_path / "chroma",
    )
    store.add_document(
        document_id="poison-test-1",
        source="red-team-test",
        trust="untrusted",
        tags=["poison"],
        text="Controlled poison document.",
    )

    deleted_count = store.delete_untrusted_documents()

    documents = _sample_documents()
    initial_untrusted_count = sum(
        document["trust"] == "untrusted" for document in documents
    )
    trusted_count = len(documents) - initial_untrusted_count

    assert deleted_count == initial_untrusted_count + 1
    assert store.count() == trusted_count
    assert not store.contains("poison-test-1")
    assert store.contains("nq-sample-001")
