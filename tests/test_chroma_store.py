import json
from pathlib import Path

from services.common.chroma_store import ChromaDocumentStore
from services.common.embeddings import DeterministicHashEmbeddings

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class CountingEmbeddings(DeterministicHashEmbeddings):
    def __init__(self) -> None:
        super().__init__()
        self.document_count = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_count += len(texts)
        return super().embed_documents(texts)


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
    assert store.document_counts() == {"trusted": 3, "untrusted": 0, "total": 3}


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


def test_chroma_store_does_not_reembed_existing_documents(tmp_path: Path) -> None:
    embedding = CountingEmbeddings()
    options = {
        "data_file": PROJECT_ROOT / "datasets/sample/nq_sample.json",
        "collection_name": "test-no-reembedding",
        "persist_directory": tmp_path / "chroma",
        "embedding": embedding,
    }

    ChromaDocumentStore(**options)
    first_count = embedding.document_count
    ChromaDocumentStore(**options)

    assert first_count == 3
    assert embedding.document_count == first_count


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

    assert deleted_count == 1
    assert store.count() == 3
    assert not store.contains("poison-test-1")
    assert store.contains("nq-sample-001")


def test_chroma_store_syncs_trusted_corpus_but_preserves_untrusted(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("CHROMA_SYNC_TRUSTED_CORPUS", "true")
    persist_directory = tmp_path / "chroma"
    store = ChromaDocumentStore(
        PROJECT_ROOT / "datasets/sample/nq_sample.json",
        collection_name="test-sync",
        persist_directory=persist_directory,
    )
    store.add_document(
        document_id="poison-preserved",
        source="red-team-test",
        trust="untrusted",
        tags=["poison"],
        text="Controlled poison document.",
    )
    replacement = tmp_path / "replacement.json"
    replacement.write_text(json.dumps([{
        "id": "replacement-trusted",
        "source": "replacement",
        "trust": "trusted",
        "text": "Replacement trusted corpus document.",
    }]), encoding="utf-8")

    synced = ChromaDocumentStore(
        replacement,
        collection_name="test-sync",
        persist_directory=persist_directory,
    )

    assert synced.count() == 2
    assert synced.contains("replacement-trusted")
    assert synced.contains("poison-preserved")
    assert not synced.contains("nq-sample-001")
