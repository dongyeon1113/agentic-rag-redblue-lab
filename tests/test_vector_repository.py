from __future__ import annotations

import asyncio
import json
from pathlib import Path

from agent_system.infrastructure.embeddings import DeterministicEmbeddingClient
from agent_system.services.local_db.vector_repository import VectorDocumentRepository


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_vector_repository_indexes_and_searches_documents(tmp_path: Path) -> None:
    knowledge = tmp_path / "seed/nq.json"
    secrets = tmp_path / "seed/secrets.json"
    _write(knowledge, [
        {
            "id": "doc-chicago",
            "source": "nq",
            "trust": "trusted",
            "tags": [],
            "text": "Chicago Fire season four premiered in October 2015.",
        },
        {
            "id": "doc-triangle",
            "source": "nq",
            "trust": "trusted",
            "tags": [],
            "text": "An equilateral triangle has three equal sides.",
        },
    ])
    _write(secrets, {"secrets": {"test_api_key": "fake-secret-value"}})
    repository = VectorDocumentRepository(
        data_file=tmp_path / "data/documents.json",
        knowledge_seed_file=knowledge,
        secret_seed_file=secrets,
        persist_directory=tmp_path / "data/chroma",
        embedding=DeterministicEmbeddingClient(),
        collection_name="test-vector-documents",
        batch_size=2,
    )

    knowledge_hits = asyncio.run(
        repository.search("Chicago Fire season", "knowledge", 1)
    )
    secret_hits = asyncio.run(
        repository.search("test_api_key", "secret", 1)
    )

    assert knowledge_hits[0].document_id == "doc-chicago"
    assert secret_hits[0].document_id == "secret-test_api_key"

