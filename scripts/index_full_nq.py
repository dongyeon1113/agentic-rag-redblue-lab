from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from agent_system.infrastructure.embeddings import OllamaEmbeddingClient
from agent_system.services.local_db.vector_repository import VectorDocumentRepository


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and verify the full NQ Chroma index."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/full-nq-check"),
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    repository = VectorDocumentRepository(
        data_file=args.data_dir / "documents.json",
        knowledge_seed_file=root / "mock_data/local_db/nq_10000.json",
        secret_seed_file=root / "mock_data/local_db/secrets.json",
        persist_directory=args.data_dir / "chroma",
        embedding=OllamaEmbeddingClient(
            model=os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            timeout_seconds=300,
        ),
        collection_name="full-nq-nomic-check",
        batch_size=int(os.getenv("CHROMA_INDEX_BATCH_SIZE", "500")),
    )
    hits = await repository.search(
        "Chicago Fire season four premiere date",
        "knowledge",
        3,
    )
    print(f"indexed_documents={repository._index.count()}")
    for hit in hits:
        print(f"{hit.document_id}: {hit.title}")
    if not any(hit.document_id == "beir-nq-doc6" for hit in hits):
        raise SystemExit("Full-index verification failed")


if __name__ == "__main__":
    asyncio.run(main())

