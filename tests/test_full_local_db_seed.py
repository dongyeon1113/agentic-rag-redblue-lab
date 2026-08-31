from __future__ import annotations

import asyncio
from pathlib import Path

from agent_system.services.local_db.json_repository import JsonDocumentRepository


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_full_nq_and_secret_seed_conversion(tmp_path: Path) -> None:
    repository = JsonDocumentRepository(
        data_file=tmp_path / "local_db/documents.json",
        knowledge_seed_file=PROJECT_ROOT / "mock_data/local_db/nq_10000.json",
        secret_seed_file=PROJECT_ROOT / "mock_data/local_db/secrets.json",
    )

    knowledge = asyncio.run(
        repository.get("beir-nq-doc6")
    )
    secret = asyncio.run(
        repository.get("secret-test_api_key")
    )

    assert knowledge is not None
    assert knowledge.namespace == "knowledge"
    assert secret is not None
    assert secret.namespace == "secret"

