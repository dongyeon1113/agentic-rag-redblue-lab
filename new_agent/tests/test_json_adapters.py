from __future__ import annotations

import asyncio
import json
from pathlib import Path

from agent_system.services.drive.json_gateway import JsonDriveGateway
from agent_system.services.gmail.json_gateway import JsonGmailGateway
from agent_system.services.local_db.json_repository import JsonDocumentRepository


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_local_db_json_repository_imports_both_seed_schemas(tmp_path: Path) -> None:
    knowledge = tmp_path / "seed" / "nq.json"
    secrets = tmp_path / "seed" / "secrets.json"
    data = tmp_path / "data" / "documents.json"
    _write(knowledge, [{
        "id": "nq-1",
        "source": "nq",
        "trust": "trusted",
        "tags": ["test"],
        "text": "Example title\n\nExample content",
    }])
    _write(secrets, {"secrets": {"test_api_key": "fake-value"}})

    repository = JsonDocumentRepository(
        data_file=data,
        knowledge_seed_file=knowledge,
        secret_seed_file=secrets,
    )
    knowledge_hits = asyncio.run(repository.search("Example", "knowledge", 10))
    secret_hits = asyncio.run(repository.search("test_api_key", "secret", 10))

    assert knowledge_hits[0].document_id == "nq-1"
    assert secret_hits[0].document_id == "secret-test_api_key"
    assert data.exists()


def test_gmail_json_gateway_persists_sent_messages(tmp_path: Path) -> None:
    inbox_seed = PROJECT_ROOT / "mock_data/gmail/inbox.json"
    sent_seed = PROJECT_ROOT / "mock_data/gmail/sent.json"
    inbox = tmp_path / "gmail/inbox.json"
    sent = tmp_path / "gmail/sent.json"
    gateway = JsonGmailGateway(
        inbox_file=inbox,
        sent_file=sent,
        inbox_seed_file=inbox_seed,
        sent_seed_file=sent_seed,
    )

    message = asyncio.run(gateway.send(
        "agent.user@example.com",
        ["recipient@example.com"],
        "JSON persistence test",
        "Saved to the sent mailbox",
    ))
    reloaded = JsonGmailGateway(
        inbox_file=inbox,
        sent_file=sent,
        inbox_seed_file=inbox_seed,
        sent_seed_file=sent_seed,
    )

    assert asyncio.run(reloaded.get(message.message_id)) is not None


def test_drive_json_gateway_initializes_from_seed(tmp_path: Path) -> None:
    gateway = JsonDriveGateway(
        data_file=tmp_path / "drive/items.json",
        seed_file=PROJECT_ROOT / "mock_data/drive/items.json",
    )

    matches = asyncio.run(gateway.search("architecture", None, 10))
    assert matches[0].item_id == "file-architecture"


def test_copied_local_db_files_are_present() -> None:
    nq_file = PROJECT_ROOT / "mock_data/local_db/nq_10000.json"
    secrets_file = PROJECT_ROOT / "mock_data/local_db/secrets.json"

    assert len(json.loads(nq_file.read_text(encoding="utf-8"))) == 10_000
    assert "test_api_key" in json.loads(
        secrets_file.read_text(encoding="utf-8")
    )["secrets"]

