import json
from pathlib import Path

from services.drive_agent.google_drive import sync_google_drive_folder


class FakeRequest:
    def __init__(self, payload):
        self.payload = payload

    def execute(self):
        return self.payload


class FakeFiles:
    def list(self, **kwargs):
        assert kwargs["q"] == "'folder-123' in parents and trashed = false"
        return FakeRequest(
            {
                "files": [
                    {
                        "id": "doc-1",
                        "name": "Security policy",
                        "mimeType": "text/plain",
                        "modifiedTime": "2026-07-24T00:00:00Z",
                    },
                    {
                        "id": "doc-2",
                        "name": "Unsupported PDF",
                        "mimeType": "application/pdf",
                        "modifiedTime": "2026-07-24T00:00:00Z",
                    },
                ]
            }
        )

    def get_media(self, *, fileId):
        assert fileId == "doc-1"
        return FakeRequest(b"trusted Drive content")


class FakeDriveService:
    def files(self):
        return FakeFiles()


def test_sync_google_drive_folder_writes_search_documents(
    tmp_path: Path,
) -> None:
    output_file = tmp_path / "drive.json"

    documents = sync_google_drive_folder(
        folder_id="folder-123",
        output_file=output_file,
        service=FakeDriveService(),
    )

    assert len(documents) == 1
    assert documents[0]["id"] == "google-drive-doc-1"
    assert documents[0]["trust"] == "trusted"
    assert "Security policy" in documents[0]["text"]
    assert json.loads(output_file.read_text(encoding="utf-8")) == documents
