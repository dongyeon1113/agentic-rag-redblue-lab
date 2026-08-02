import base64
import json
from pathlib import Path

from services.gmail_agent.google_gmail import sync_gmail_messages


class FakeRequest:
    def __init__(self, payload):
        self.payload = payload

    def execute(self):
        return self.payload


class FakeMessages:
    def list(self, **kwargs):
        assert kwargs["userId"] == "me"
        assert kwargs["q"] == "in:inbox"
        return FakeRequest({"messages": [{"id": "message-1"}]})

    def get(self, **kwargs):
        assert kwargs == {
            "userId": "me",
            "id": "message-1",
            "format": "full",
        }
        body = base64.urlsafe_b64encode(b"Dummy security mail").decode()
        return FakeRequest(
            {
                "id": "message-1",
                "labelIds": ["INBOX"],
                "payload": {
                    "mimeType": "text/plain",
                    "headers": [
                        {"name": "Subject", "value": "Security notice"},
                        {"name": "From", "value": "sender@example.com"},
                        {"name": "To", "value": "kangdummy3@gmail.com"},
                    ],
                    "body": {"data": body},
                },
            }
        )


class FakeUsers:
    def messages(self):
        return FakeMessages()


class FakeGmailService:
    def users(self):
        return FakeUsers()


def test_sync_gmail_messages_writes_search_documents(
    tmp_path: Path,
) -> None:
    output_file = tmp_path / "gmail.json"

    documents = sync_gmail_messages(
        output_file=output_file,
        query="in:inbox",
        max_messages=10,
        service=FakeGmailService(),
    )

    assert len(documents) == 1
    assert documents[0]["id"] == "google-gmail-message-1"
    assert documents[0]["trust"] == "trusted"
    assert "Security notice" in documents[0]["text"]
    assert "Dummy security mail" in documents[0]["text"]
    assert json.loads(output_file.read_text(encoding="utf-8")) == documents
