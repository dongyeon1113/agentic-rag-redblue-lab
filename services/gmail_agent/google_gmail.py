import base64
import json
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

READ_ONLY_SCOPE = "https://www.googleapis.com/auth/gmail.readonly"


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        if data.strip():
            self.parts.append(data.strip())


def _decode_body(data: str) -> str:
    padding = "=" * (-len(data) % 4)
    payload = base64.urlsafe_b64decode(data + padding)
    return payload.decode("utf-8", errors="replace").strip()


def _extract_message_text(payload: dict[str, Any]) -> str:
    mime_type = payload.get("mimeType", "")
    body_data = payload.get("body", {}).get("data")
    if body_data and mime_type == "text/plain":
        return _decode_body(body_data)

    plain_parts: list[str] = []
    html_parts: list[str] = []
    for part in payload.get("parts", []):
        text = _extract_message_text(part)
        if not text:
            continue
        if part.get("mimeType") == "text/html":
            html_parts.append(text)
        else:
            plain_parts.append(text)

    if plain_parts:
        return "\n\n".join(plain_parts)
    if html_parts:
        return "\n\n".join(html_parts)
    if body_data and mime_type == "text/html":
        parser = _HTMLTextExtractor()
        parser.feed(_decode_body(body_data))
        return " ".join(parser.parts)
    return ""


def _headers(message: dict[str, Any]) -> dict[str, str]:
    return {
        header.get("name", "").lower(): header.get("value", "")
        for header in message.get("payload", {}).get("headers", [])
    }


def _gmail_service(token_file: Path) -> Any:
    if not token_file.is_file():
        raise FileNotFoundError(
            f"Google Gmail OAuth token not found: {token_file}"
        )

    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build

    credentials = Credentials.from_authorized_user_file(
        str(token_file),
        [READ_ONLY_SCOPE],
    )
    if credentials.expired and credentials.refresh_token:
        credentials.refresh(Request())
    if not credentials.valid:
        raise ValueError("Google Gmail OAuth token is invalid")
    return build("gmail", "v1", credentials=credentials, cache_discovery=False)


def sync_gmail_messages(
    *,
    output_file: Path,
    token_file: Path | None = None,
    query: str = "in:anywhere",
    max_messages: int = 100,
    service: Any | None = None,
) -> list[dict[str, Any]]:
    if max_messages < 1:
        raise ValueError("GMAIL_MAX_MESSAGES must be at least 1")
    if service is None:
        if token_file is None:
            raise ValueError("GOOGLE_GMAIL_TOKEN_FILE is required")
        service = _gmail_service(token_file)

    message_refs: list[dict[str, Any]] = []
    page_token: str | None = None
    while len(message_refs) < max_messages:
        response = (
            service.users()
            .messages()
            .list(
                userId="me",
                q=query,
                maxResults=min(100, max_messages - len(message_refs)),
                pageToken=page_token,
            )
            .execute()
        )
        message_refs.extend(response.get("messages", []))
        page_token = response.get("nextPageToken")
        if not page_token:
            break

    documents: list[dict[str, Any]] = []
    for message_ref in message_refs[:max_messages]:
        message = (
            service.users()
            .messages()
            .get(
                userId="me",
                id=message_ref["id"],
                format="full",
            )
            .execute()
        )
        headers = _headers(message)
        content = _extract_message_text(message.get("payload", {}))
        if not content:
            content = message.get("snippet", "").strip()
        if not content:
            continue

        labels = message.get("labelIds", [])
        documents.append(
            {
                "id": f"google-gmail-{message['id']}",
                "source": "google-gmail",
                "trust": "trusted",
                "tags": ["google-gmail", *[f"label:{label}" for label in labels]],
                "text": (
                    f"Subject: {headers.get('subject', '(no subject)')}\n"
                    f"From: {headers.get('from', '')}\n"
                    f"To: {headers.get('to', '')}\n"
                    f"Date: {headers.get('date', '')}\n\n"
                    f"{content}"
                ),
            }
        )

    if not documents:
        raise ValueError("No readable Gmail messages found")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = output_file.with_suffix(f"{output_file.suffix}.tmp")
    temporary_file.write_text(
        json.dumps(documents, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_file.replace(output_file)
    return documents
