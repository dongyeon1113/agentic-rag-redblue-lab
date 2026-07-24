import json
from pathlib import Path
from typing import Any

GOOGLE_DOC_MIME_TYPE = "application/vnd.google-apps.document"
TEXT_MIME_TYPE = "text/plain"
READ_ONLY_SCOPE = "https://www.googleapis.com/auth/drive.readonly"


def _download_text(service: Any, file: dict[str, Any]) -> str | None:
    mime_type = file["mimeType"]
    if mime_type == GOOGLE_DOC_MIME_TYPE:
        payload = service.files().export_media(
            fileId=file["id"],
            mimeType=TEXT_MIME_TYPE,
        ).execute()
    elif mime_type == TEXT_MIME_TYPE:
        payload = service.files().get_media(fileId=file["id"]).execute()
    else:
        return None

    if isinstance(payload, bytes):
        return payload.decode("utf-8", errors="replace").strip()
    return str(payload).strip()


def sync_google_drive_folder(
    *,
    folder_id: str,
    output_file: Path,
    credentials_file: Path | None = None,
    service: Any | None = None,
) -> list[dict[str, Any]]:
    if not folder_id.strip():
        raise ValueError("GOOGLE_DRIVE_FOLDER_ID is required")

    if service is None:
        if credentials_file is None or not credentials_file.is_file():
            raise FileNotFoundError(
                "Google service-account key not found: "
                f"{credentials_file or '<unset>'}"
            )

        from google.oauth2 import service_account
        from googleapiclient.discovery import build

        credentials = service_account.Credentials.from_service_account_file(
            str(credentials_file),
            scopes=[READ_ONLY_SCOPE],
        )
        service = build(
            "drive",
            "v3",
            credentials=credentials,
            cache_discovery=False,
        )

    files: list[dict[str, Any]] = []
    page_token: str | None = None
    while True:
        response = (
            service.files()
            .list(
                q=f"'{folder_id}' in parents and trashed = false",
                fields="nextPageToken, files(id,name,mimeType,modifiedTime)",
                orderBy="name",
                pageToken=page_token,
            )
            .execute()
        )
        files.extend(response.get("files", []))
        page_token = response.get("nextPageToken")
        if not page_token:
            break

    documents: list[dict[str, Any]] = []
    for file in files:
        text = _download_text(service, file)
        if not text:
            continue
        documents.append(
            {
                "id": f"google-drive-{file['id']}",
                "source": "google-drive",
                "trust": "trusted",
                "tags": ["google-drive", file["mimeType"]],
                "text": f"Title: {file['name']}\n\n{text}",
            }
        )

    if not documents:
        raise ValueError(
            "No supported Google Drive documents found; "
            "share the folder with the service-account email"
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = output_file.with_suffix(f"{output_file.suffix}.tmp")
    temporary_file.write_text(
        json.dumps(documents, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_file.replace(output_file)
    return documents
