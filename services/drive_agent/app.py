import os
from pathlib import Path

from services.common.agent_factory import create_search_agent
from services.drive_agent.google_drive import sync_google_drive_folder

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_FILE = PROJECT_ROOT / "datasets/sample/drive_dummy.json"

DRIVE_SYNC_ENABLED = os.getenv("DRIVE_SYNC_ENABLED", "false").lower() in {
    "1",
    "true",
    "yes",
}

if DRIVE_SYNC_ENABLED:
    data_file = Path(
        os.getenv("DATA_FILE", "/app/data/google_drive_documents.json")
    )
    sync_google_drive_folder(
        folder_id=os.getenv("GOOGLE_DRIVE_FOLDER_ID", ""),
        output_file=data_file,
        credentials_file=Path(
            os.getenv(
                "GOOGLE_APPLICATION_CREDENTIALS",
                "/run/secrets/google-service-account.json",
            )
        ),
    )

app = create_search_agent(
    service_name="drive-agent",
    default_data_file=DEFAULT_DATA_FILE,
    default_search_backend="chroma",
    data_file_override=DEFAULT_DATA_FILE if not DRIVE_SYNC_ENABLED else None,
)


@app.get("/source-status")
async def source_status() -> dict[str, str | bool]:
    return {
        "source": "drive",
        "mode": "live" if DRIVE_SYNC_ENABLED else "sample",
        "connected": DRIVE_SYNC_ENABLED,
    }
