import os
from pathlib import Path

from services.common.agent_factory import create_search_agent
from services.gmail_agent.google_gmail import sync_gmail_messages

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_FILE = PROJECT_ROOT / "datasets/sample/gmail_dummy.json"

GMAIL_SYNC_ENABLED = os.getenv("GMAIL_SYNC_ENABLED", "false").lower() in {
    "1",
    "true",
    "yes",
}

if GMAIL_SYNC_ENABLED:
    data_file = Path(
        os.getenv("DATA_FILE", "/app/data/google_gmail_messages.json")
    )
    sync_gmail_messages(
        output_file=data_file,
        token_file=Path(
            os.getenv(
                "GOOGLE_GMAIL_TOKEN_FILE",
                "/run/secrets/google-gmail-token.json",
            )
        ),
        query=os.getenv("GMAIL_QUERY", "in:anywhere"),
        max_messages=int(os.getenv("GMAIL_MAX_MESSAGES", "100")),
    )

app = create_search_agent(
    service_name="gmail-agent",
    default_data_file=DEFAULT_DATA_FILE,
    default_search_backend="chroma",
    data_file_override=DEFAULT_DATA_FILE if not GMAIL_SYNC_ENABLED else None,
)


@app.get("/source-status")
async def source_status() -> dict[str, str | bool]:
    return {
        "source": "gmail",
        "mode": "live" if GMAIL_SYNC_ENABLED else "sample",
        "connected": GMAIL_SYNC_ENABLED,
    }
