from pathlib import Path

from services.common.agent_factory import create_search_agent

PROJECT_ROOT = Path(__file__).resolve().parents[2]

app = create_search_agent(
    service_name="gmail-agent",
    default_data_file=PROJECT_ROOT / "datasets/sample/gmail_dummy.json",
)
