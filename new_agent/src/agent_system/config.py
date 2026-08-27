from __future__ import annotations

import os
from dataclasses import dataclass


def _integer(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _floating(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


@dataclass(frozen=True)
class AgentSettings:
    ollama_base_url: str
    ollama_model: str
    temperature: float
    num_predict: int
    request_timeout_seconds: float
    max_tool_iterations: int
    local_db_agent_url: str
    gmail_agent_url: str
    drive_agent_url: str
    memory_data_dir: str

    @classmethod
    def from_env(cls) -> AgentSettings:
        return cls(
            ollama_base_url=os.getenv(
                "OLLAMA_BASE_URL", "http://localhost:11434"
            ).rstrip("/"),
            ollama_model=os.getenv("OLLAMA_MODEL", "qwen3:8b"),
            temperature=_floating("OLLAMA_TEMPERATURE", 0.0),
            num_predict=_integer("OLLAMA_NUM_PREDICT", 1024),
            request_timeout_seconds=_floating("REQUEST_TIMEOUT_SECONDS", 120.0),
            max_tool_iterations=_integer("MAX_TOOL_ITERATIONS", 8),
            local_db_agent_url=os.getenv(
                "LOCAL_DB_AGENT_URL", "http://localhost:8001"
            ),
            gmail_agent_url=os.getenv(
                "GMAIL_AGENT_URL", "http://localhost:8002"
            ),
            drive_agent_url=os.getenv(
                "DRIVE_AGENT_URL", "http://localhost:8003"
            ),
            memory_data_dir=os.getenv(
                "MEMORY_DATA_DIR", "data/orchestrator"
            ),
        )

