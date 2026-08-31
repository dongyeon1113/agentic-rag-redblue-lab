from __future__ import annotations

import os
from dataclasses import dataclass


ALLOWED_AGENT_PERMISSIONS = frozenset({
    "document:read",
    "document:write",
    "document:delete",
    "secret:read",
    "secret:write",
    "secret:delete",
    "gmail:read",
    "gmail:send",
    "gmail:delete",
    "drive:read",
    "drive:write",
    "drive:delete",
})
DEFAULT_AGENT_PERMISSIONS = frozenset({
    "document:read",
    "gmail:read",
    "drive:read",
})


def _integer(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _floating(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _boolean(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.casefold() in {"1", "true", "yes", "on"}


def _permissions(name: str) -> frozenset[str]:
    raw = os.getenv(name)
    if raw is None:
        return DEFAULT_AGENT_PERMISSIONS
    permissions = frozenset(
        item.strip() for item in raw.split(",") if item.strip()
    )
    unknown = permissions - ALLOWED_AGENT_PERMISSIONS
    if unknown:
        raise ValueError(
            f"Unknown {name} value(s): {', '.join(sorted(unknown))}"
        )
    return permissions


@dataclass(frozen=True)
class AgentSettings:
    agent_permissions: frozenset[str]
    ollama_base_url: str
    ollama_model: str
    temperature: float
    num_predict: int
    ollama_think: bool
    request_timeout_seconds: float
    max_tool_iterations: int
    local_db_agent_url: str
    gmail_agent_url: str
    drive_agent_url: str
    memory_data_dir: str
    ollama_embedding_base_url: str
    ollama_embedding_model: str
    embedding_timeout_seconds: float
    memory_chroma_collection: str
    chroma_index_batch_size: int
    auto_memory_enabled: bool
    auto_memory_min_confidence: float
    auto_memory_max_items: int

    @classmethod
    def from_env(cls) -> AgentSettings:
        return cls(
            agent_permissions=_permissions("AGENT_PERMISSIONS"),
            ollama_base_url=os.getenv(
                "OLLAMA_BASE_URL", "http://localhost:11434"
            ).rstrip("/"),
            ollama_model=os.getenv("OLLAMA_MODEL", "qwen3:8b"),
            temperature=_floating("OLLAMA_TEMPERATURE", 0.0),
            num_predict=_integer("OLLAMA_NUM_PREDICT", 1024),
            ollama_think=_boolean("OLLAMA_THINK", True),
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
            ollama_embedding_base_url=os.getenv(
                "OLLAMA_EMBEDDING_BASE_URL", "http://localhost:11434"
            ).rstrip("/"),
            ollama_embedding_model=os.getenv(
                "OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"
            ),
            embedding_timeout_seconds=_floating(
                "EMBEDDING_TIMEOUT_SECONDS", 180.0
            ),
            memory_chroma_collection=os.getenv(
                "MEMORY_CHROMA_COLLECTION", "long-term-memory-nomic-v1"
            ),
            chroma_index_batch_size=_integer("CHROMA_INDEX_BATCH_SIZE", 500),
            auto_memory_enabled=_boolean("AUTO_MEMORY_ENABLED", True),
            auto_memory_min_confidence=_floating(
                "AUTO_MEMORY_MIN_CONFIDENCE", 0.8
            ),
            auto_memory_max_items=_integer("AUTO_MEMORY_MAX_ITEMS", 3),
        )

