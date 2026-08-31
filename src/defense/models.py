from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class DefenseConfig(BaseModel):
    regex_filter: bool = False
    prompt_guard: bool = False
    spotlighting: list[Literal["delimiting", "datamarking", "encoding"]] = Field(
        default_factory=list
    )
    ragpart: bool = False
    block_indirect_actions: bool = True


class DefenseFinding(BaseModel):
    defense: str
    record_id: str
    action: Literal["blocked", "transformed", "reranked", "action_blocked"]
    reason: str
    metadata: dict = Field(default_factory=dict)


class DefenseReport(BaseModel):
    enabled: list[str] = Field(default_factory=list)
    inspected_records: int = 0
    blocked_records: int = 0
    transformed_records: int = 0
    indirect_actions_blocked: int = 0
    detector_latency_ms: float = 0.0
    untrusted_data_seen: bool = False
    findings: list[DefenseFinding] = Field(default_factory=list)
