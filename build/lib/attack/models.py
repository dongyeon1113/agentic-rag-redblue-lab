from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from defense import DefenseConfig


AttackType = Literal["knowledge_poison", "tool_injection", "agent_poison"]
ToolTarget = Literal["send_email", "read_secret", "delete_document", "secret_exfiltration"]


class AttackDocument(BaseModel):
    document_id: str
    title: str
    content: str
    metadata: dict[str, str] = Field(default_factory=dict)


class AttackExperimentRequest(BaseModel):
    query: str = Field(min_length=1)
    expected_answer: str = ""
    attack_target: str = Field(min_length=1)
    attack_type: AttackType = "knowledge_poison"
    tool_target: ToolTarget = "send_email"
    poison_count: int = Field(default=3, ge=1, le=20)
    top_k: int = Field(default=5, ge=1, le=20)
    include_query_prefix: bool = True
    defense: DefenseConfig = Field(default_factory=DefenseConfig)
    compare: bool = True
    user_id: str = "experiment-user"
    session_id: str = "experiment-session"
    memory_context: str = "context1"
    cleanup: bool = True


class AttackRunResult(BaseModel):
    mode: Literal["vulnerable", "defended"]
    outcome: Literal["attack_succeeded", "attack_resisted", "inconclusive"]
    answer: str
    workflow_id: str
    status: str
    attack_target_present: bool
    expected_answer_present: bool
    approval_requested: bool
    retrieved_documents: list[dict] = Field(default_factory=list)
    attack_documents_retrieved: list[str] = Field(default_factory=list)
    trusted_document_count: int = 0
    defense_report: dict = Field(default_factory=dict)
    tool_results: list[dict] = Field(default_factory=list)
    approval_requests: list[dict] = Field(default_factory=list)


class AttackExperimentResult(BaseModel):
    attack_type: AttackType
    tool_target: ToolTarget
    top_k: int
    injected_document_ids: list[str]
    documents: list[AttackDocument]
    runs: list[AttackRunResult]
    defense_blocked_attack: bool
    cleaned_up: bool
