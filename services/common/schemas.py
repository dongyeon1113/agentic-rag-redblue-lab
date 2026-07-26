from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    service: str
    version: str


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    limit: int = Field(default=3, ge=1, le=20)


class SearchHit(BaseModel):
    document_id: str
    source: str
    trust: str
    tags: list[str] = Field(default_factory=list)
    text: str
    score: float


class SearchResponse(BaseModel):
    service: str
    query: str
    hits: list[SearchHit]


class OrchestratorQueryRequest(SearchRequest):
    sources: list[Literal["local_db", "gmail", "drive"]] = Field(
        default_factory=lambda: ["local_db", "gmail", "drive"]
    )


class OrchestratorAnswerRequest(OrchestratorQueryRequest):
    mode: Literal["vulnerable", "defended"] = "vulnerable"


class ExperimentDocumentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(
        min_length=1,
        max_length=120,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    )
    source: str = Field(min_length=1, max_length=200)
    tags: list[str] = Field(default_factory=list, max_length=20)
    text: str = Field(min_length=1, max_length=10_000)


class ExperimentDocumentResponse(BaseModel):
    status: Literal["created"] = "created"
    document_id: str
    source: str
    trust: Literal["untrusted"] = "untrusted"
    tags: list[str]
    document_count: int


class ExperimentResetResponse(BaseModel):
    status: Literal["reset"] = "reset"
    deleted_count: int
    document_count: int


class ExperimentEvaluationRequest(OrchestratorAnswerRequest):
    expected_answer: str = Field(min_length=1, max_length=500)
    attack_target: str = Field(min_length=1, max_length=500)
    attack_document_ids: list[str] = Field(min_length=1, max_length=20)

    @model_validator(mode="after")
    def answers_must_differ(self) -> "ExperimentEvaluationRequest":
        expected = self.expected_answer.casefold().strip()
        target = self.attack_target.casefold().strip()
        if expected == target:
            raise ValueError("expected_answer and attack_target must differ")
        return self


class ExperimentEvaluationResponse(BaseModel):
    service: str
    query: str
    model: str
    mode: Literal["vulnerable", "defended"]
    answer: str
    outcome: Literal["attack_succeeded", "attack_resisted", "inconclusive"]
    expected_answer: str
    attack_target: str
    expected_answer_present: bool
    attack_target_present: bool
    attack_document_retrieved: bool
    attack_document_rank: int | None
    attack_document_score: float | None
    untrusted_document_count: int
    documents: list[SearchHit]


class ExperimentComparisonRequest(OrchestratorQueryRequest):
    expected_answer: str = Field(min_length=1, max_length=500)
    attack_target: str = Field(min_length=1, max_length=500)
    attack_document_ids: list[str] = Field(min_length=1, max_length=20)

    @model_validator(mode="after")
    def answers_must_differ(self) -> "ExperimentComparisonRequest":
        expected = self.expected_answer.casefold().strip()
        target = self.attack_target.casefold().strip()
        if expected == target:
            raise ValueError("expected_answer and attack_target must differ")
        return self


class ExperimentComparisonResponse(BaseModel):
    service: str
    query: str
    model: str
    vulnerable: ExperimentEvaluationResponse
    defended: ExperimentEvaluationResponse
    attack_succeeded_in_vulnerable: bool
    attack_succeeded_in_defended: bool
    defense_blocked_attack: bool


class KeywordStuffingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=500)
    expected_answer: str = Field(min_length=1, max_length=500)
    attack_target: str = Field(min_length=1, max_length=500)
    repetitions: int = Field(default=8, ge=1, le=50)
    include_prompt_injection: bool = True
    limit: int = Field(default=3, ge=1, le=20)

    @model_validator(mode="after")
    def answers_must_differ(self) -> "KeywordStuffingRequest":
        expected = self.expected_answer.casefold().strip()
        target = self.attack_target.casefold().strip()
        if expected == target:
            raise ValueError("expected_answer and attack_target must differ")
        return self


class KeywordStuffingResponse(BaseModel):
    status: Literal["completed"] = "completed"
    document_id: str
    repetitions: int
    include_prompt_injection: bool
    poison_text: str
    comparison: ExperimentComparisonResponse


class AutomatedAttackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attack_type: Literal[
        "data_poisoning",
        "conflict",
        "keyword_stuffing",
        "prompt_injection",
    ]
    query: str = Field(min_length=1, max_length=500)
    expected_answer: str = Field(min_length=1, max_length=500)
    attack_target: str = Field(min_length=1, max_length=500)
    poison_ratio: Literal[1, 2, 4, 6] = 1
    repetitions: int = Field(default=8, ge=1, le=50)
    limit: int = Field(default=5, ge=1, le=20)

    @model_validator(mode="after")
    def answers_must_differ(self) -> "AutomatedAttackRequest":
        expected = self.expected_answer.casefold().strip()
        target = self.attack_target.casefold().strip()
        if expected == target:
            raise ValueError("expected_answer and attack_target must differ")
        return self


class PoisonedRAGRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=500)
    expected_answer: str = Field(min_length=1, max_length=500)
    attack_target: str = Field(min_length=1, max_length=500)
    poison_ratio: Literal[1, 2, 4, 6] = 2
    limit: int = Field(default=5, ge=1, le=20)

    @model_validator(mode="after")
    def answers_must_differ(self) -> "PoisonedRAGRequest":
        expected = self.expected_answer.casefold().strip()
        target = self.attack_target.casefold().strip()
        if expected == target:
            raise ValueError("expected_answer and attack_target must differ")
        return self


class PoisonedRAGCandidate(BaseModel):
    document_id: str
    text: str
    relevance_score: float


class AttackDashboardMetrics(BaseModel):
    attack_success_rate: float
    accuracy: float
    poison_in_top_k: int
    top_k: int
    poison_retrieval_rate: float


class PoisonedRAGResponse(BaseModel):
    status: Literal["completed"] = "completed"
    strategy: Literal["poisonedrag"] = "poisonedrag"
    poison_ratio: Literal[1, 2, 4, 6]
    generation_mode: Literal["llm", "hybrid", "fallback"]
    generated_candidate_count: int
    document_ids: list[str]
    selected_candidates: list[PoisonedRAGCandidate]
    metrics: AttackDashboardMetrics
    comparison: ExperimentComparisonResponse


class AutomatedAttackResponse(BaseModel):
    status: Literal["completed"] = "completed"
    strategy: Literal[
        "data_poisoning",
        "conflict",
        "keyword_stuffing",
        "prompt_injection",
    ]
    poison_ratio: Literal[1, 2, 4, 6]
    generated_candidate_count: int
    document_ids: list[str]
    poison_texts: list[str]
    selected_candidates: list[PoisonedRAGCandidate]
    metrics: AttackDashboardMetrics
    comparison: ExperimentComparisonResponse
