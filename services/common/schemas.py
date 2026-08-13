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
    allowed_untrusted_document_ids: list[str] | None = Field(
        default=None,
        max_length=20,
    )


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


class ExperimentDeleteResponse(BaseModel):
    status: Literal["deleted"] = "deleted"
    document_id: str
    deleted: bool
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
    poison_count: int = Field(default=5, ge=0, le=10)
    top_k: int = Field(default=5, ge=1, le=20)
    max_generation_trials: int = Field(default=10, ge=1, le=50)
    passage_word_count: int = Field(default=30, ge=10, le=120)
    generation_temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    cleanup_before_run: bool = True
    candidate_multiplier: int = Field(default=2, ge=1, le=5)
    fixed_candidates: list["PoisonedRAGCandidate"] | None = Field(default=None, max_length=10)

    @model_validator(mode="after")
    def answers_must_differ(self) -> "PoisonedRAGRequest":
        expected = self.expected_answer.casefold().strip()
        target = self.attack_target.casefold().strip()
        if expected == target:
            raise ValueError("expected_answer and attack_target must differ")
        return self


class PoisonedRAGCandidate(BaseModel):
    document_id: str
    instruction: str
    poison_text: str
    verification_answer: str
    verified: bool
    generation_queries: int
    generation_seconds: float
    word_count: int
    retrieval_score: float | None = None
    selection_score: float | None = None
    selected: bool = False
    rejection_reason: str | None = None


class AttackDashboardMetrics(BaseModel):
    attack_success_rate: float
    accuracy: float
    poison_in_top_k: int
    top_k: int
    retrieval_precision: float
    retrieval_recall: float
    retrieval_f1: float
    generation_queries: int
    generation_seconds: float
    total_seconds: float = 0.0


class PoisonedRAGPipelineStatus(BaseModel):
    baseline: Literal["completed"] = "completed"
    generation: Literal["completed", "partial", "skipped"]
    injection: Literal["completed", "partial", "skipped"]
    retrieval: Literal["completed"] = "completed"
    answer_evaluation: Literal["completed"] = "completed"


class PoisonedRAGRunMetadata(BaseModel):
    started_at: str
    completed_at: str
    service_version: str
    model: str
    generation_temperature: float
    max_generation_trials: int
    passage_word_count: int
    cleanup_before_run: bool
    candidate_multiplier: int = 1
    selection_policy: str = "verified_then_retrieval_score_with_deduplication"


class PoisonedRAGResponse(BaseModel):
    status: Literal["completed"] = "completed"
    strategy: Literal["poisonedrag"] = "poisonedrag"
    run_id: str
    scenario_name: str | None = None
    construction: Literal["black_box_q_plus_i"] = "black_box_q_plus_i"
    requested_poison_count: int
    generated_candidate_count: int = 0
    verified_poison_count: int
    injected_poison_count: int
    top_k: int
    document_ids: list[str]
    generated_documents: list[PoisonedRAGCandidate]
    pipeline: PoisonedRAGPipelineStatus
    metadata: PoisonedRAGRunMetadata
    metrics: AttackDashboardMetrics
    baseline: ExperimentEvaluationResponse
    attacked: ExperimentEvaluationResponse


class PoisonedRAGScenario(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    query: str = Field(min_length=1, max_length=500)
    expected_answer: str = Field(min_length=1, max_length=500)
    attack_target: str = Field(min_length=1, max_length=500)

    @model_validator(mode="after")
    def answers_must_differ(self) -> "PoisonedRAGScenario":
        if self.expected_answer.casefold().strip() == self.attack_target.casefold().strip():
            raise ValueError("expected_answer and attack_target must differ")
        return self


class PoisonedRAGBenchmarkRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenarios: list[PoisonedRAGScenario] = Field(min_length=1, max_length=20)
    poison_counts: list[int] = Field(default_factory=lambda: [0, 1, 3, 5], min_length=1, max_length=6)
    repetitions: int = Field(default=1, ge=1, le=5)
    top_k: int = Field(default=5, ge=1, le=20)
    max_generation_trials: int = Field(default=10, ge=1, le=50)
    passage_word_count: int = Field(default=30, ge=10, le=120)
    generation_temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    candidate_multiplier: int = Field(default=2, ge=1, le=5)
    fixed_poison_pool: bool = True

    @model_validator(mode="after")
    def validate_counts(self) -> "PoisonedRAGBenchmarkRequest":
        if any(count < 0 or count > 10 for count in self.poison_counts):
            raise ValueError("poison_counts values must be between 0 and 10")
        self.poison_counts = sorted(set(self.poison_counts))
        return self


class PoisonedRAGBenchmarkPoint(BaseModel):
    poison_count: int
    trials: int
    successful_trials: int = 0
    failed_trials: int = 0
    attack_success_rate: float
    accuracy: float
    retrieval_precision: float
    retrieval_recall: float
    retrieval_f1: float
    average_poison_in_top_k: float
    average_generation_queries: float
    average_generation_seconds: float
    average_total_seconds: float = 0.0


class PoisonedRAGRunFailure(BaseModel):
    scenario_name: str
    poison_count: int
    repetition: int
    stage: Literal["setup", "baseline", "generation", "injection", "retrieval_or_answer", "unknown"]
    error_type: str
    detail: str
    elapsed_seconds: float


class PoisonedRAGBenchmarkResponse(BaseModel):
    status: Literal["completed"] = "completed"
    experiment_id: str
    model: str
    points: list[PoisonedRAGBenchmarkPoint]
    runs: list[PoisonedRAGResponse]
    failures: list[PoisonedRAGRunFailure] = Field(default_factory=list)
    json_url: str
    csv_url: str
    fixed_poison_pool: bool = False


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


class RatioSweepRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attack_type: Literal[
        "poisonedrag",
        "data_poisoning",
        "conflict",
        "keyword_stuffing",
        "prompt_injection",
    ]
    query: str = Field(min_length=1, max_length=500)
    expected_answer: str = Field(min_length=1, max_length=500)
    attack_target: str = Field(min_length=1, max_length=500)
    repetitions: int = Field(default=8, ge=1, le=50)
    limit: int = Field(default=5, ge=1, le=20)

    @model_validator(mode="after")
    def answers_must_differ(self) -> "RatioSweepRequest":
        expected = self.expected_answer.casefold().strip()
        target = self.attack_target.casefold().strip()
        if expected == target:
            raise ValueError("expected_answer and attack_target must differ")
        return self


class RatioSweepPoint(BaseModel):
    poison_ratio: Literal[0, 1, 2, 4, 6]
    attack_success_rate: float
    accuracy: float
    poison_in_top_k: int
    top_k: int


class RatioSweepResponse(BaseModel):
    status: Literal["completed"] = "completed"
    strategy: Literal[
        "poisonedrag",
        "data_poisoning",
        "conflict",
        "keyword_stuffing",
        "prompt_injection",
    ]
    ratios: list[Literal[0, 1, 2, 4, 6]]
    points: list[RatioSweepPoint]
    cleaned_document_count: int
