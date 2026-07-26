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
    poison_text: str
    comparison: ExperimentComparisonResponse
