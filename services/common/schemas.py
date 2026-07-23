from typing import Literal

from pydantic import BaseModel, Field


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
