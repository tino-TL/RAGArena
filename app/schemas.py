from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


SearchMode = Literal["bm25", "dense", "vector", "hybrid"]


class ResponseMetadata(BaseModel):
    request_id: str
    latency_ms: float


class ErrorBody(BaseModel):
    code: str
    message: str
    retryable: bool


class ErrorResponse(BaseModel):
    error: ErrorBody
    request_id: str


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    mode: SearchMode = "hybrid"
    use_hyde: bool = False
    use_rerank: bool = False


class SearchResultResponse(BaseModel):
    score: float
    source_scores: dict[str, float]
    chunk_id: int
    document_id: int
    model_name: str
    content: str
    metadata: dict[str, object] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    request_id: str
    latency_ms: float
    query: str
    top_k: int
    mode: str
    strategy: str = ""
    retrieval_latency_ms: float = 0.0
    candidate_count: int = 0
    rrf_k: int | None = None
    used_hyde: bool = False
    rerank_attempted: bool = False
    rerank_succeeded: bool = False
    used_rerank: bool = False
    rerank_fallback_reason: str | None = None
    results: list[SearchResultResponse]
    trace_summary: dict[str, object] = Field(default_factory=dict)


class AskRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=3, ge=1, le=20)
    use_hyde: bool | None = None
    use_rerank: bool | None = None


class AskResponse(BaseModel):
    request_id: str
    latency_ms: float
    query: str
    answer: str
    retrieved_chunks: list[SearchResultResponse]
    cache_hit: bool = False
    trace_summary: dict[str, object] = Field(default_factory=dict)


class StreamRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=3, ge=1, le=20)
    use_hyde: bool | None = None
    use_rerank: bool | None = None


class AgentRequest(BaseModel):
    query: str = Field(min_length=1)
    max_rewrite: int = Field(default=3, ge=0, le=10)


class AgentTraceStep(BaseModel):
    node: str
    query: str | None = None
    route: str | None = None
    route_confidence: float | None = None
    route_reason: str | None = None
    retrieval_mode: str | None = None
    chunk_ids: list[int] = Field(default_factory=list)
    scores: list[float] = Field(default_factory=list)
    grade: bool | None = None
    grade_score: float | None = None
    grade_reason: str | None = None
    latency_ms: float = 0.0
    message: str | None = None
    rerank_attempted: bool = False
    rerank_succeeded: bool = False
    rerank_fallback_reason: str | None = None


class AgentResponse(BaseModel):
    request_id: str
    latency_ms: float
    original_query: str
    current_query: str
    route: str
    route_reason: str | None = None
    route_confidence: float = 0.0
    guardrail_passed: bool = True
    guardrail_reason: str | None = None
    trace: list[str]
    generation: str
    grade: bool
    grade_score: float = 0.0
    grade_reason: str | None = None
    rewrite_reason: str | None = None
    rewrite_count: int
    max_rewrite: int
    documents_count: int
    trace_steps: list[AgentTraceStep] = Field(default_factory=list)
    trace_summary: dict[str, object] = Field(default_factory=dict)


class FeedbackRequest(BaseModel):
    request_id: str = Field(min_length=1)
    score: int = Field(ge=-1, le=1)
    comment: str | None = None
    endpoint: str | None = None


class FeedbackResponse(BaseModel):
    request_id: str
    latency_ms: float
    stored: bool
    message: str


class PaperResponse(BaseModel):
    id: int
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    categories: list[str]
    published_at: str | None
    updated_at: str | None
    pdf_url: str
    source_url: str


class PapersResponse(BaseModel):
    request_id: str
    latency_ms: float
    papers: list[PaperResponse]


class PaperFetchRequest(BaseModel):
    query: str = Field(default="all:retrieval augmented generation", min_length=1)
    max_results: int = Field(default=20, ge=1, le=100)
    sync_documents: bool = True


class PaperFetchResponse(BaseModel):
    request_id: str
    latency_ms: float
    fetched: int
    stored: int
    synced_documents: bool
