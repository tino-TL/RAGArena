from __future__ import annotations

from typing import TypedDict


class AgentTraceStep(TypedDict, total=False):
    node: str
    query: str | None
    route: str | None
    route_confidence: float
    route_reason: str | None
    retrieval_mode: str | None
    chunk_ids: list[int]
    scores: list[float]
    grade: bool | None
    grade_score: float
    grade_reason: str | None
    latency_ms: float
    message: str | None
    rerank_attempted: bool
    rerank_succeeded: bool
    rerank_fallback_reason: str | None


class AgentState(TypedDict, total=False):
    original_query: str
    current_query: str
    route: str
    route_reason: str | None
    route_confidence: float
    guardrail_passed: bool
    guardrail_reason: str | None
    retrieval_strategy: str | None
    documents: list[str]
    retrieval_results: list[dict[str, object]]
    generation: str
    grade: bool
    grade_score: float
    grade_reason: str | None
    useful_chunk_ids: list[int]
    suggested_rewrite: str | None
    citations: list[dict[str, object]]
    rewrite_count: int
    rewrite_reason: str | None
    max_rewrite: int
    trace: list[str]
    trace_steps: list[AgentTraceStep]
    trace_summary: dict[str, object]
    trace_id: str | None
    trace_url: str | None
    retrieval_candidate_count: int
    used_hyde: bool
    rerank_attempted: bool
    rerank_succeeded: bool
    used_rerank: bool
    rerank_fallback_reason: str | None


def initial_agent_state(query: str, *, max_rewrite: int) -> AgentState:
    return {
        "original_query": query,
        "current_query": query,
        "route": "local_rag",
        "route_reason": None,
        "route_confidence": 0.0,
        "guardrail_passed": True,
        "guardrail_reason": None,
        "retrieval_strategy": None,
        "documents": [],
        "retrieval_results": [],
        "generation": "",
        "grade": False,
        "grade_score": 0.0,
        "grade_reason": None,
        "useful_chunk_ids": [],
        "suggested_rewrite": None,
        "citations": [],
        "rewrite_count": 0,
        "rewrite_reason": None,
        "max_rewrite": max_rewrite,
        "trace": [],
        "trace_steps": [],
        "trace_summary": {},
        "trace_id": None,
        "trace_url": None,
        "retrieval_candidate_count": 0,
        "used_hyde": False,
        "rerank_attempted": False,
        "rerank_succeeded": False,
        "used_rerank": False,
        "rerank_fallback_reason": None,
    }
