from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, cast

from ragarena.retrieval.search import SearchResponse
from ragarena.retrieval.vector_store import SearchResult

PREVIEW_CHARS = 200


@dataclass
class TraceSummary:
    trace_id: str | None = None
    trace_url: str | None = None
    route: str | None = None
    route_confidence: float | None = None
    route_reason: str | None = None
    guardrail_reason: str | None = None
    strategy: str | None = None
    used_hyde: bool = False
    rerank_attempted: bool = False
    rerank_succeeded: bool = False
    used_rerank: bool = False
    rerank_fallback_reason: str | None = None
    rewrite_count: int = 0
    rewrite_reason: str | None = None
    retrieved_chunks: list[dict[str, Any]] = field(default_factory=list)
    grader_decision: str | None = None
    grader_score: float | None = None
    grader_reason: str | None = None
    useful_chunk_ids: list[int] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "TraceSummary":
        return cls(
            trace_id=_optional_str(state.get("trace_id")),
            trace_url=_optional_str(state.get("trace_url")),
            route=_optional_str(state.get("route")),
            route_confidence=_optional_float(state.get("route_confidence")),
            route_reason=_optional_str(state.get("route_reason")),
            guardrail_reason=_optional_str(state.get("guardrail_reason")),
            strategy=_optional_str(state.get("retrieval_strategy")),
            used_hyde=bool(state.get("used_hyde", False)),
            rerank_attempted=bool(state.get("rerank_attempted", False)),
            rerank_succeeded=bool(state.get("rerank_succeeded", False)),
            used_rerank=bool(state.get("rerank_succeeded", state.get("used_rerank", False))),
            rerank_fallback_reason=_optional_str(state.get("rerank_fallback_reason")),
            rewrite_count=int(state.get("rewrite_count", 0) or 0),
            rewrite_reason=_optional_str(state.get("rewrite_reason")),
            retrieved_chunks=_compact_state_results(state.get("retrieval_results", [])),
            grader_decision=_grade_to_decision(state.get("grade")),
            grader_score=_optional_float(state.get("grade_score")),
            grader_reason=_optional_str(state.get("grade_reason")),
            useful_chunk_ids=_int_list(state.get("useful_chunk_ids")),
            citations=list(state.get("citations", [])),
        )

    @classmethod
    def from_search_response(
        cls,
        response: SearchResponse,
        *,
        trace_id: str | None = None,
        trace_url: str | None = None,
        route: str | None = None,
    ) -> "TraceSummary":
        return cls(
            trace_id=trace_id,
            trace_url=trace_url,
            route=route,
            strategy=response.strategy or response.mode,
            used_hyde=response.used_hyde,
            rerank_attempted=response.rerank_attempted,
            rerank_succeeded=response.rerank_succeeded,
            used_rerank=response.rerank_succeeded,
            rerank_fallback_reason=response.rerank_fallback_reason,
            retrieved_chunks=[compact_search_result(result) for result in response.results],
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compact_search_result(result: SearchResult) -> dict[str, Any]:
    section_name = result.section_name or _optional_str(result.metadata.get("section_name"))
    return {
        "chunk_id": result.chunk_id,
        "document_id": result.document_id,
        "section_name": section_name,
        "score": result.score,
        "source_scores": result.source_scores,
        "content_preview": compact_preview(result.content),
    }


def compact_preview(content: str, limit: int = PREVIEW_CHARS) -> str:
    preview = " ".join(content.split())
    if len(preview) <= limit:
        return preview
    return f"{preview[: limit - 3]}..."


def _compact_state_results(values: object) -> list[dict[str, Any]]:
    if not isinstance(values, list):
        return []
    chunks = []
    for value in values:
        if not isinstance(value, dict):
            continue
        content = str(value.get("content", ""))
        metadata = value.get("metadata", {})
        metadata_dict = metadata if isinstance(metadata, dict) else {}
        source_scores = value.get("source_scores", {})
        chunks.append(
            {
                "chunk_id": _optional_int(value.get("chunk_id")),
                "document_id": _optional_int(value.get("document_id")),
                "section_name": _optional_str(value.get("section_name"))
                or _optional_str(metadata_dict.get("section_name")),
                "score": _optional_float(value.get("score")),
                "source_scores": source_scores if isinstance(source_scores, dict) else {},
                "content_preview": compact_preview(content),
            }
        )
    return chunks


def _grade_to_decision(value: object) -> str | None:
    if value is None:
        return None
    return "relevant" if bool(value) else "not_relevant"


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _optional_int(value: object) -> int | None:
    try:
        return int(cast(Any, value))
    except (TypeError, ValueError):
        return None


def _optional_float(value: object) -> float | None:
    try:
        return float(cast(Any, value))
    except (TypeError, ValueError):
        return None


def _int_list(value: object) -> list[int]:
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        try:
            result.append(int(cast(Any, item)))
        except (TypeError, ValueError):
            continue
    return result
