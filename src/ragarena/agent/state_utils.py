from __future__ import annotations

import logging
from time import perf_counter
from typing import Any, cast

from ragarena.agent.state import AgentState
from ragarena.retrieval.search import SearchResponse
from ragarena.retrieval.vector_store import SearchResult

logger = logging.getLogger(__name__)


def sync_retrieval_state(state: AgentState, response: SearchResponse) -> None:
    serialized = serialize_search_results(response.results)
    state["retrieval_results"] = serialized
    state["documents"] = [result.content for result in response.results]
    state["retrieval_strategy"] = response.strategy
    state["retrieval_candidate_count"] = response.candidate_count
    state["used_hyde"] = response.used_hyde
    state["rerank_attempted"] = response.rerank_attempted
    state["rerank_succeeded"] = response.rerank_succeeded
    state["used_rerank"] = response.used_rerank
    state["rerank_fallback_reason"] = response.rerank_fallback_reason


def sync_reranked_results(state: AgentState, results: list[SearchResult]) -> None:
    serialized = serialize_search_results(results)
    state["retrieval_results"] = serialized
    state["documents"] = [result.content for result in results]


def sync_citations_state(state: AgentState, citations: list[dict[str, object]]) -> None:
    state["citations"] = citations


def serialize_search_results(results: list[SearchResult]) -> list[dict[str, object]]:
    return [
        {
            "chunk_id": result.chunk_id,
            "document_id": result.document_id,
            "content": result.content,
            "score": result.score,
            "model_name": result.model_name,
            "source_scores": result.source_scores,
            "section_name": result.section_name,
            "metadata": result.metadata,
        }
        for result in results
    ]


def deserialize_search_results(values: list[dict[str, object]]) -> list[SearchResult]:
    results: list[SearchResult] = []
    for value in values:
        source_scores = cast(dict[str, Any], value.get("source_scores", {}))
        metadata = cast(dict[str, Any], value.get("metadata", {}))
        results.append(
            SearchResult(
                chunk_id=int(cast(Any, value["chunk_id"])),
                document_id=int(cast(Any, value["document_id"])),
                content=str(value["content"]),
                score=float(cast(Any, value["score"])),
                model_name=str(value["model_name"]),
                source_scores={str(key): float(score) for key, score in source_scores.items()},
                section_name=_optional_str(value.get("section_name")),
                metadata=dict(metadata),
            )
        )
    return results


def record_trace_step(
    state: AgentState,
    node: str,
    started_at: float,
    **fields: object,
) -> None:
    step = {
        "node": node,
        "latency_ms": round((perf_counter() - started_at) * 1000, 3),
        **fields,
    }
    state.setdefault("trace_steps", []).append(cast(Any, step))
    logger.info("agent_trace_step", extra={"agent_step": step})


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None
