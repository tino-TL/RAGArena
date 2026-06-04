from __future__ import annotations

from app.schemas import SearchResponse, SearchResultResponse
from ragarena.observability.trace_summary import TraceSummary
from ragarena.retrieval.search import SearchResponse as RetrievalSearchResponse
from ragarena.retrieval.vector_store import SearchResult


def serialize_search_result(result: SearchResult) -> SearchResultResponse:
    return SearchResultResponse(
        score=result.score,
        source_scores=result.source_scores,
        chunk_id=result.chunk_id,
        document_id=result.document_id,
        model_name=result.model_name,
        content=result.content,
        metadata=result.metadata,
    )


def serialize_search_response(
    response: RetrievalSearchResponse,
    *,
    request_id: str,
    latency_ms: float,
    request_mode: str,
    fallback_strategy: str,
    trace_id: str | None,
    trace_url: str | None,
    route: str,
) -> SearchResponse:
    return SearchResponse(
        request_id=request_id,
        latency_ms=latency_ms,
        query=response.query,
        top_k=response.top_k,
        mode=request_mode,
        strategy=response.strategy or fallback_strategy,
        retrieval_latency_ms=response.latency_ms,
        candidate_count=response.candidate_count,
        rrf_k=response.rrf_k,
        used_hyde=response.used_hyde,
        rerank_attempted=response.rerank_attempted,
        rerank_succeeded=response.rerank_succeeded,
        used_rerank=response.rerank_succeeded,
        rerank_fallback_reason=response.rerank_fallback_reason,
        results=[serialize_search_result(result) for result in response.results],
        trace_summary=TraceSummary.from_search_response(
            response,
            trace_id=trace_id,
            trace_url=trace_url,
            route=route,
        ).to_dict(),
    )
