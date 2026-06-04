from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.v1.serializers import serialize_search_response
from app.dependencies import RequestContext, get_request_context, latency_ms, raise_api_error
from app.schemas import SearchRequest, SearchResponse
from ragarena.config import settings
from ragarena.observability import get_langfuse_tracer
from ragarena.retrieval.search import search_with_strategy
from ragarena.retrieval.service import normalize_strategy

router = APIRouter(tags=["search"])


@router.post("/search", response_model=SearchResponse)
def search(
    request: SearchRequest,
    context: RequestContext = Depends(get_request_context),
) -> SearchResponse:
    tracer = get_langfuse_tracer()
    with tracer.span(
        "api.search",
        input={
            "query": request.query,
            "top_k": request.top_k,
            "mode": request.mode,
            "use_hyde": request.use_hyde,
            "use_rerank": request.use_rerank,
        },
        metadata={"request_id": context.request_id},
    ) as span:
        try:
            strategy = normalize_strategy(
                request.mode,
                use_hyde=request.use_hyde,
                use_rerank=request.use_rerank,
            )
            response = search_with_strategy(
                query=request.query,
                strategy=strategy,
                elasticsearch_url=settings.elasticsearch_url,
                index_name=settings.elasticsearch_index,
                model_name=settings.embedding_model,
                top_k=request.top_k,
                rrf_k=settings.retrieval_rrf_k,
                reranker_model=settings.reranker_model,
            )
        except Exception as exc:
            span.update(output={"error": str(exc)})
            raise_api_error(
                code="search_failed",
                message=str(exc),
                context=context,
            )

        api_response = serialize_search_response(
            request_id=context.request_id,
            latency_ms=latency_ms(context),
            response=response,
            request_mode=request.mode,
            fallback_strategy=strategy,
            trace_id=tracer.get_trace_id(),
            trace_url=tracer.get_trace_url(),
            route="search",
        )
        span.update(
            output={
                "results_count": len(api_response.results),
                "chunk_ids": [result.chunk_id for result in api_response.results],
                "latency_ms": api_response.latency_ms,
            }
        )
        return api_response
