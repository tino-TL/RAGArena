from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.v1.cache_helpers import get_cache_json, set_cache_json
from app.api.v1.serializers import serialize_search_result
from app.dependencies import RequestContext, get_request_context, latency_ms, raise_api_error
from app.schemas import AskRequest, AskResponse
from ragarena.config import settings
from ragarena.generation import service as generation_service
from ragarena.observability import get_langfuse_tracer
from ragarena.observability.trace_summary import TraceSummary
from ragarena.cache.redis_cache import cache_key

router = APIRouter(tags=["ask"])


@router.post("/ask", response_model=AskResponse)
def ask(
    request: AskRequest,
    context: RequestContext = Depends(get_request_context),
) -> AskResponse:
    tracer = get_langfuse_tracer()
    with tracer.span(
        "api.ask",
        input={
            "query": request.query,
            "top_k": request.top_k,
            "use_hyde": request.use_hyde,
            "use_rerank": request.use_rerank,
        },
        metadata={"request_id": context.request_id},
    ) as span:
        use_hyde, use_rerank = generation_service.resolve_retrieval_flags(
            use_hyde=request.use_hyde,
            use_rerank=request.use_rerank,
        )
        key = cache_key(
            "ask",
            {
                "query": request.query,
                "top_k": request.top_k,
                "use_hyde": use_hyde,
                "use_rerank": use_rerank,
                "model": settings.deepseek_model,
                "embedding_model": settings.embedding_model,
                "reranker_model": settings.reranker_model,
                "index": settings.elasticsearch_index,
            },
        )
        cached = get_cache_json(key)
        if cached:
            response = AskResponse(
                request_id=context.request_id,
                latency_ms=latency_ms(context),
                query=request.query,
                answer=str(cached["answer"]),
                retrieved_chunks=cached["retrieved_chunks"],
                cache_hit=True,
                trace_summary=dict(cached.get("trace_summary", {})),
            )
            span.update(output={"cache_hit": True, "latency_ms": response.latency_ms})
            return response

        try:
            retrieval = generation_service.retrieve_for_answer(
                query=request.query,
                top_k=request.top_k,
                use_hyde=use_hyde,
                use_rerank=use_rerank,
            )
        except Exception as exc:
            span.update(output={"error": str(exc), "stage": "retrieval"})
            raise_api_error(
                code="retrieval_failed",
                message=str(exc),
                context=context,
            )

        try:
            answer = generation_service.generate_answer(request.query, retrieval)
        except Exception as exc:
            span.update(output={"error": str(exc), "stage": "generation"})
            raise_api_error(
                code="generation_failed",
                message=str(exc),
                context=context,
            )

        response = AskResponse(
            request_id=context.request_id,
            latency_ms=latency_ms(context),
            query=request.query,
            answer=answer,
            retrieved_chunks=[serialize_search_result(result) for result in retrieval.results],
            cache_hit=False,
            trace_summary=TraceSummary.from_search_response(
                retrieval,
                trace_id=tracer.get_trace_id(),
                trace_url=tracer.get_trace_url(),
                route="ask",
            ).to_dict(),
        )
        set_cache_json(
            key,
            {
                "answer": response.answer,
                "retrieved_chunks": [chunk.model_dump() for chunk in response.retrieved_chunks],
                "trace_summary": response.trace_summary,
            },
        )
        span.update(
            output={
                "cache_hit": False,
                "retrieved_chunks_count": len(response.retrieved_chunks),
                "chunk_ids": [chunk.chunk_id for chunk in response.retrieved_chunks],
                "answer": response.answer,
                "latency_ms": response.latency_ms,
            }
        )
        return response
