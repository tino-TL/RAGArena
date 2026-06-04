from __future__ import annotations

import json
from typing import Iterator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.api.v1.serializers import serialize_search_result
from app.dependencies import RequestContext, get_request_context, latency_ms
from app.schemas import StreamRequest
from ragarena.generation import service as generation_service
from ragarena.observability import get_langfuse_tracer

router = APIRouter(tags=["stream"])


@router.post("/stream")
def stream(
    request: StreamRequest,
    context: RequestContext = Depends(get_request_context),
) -> StreamingResponse:
    return StreamingResponse(
        stream_events(request, context),
        media_type="text/event-stream",
    )


def stream_events(request: StreamRequest, context: RequestContext) -> Iterator[str]:
    with get_langfuse_tracer().span(
        "api.stream",
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
        yield sse(
            "metadata",
            {
                "request_id": context.request_id,
                "query": request.query,
                "top_k": request.top_k,
            },
        )

        try:
            retrieval = generation_service.retrieve_for_answer(
                query=request.query,
                top_k=request.top_k,
                use_hyde=use_hyde,
                use_rerank=use_rerank,
            )
        except Exception as exc:
            span.update(output={"error": str(exc), "stage": "retrieval"})
            yield sse(
                "error",
                {
                    "request_id": context.request_id,
                    "error": {
                        "code": "retrieval_failed",
                        "message": str(exc),
                        "retryable": True,
                    },
                },
            )
            return

        yield sse(
            "retrieval",
            {
                "request_id": context.request_id,
                "chunks": [serialize_search_result(result).model_dump() for result in retrieval.results],
            },
        )

        prompt_chunks: list[str] = []
        try:
            for chunk in generation_service.stream_answer(request.query, retrieval):
                prompt_chunks.append(chunk)
                yield sse("chunk", {"request_id": context.request_id, "content": chunk})
        except generation_service.DeepSeekNotConfiguredError:
            span.update(output={"error": "DEEPSEEK_API_KEY is not configured.", "stage": "generation"})
            yield sse(
                "error",
                {
                    "request_id": context.request_id,
                    "error": {
                        "code": "deepseek_not_configured",
                        "message": "DEEPSEEK_API_KEY is not configured.",
                        "retryable": False,
                    },
                },
            )
            return
        except Exception as exc:
            span.update(output={"error": str(exc), "stage": "generation"})
            yield sse(
                "error",
                {
                    "request_id": context.request_id,
                    "error": {
                        "code": "generation_failed",
                        "message": str(exc),
                        "retryable": True,
                    },
                },
            )
            return

        latency = latency_ms(context)
        span.update(
            output={
                "retrieved_chunks_count": len(retrieval.results),
                "answer": "".join(prompt_chunks),
                "latency_ms": latency,
            }
        )
        yield sse(
            "done",
            {
                "request_id": context.request_id,
                "latency_ms": latency,
            },
        )


def sse(event: str, payload: dict[str, object]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
