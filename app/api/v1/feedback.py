from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends

from app.api.v1.cache_helpers import append_feedback
from app.dependencies import RequestContext, get_request_context, latency_ms
from app.schemas import FeedbackRequest, FeedbackResponse
from ragarena.observability import get_langfuse_tracer

router = APIRouter(tags=["feedback"])


@router.post("/feedback", response_model=FeedbackResponse)
def feedback(
    request: FeedbackRequest,
    context: RequestContext = Depends(get_request_context),
) -> FeedbackResponse:
    feedback_payload = {
        "request_id": request.request_id,
        "score": request.score,
        "comment": request.comment,
        "endpoint": request.endpoint,
        "received_request_id": context.request_id,
        "created_at": datetime.now(UTC).isoformat(),
    }
    with get_langfuse_tracer().span(
        "api.feedback",
        input=feedback_payload,
        metadata={"request_id": context.request_id},
    ) as span:
        stored = append_feedback(feedback_payload)
        response = FeedbackResponse(
            request_id=context.request_id,
            latency_ms=latency_ms(context),
            stored=stored,
            message="feedback stored" if stored else "feedback accepted but Redis is unavailable",
        )
        span.update(output={"stored": stored, "latency_ms": response.latency_ms})
        return response
