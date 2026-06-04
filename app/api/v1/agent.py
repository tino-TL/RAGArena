from __future__ import annotations

from typing import Any, cast

from fastapi import APIRouter, Depends

from app.dependencies import RequestContext, get_request_context, latency_ms, raise_api_error
from app.schemas import AgentRequest, AgentResponse
from ragarena.agent.workflow import run_agentic_rag
from ragarena.observability import get_langfuse_tracer
from ragarena.observability.trace_summary import TraceSummary

router = APIRouter(tags=["agent"])


@router.post("/agent", response_model=AgentResponse)
def agent(
    request: AgentRequest,
    context: RequestContext = Depends(get_request_context),
) -> AgentResponse:
    with get_langfuse_tracer().span(
        "api.agent",
        input={"query": request.query, "max_rewrite": request.max_rewrite},
        metadata={"request_id": context.request_id},
    ) as span:
        try:
            state = run_agentic_rag(request.query, max_rewrite=request.max_rewrite)
        except Exception as exc:
            span.update(output={"error": str(exc)})
            raise_api_error(
                code="agent_workflow_failed",
                message=str(exc),
                context=context,
            )

        response = AgentResponse(
            request_id=context.request_id,
            latency_ms=latency_ms(context),
            original_query=state["original_query"],
            current_query=state["current_query"],
            route=state["route"],
            route_reason=state.get("route_reason"),
            route_confidence=state.get("route_confidence", 0.0),
            guardrail_passed=state.get("guardrail_passed", True),
            guardrail_reason=state.get("guardrail_reason"),
            trace=state["trace"],
            generation=state["generation"],
            grade=state["grade"],
            grade_score=state.get("grade_score", 0.0),
            grade_reason=state.get("grade_reason"),
            rewrite_reason=state.get("rewrite_reason"),
            rewrite_count=state["rewrite_count"],
            max_rewrite=state["max_rewrite"],
            documents_count=len(state["documents"]),
            trace_steps=cast(Any, state.get("trace_steps", [])),
            trace_summary=cast(
                dict[str, object],
                state.get("trace_summary") or TraceSummary.from_state(cast(dict[str, Any], state)).to_dict(),
            ),
        )
        span.update(
            output={
                "route": response.route,
                "grade": response.grade,
                "rewrite_count": response.rewrite_count,
                "documents_count": response.documents_count,
                "generation": response.generation,
                "latency_ms": response.latency_ms,
                "trace_summary": response.trace_summary,
            }
        )
        return response
