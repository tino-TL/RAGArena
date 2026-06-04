from __future__ import annotations

from time import perf_counter

from ragarena.agent.policies.grader import GradeDecision, grade_documents
from ragarena.agent.policies.guardrail import evaluate_guardrail
from ragarena.agent.policies.rewriter import rewrite_query
from ragarena.agent.policies.router import route_query
from ragarena.config import settings
from ragarena.generation.direct import generate_direct_answer
from ragarena.generation.prompt import build_rag_prompt
from ragarena.agent.state import AgentState
from ragarena.agent.state_utils import (
    deserialize_search_results,
    record_trace_step,
    sync_citations_state,
    sync_reranked_results,
    sync_retrieval_state,
)
from ragarena.observability import get_langfuse_tracer
from ragarena.observability.trace_summary import compact_preview, compact_search_result
from ragarena.retrieval.search import hybrid_search
from ragarena.retrieval.vector_store import SearchResult
from ragarena.runtime import get_bge_reranker, get_deepseek_generator

DEFAULT_TOP_K = 3
GIVE_UP_MESSAGE = "\u5f53\u524d\u672c\u5730\u77e5\u8bc6\u5e93\u4e0d\u8db3\u4ee5\u56de\u7b54\u8be5\u95ee\u9898\u3002"
GUARDRAIL_MESSAGE = "\u8bf7\u8f93\u5165\u4e00\u4e2a\u6e05\u6670\u7684\u672c\u5730\u77e5\u8bc6\u5e93\u95ee\u9898\u3002"


def guardrail_node(state: AgentState) -> AgentState:
    started_at = perf_counter()
    with get_langfuse_tracer().span(
        "graph.guardrail",
        input={"query": state["current_query"]},
    ) as span:
        decision = evaluate_guardrail(state["current_query"])
        state["guardrail_passed"] = decision.passed
        state["guardrail_reason"] = decision.reason
        state["trace"].append(f"guardrail: {decision.reason}")
        record_trace_step(
            state,
            "guardrail",
            started_at,
            query=state["current_query"],
            message=decision.reason,
        )
        span.update(output={"passed": decision.passed, "reason": decision.reason})
    return state


def router_node(state: AgentState) -> AgentState:
    started_at = perf_counter()
    with get_langfuse_tracer().span(
        "graph.router",
        input={"original_query": state["original_query"], "current_query": state["current_query"]},
    ) as span:
        decision = route_query(state["current_query"])
        state["route"] = decision.route
        state["route_confidence"] = decision.confidence
        state["route_reason"] = decision.reason
        state["trace"].append(f"router: {state['route']}")
        record_trace_step(
            state,
            "router",
            started_at,
            query=state["current_query"],
            route=state["route"],
            route_confidence=state["route_confidence"],
            route_reason=state["route_reason"],
        )
        span.update(
            output={
                "route": state["route"],
                "confidence": state["route_confidence"],
                "reason": state["route_reason"],
            }
        )
    return state


def direct_answer_node(state: AgentState) -> AgentState:
    started_at = perf_counter()
    with get_langfuse_tracer().span("graph.direct_answer", input={"query": state["current_query"]}) as span:
        state["generation"] = generate_direct_answer(state["current_query"])
        state["trace"].append("direct_answer")
        record_trace_step(
            state,
            "direct_answer",
            started_at,
            query=state["current_query"],
            message="generated direct answer",
        )
        span.update(output={"generation": state["generation"]})
    return state


def hybrid_retrieve_node(state: AgentState) -> AgentState:
    started_at = perf_counter()
    with get_langfuse_tracer().span(
        "graph.retrieve",
        input={"query": state["current_query"], "top_k": DEFAULT_TOP_K},
        metadata={"index": settings.elasticsearch_index, "mode": "hybrid"},
    ) as span:
        response = hybrid_search(
            query=state["current_query"],
            elasticsearch_url=settings.elasticsearch_url,
            index_name=settings.elasticsearch_index,
            model_name=settings.embedding_model,
            top_k=DEFAULT_TOP_K,
            use_hyde=settings.hyde_enabled,
            use_rerank=False,
            reranker_model=settings.reranker_model,
        )
        sync_retrieval_state(state, response)
        state["trace"].append(f"hybrid_retrieve: {len(state['documents'])} docs")
        record_trace_step(
            state,
            "hybrid_retrieve",
            started_at,
            query=state["current_query"],
            retrieval_mode=response.mode,
            chunk_ids=[result.chunk_id for result in response.results],
            scores=[result.score for result in response.results],
        )
        span.update(
            output={
                "documents_count": len(state["documents"]),
                "strategy": response.strategy,
                "candidate_count": response.candidate_count,
                "top_k": response.top_k,
                "used_hyde": response.used_hyde,
                "used_rerank": response.used_rerank,
                "rerank_fallback_reason": response.rerank_fallback_reason,
                "chunk_ids": [result.chunk_id for result in response.results],
                "scores": [result.score for result in response.results],
                "retrieved_chunks": [compact_search_result(result) for result in response.results],
            }
        )
    return state


def rerank_node(state: AgentState) -> AgentState:
    started_at = perf_counter()
    with get_langfuse_tracer().span(
        "graph.rerank",
        input={"query": state["current_query"], "documents_count": len(state["documents"])},
    ) as span:
        results = deserialize_search_results(state.get("retrieval_results", []))
        state["rerank_attempted"] = False
        state["rerank_succeeded"] = False
        state["used_rerank"] = False
        state["rerank_fallback_reason"] = None
        if settings.rerank_enabled and results:
            state["rerank_attempted"] = True
            reranker = get_bge_reranker(settings.reranker_model)
            if reranker.model is None:
                message = reranker.load_error or "reranker model unavailable"
                state["rerank_fallback_reason"] = message
            else:
                results = reranker.rerank(state["current_query"], results, top_k=DEFAULT_TOP_K)
                sync_reranked_results(state, results)
                state["rerank_succeeded"] = True
                state["used_rerank"] = True
                state["retrieval_strategy"] = _strategy_with_rerank(state.get("retrieval_strategy"))
                message = "reranked"
        else:
            message = "disabled" if not settings.rerank_enabled else "no documents"
            if settings.rerank_enabled:
                state["rerank_fallback_reason"] = "no documents"
        state["trace"].append(f"rerank: {message}")
        record_trace_step(
            state,
            "rerank",
            started_at,
            query=state["current_query"],
            retrieval_mode="rerank" if state["rerank_succeeded"] else "none",
            chunk_ids=[result.chunk_id for result in results],
            scores=[result.score for result in results],
            message=message,
            rerank_attempted=state["rerank_attempted"],
            rerank_succeeded=state["rerank_succeeded"],
            rerank_fallback_reason=state.get("rerank_fallback_reason"),
        )
        span.update(
            output={
                "message": message,
                "documents_count": len(results),
                "rerank_attempted": state["rerank_attempted"],
                "rerank_succeeded": state["rerank_succeeded"],
                "used_rerank": state["used_rerank"],
                "rerank_fallback_reason": state.get("rerank_fallback_reason"),
                "strategy": state.get("retrieval_strategy"),
                "retrieved_chunks": [compact_search_result(result) for result in results],
            }
        )
    return state


def _strategy_with_rerank(strategy: str | None) -> str | None:
    if strategy == "hybrid":
        return "hybrid_rerank"
    if strategy == "hybrid_hyde":
        return "hybrid_hyde_rerank"
    return strategy


def grade_documents_node(state: AgentState) -> AgentState:
    started_at = perf_counter()
    with get_langfuse_tracer().span(
        "graph.grade_documents",
        input={"query": state["current_query"], "documents_count": len(state["documents"])},
    ) as span:
        decision = normalize_grade_decision(
            grade_documents(
                state["current_query"],
                state["documents"],
                retrieved_chunks=state.get("retrieval_results", []),
            ),
            document_count=len(state["documents"]),
        )
        state["grade"] = decision.sufficient
        state["grade_score"] = decision.score
        state["grade_reason"] = decision.reason
        state["useful_chunk_ids"] = decision.useful_chunk_ids
        state["suggested_rewrite"] = decision.suggested_rewrite
        state["trace"].append(f"grade_documents: {state['grade']}")
        record_trace_step(
            state,
            "grade_documents",
            started_at,
            query=state["current_query"],
            grade=state["grade"],
            grade_score=state["grade_score"],
            grade_reason=state["grade_reason"],
        )
        span.update(
            output={
                "decision": "sufficient" if decision.sufficient else "insufficient",
                "relevant": decision.relevant,
                "score": decision.score,
                "reason": decision.reason,
                "useful_chunk_ids": decision.useful_chunk_ids,
                "suggested_rewrite": decision.suggested_rewrite,
            }
        )
    return state


def rewrite_query_node(state: AgentState) -> AgentState:
    started_at = perf_counter()
    old_query = state["current_query"]
    with get_langfuse_tracer().span(
        "graph.rewrite_query",
        input={"query": state["current_query"], "rewrite_count": state["rewrite_count"]},
    ) as span:
        previews = [compact_preview(result.content) for result in deserialize_search_results(state.get("retrieval_results", []))]
        suggested_rewrite = state.get("suggested_rewrite")
        state["current_query"] = suggested_rewrite or rewrite_query(
            state["original_query"],
            current_query=state["current_query"],
            grade_reason=state.get("grade_reason"),
            retrieved_doc_previews=previews,
            rewrite_count=state["rewrite_count"],
        )
        state["rewrite_count"] += 1
        state["rewrite_reason"] = state.get("grade_reason") or "document_grade_failed"
        state["trace"].append(f"rewrite_query[{state['rewrite_count']}]: {state['current_query']}")
        record_trace_step(
            state,
            "rewrite_query",
            started_at,
            query=state["current_query"],
            message=state["rewrite_reason"],
        )
        span.update(
            output={
                "old_query": old_query,
                "new_query": state["current_query"],
                "rewrite_count": state["rewrite_count"],
                "reason": state["rewrite_reason"],
            }
        )
    return state


def generate_answer_node(state: AgentState) -> AgentState:
    started_at = perf_counter()
    with get_langfuse_tracer().span(
        "graph.generate_answer",
        input={"query": state["current_query"], "documents_count": len(state["documents"])},
    ) as span:
        chunks = deserialize_search_results(state.get("retrieval_results", []))
        if not chunks:
            chunks = [
                SearchResult(
                    chunk_id=index,
                    document_id=0,
                    content=document,
                    score=0.0,
                    model_name=settings.embedding_model,
                    source_scores={},
                )
                for index, document in enumerate(state["documents"], start=1)
            ]
        prompt = build_rag_prompt(state["current_query"], chunks)
        generator = get_deepseek_generator(settings.deepseek_api_key, settings.deepseek_model)

        if not generator.is_configured():
            state["generation"] = "DEEPSEEK_API_KEY is not configured."
        else:
            with get_langfuse_tracer().generation(
                "graph.generate_answer.llm",
                input={"query": state["current_query"], "prompt_preview": compact_preview(prompt)},
                metadata={"model": settings.deepseek_model},
            ) as generation:
                result = generator.generate(prompt)
                state["generation"] = result.answer
                generation.update(output={"answer": state["generation"]})

        citations = [
            {
                "chunk_id": result.chunk_id,
                "document_id": result.document_id,
                "section_name": result.section_name or result.metadata.get("section_name"),
            }
            for result in chunks
        ]
        sync_citations_state(state, citations)
        state["trace"].append("generate_answer")
        record_trace_step(state, "generate_answer", started_at, query=state["current_query"])
        span.update(
            output={
                "prompt_preview": compact_preview(prompt),
                "answer": state["generation"],
                "citations": state["citations"],
                "chunk_ids": [result.chunk_id for result in chunks],
            }
        )
    return state


def give_up_node(state: AgentState) -> AgentState:
    started_at = perf_counter()
    with get_langfuse_tracer().span("graph.give_up", input={"query": state["current_query"]}) as span:
        state["generation"] = GIVE_UP_MESSAGE
        state["trace"].append("give_up")
        record_trace_step(state, "give_up", started_at, query=state["current_query"])
        span.update(output={"generation": state["generation"]})
    return state


def guardrail_reject_node(state: AgentState) -> AgentState:
    started_at = perf_counter()
    with get_langfuse_tracer().span("graph.guardrail_reject", input={"query": state["current_query"]}) as span:
        state["generation"] = GUARDRAIL_MESSAGE
        state["trace"].append("guardrail_reject")
        record_trace_step(
            state,
            "guardrail_reject",
            started_at,
            query=state["current_query"],
            message=state.get("guardrail_reason"),
        )
        span.update(output={"generation": state["generation"], "reason": state.get("guardrail_reason")})
    return state


def normalize_grade_decision(value: object, *, document_count: int) -> GradeDecision:
    if isinstance(value, GradeDecision):
        return value
    relevant = bool(value)
    return GradeDecision(
        relevant=relevant,
        sufficient=relevant,
        score=1.0 if relevant else 0.0,
        reason="legacy_bool_grade",
        useful_chunk_ids=list(range(1, document_count + 1)) if relevant else [],
    )


retrieve_node = hybrid_retrieve_node
grade_node = grade_documents_node
rewrite_node = rewrite_query_node
generate_node = generate_answer_node
