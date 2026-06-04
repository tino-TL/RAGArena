from __future__ import annotations

from ragarena.observability.langfuse_tracer import LangfuseTracer
from ragarena.observability.trace_summary import TraceSummary
from ragarena.retrieval.search import SearchResponse
from ragarena.retrieval.vector_store import SearchResult


def test_langfuse_tracer_noop_when_disabled() -> None:
    tracer = LangfuseTracer(
        enabled=False,
        public_key=None,
        secret_key=None,
        host="https://cloud.langfuse.com",
    )

    with tracer.span("test.span", input={"query": "hello"}) as span:
        span.update(output={"ok": True})

    assert tracer.enabled is False


def test_trace_summary_keeps_hybrid_rerank_scores() -> None:
    response = SearchResponse(
        query="query",
        top_k=1,
        mode="hybrid+rerank",
        strategy="hybrid_rerank",
        rerank_attempted=True,
        rerank_succeeded=True,
        used_rerank=True,
        results=[
            SearchResult(
                chunk_id=10,
                document_id=2,
                content="Relevant chunk content",
                score=0.8,
                model_name="BAAI/bge-m3",
                source_scores={"bm25": 3.0, "vector": 0.7, "rerank": 0.8},
                section_name="3 Method",
            )
        ],
    )

    summary = TraceSummary.from_search_response(response).to_dict()

    assert summary["strategy"] == "hybrid_rerank"
    assert summary["rerank_attempted"] is True
    assert summary["rerank_succeeded"] is True
    assert summary["used_rerank"] is True
    assert summary["rerank_fallback_reason"] is None
    chunk = summary["retrieved_chunks"][0]
    assert chunk["chunk_id"] == 10
    assert chunk["section_name"] == "3 Method"
    assert chunk["source_scores"]["rerank"] == 0.8


def test_trace_summary_from_state_without_langfuse_ids() -> None:
    summary = TraceSummary.from_state(
        {
            "route": "local_rag",
            "route_confidence": 0.9,
            "route_reason": "local_knowledge_terms",
            "retrieval_strategy": "hybrid",
            "used_hyde": True,
            "rerank_attempted": True,
            "rerank_succeeded": False,
            "used_rerank": False,
            "rerank_fallback_reason": "model unavailable",
            "rewrite_count": 1,
            "rewrite_reason": "missing evidence",
            "grade": True,
            "grade_score": 0.8,
            "grade_reason": "enough evidence",
            "useful_chunk_ids": [1],
            "retrieval_results": [],
        }
    ).to_dict()

    assert summary["trace_id"] is None
    assert summary["trace_url"] is None
    assert summary["route"] == "local_rag"
    assert summary["route_confidence"] == 0.9
    assert summary["route_reason"] == "local_knowledge_terms"
    assert summary["grader_decision"] == "relevant"
    assert summary["rerank_attempted"] is True
    assert summary["rerank_succeeded"] is False
    assert summary["used_rerank"] is False
    assert summary["rerank_fallback_reason"] == "model unavailable"
    assert summary["grader_score"] == 0.8
    assert summary["grader_reason"] == "enough evidence"
    assert summary["useful_chunk_ids"] == [1]
    assert summary["rewrite_reason"] == "missing evidence"


def test_langfuse_tracer_noop_without_credentials() -> None:
    tracer = LangfuseTracer(
        enabled=True,
        public_key=None,
        secret_key=None,
        host="https://cloud.langfuse.com",
    )

    with tracer.generation("test.generation", model="deepseek-chat", input="hello") as generation:
        generation.update(output="world")

    assert tracer.enabled is False
