from __future__ import annotations

import pytest

from ragarena.agent import nodes
from ragarena.agent.workflow import build_agentic_rag_graph, run_agentic_rag
from ragarena.retrieval.search import SearchResponse
from ragarena.retrieval.vector_store import SearchResult


def fake_search_response(query: str) -> SearchResponse:
    return SearchResponse(
        query=query,
        top_k=3,
        mode="hybrid",
        results=[
            SearchResult(
                chunk_id=1,
                document_id=1,
                content="LangGraph is a graph workflow framework. LangChain is an LLM app framework.",
                score=1.0,
                model_name="BAAI/bge-m3",
                source_scores={"bm25": 2.0, "vector": 1.0, "rerank": 0.9},
                section_name="2 Workflow",
            )
        ],
        strategy="hybrid",
    )


def pe_ratio_search_response(query: str) -> SearchResponse:
    return SearchResponse(
        query=query,
        top_k=3,
        mode="hybrid",
        results=[
            SearchResult(
                chunk_id=41,
                document_id=7,
                content="PE Ratio compares a company's market price with earnings per share.",
                score=1.0,
                model_name="BAAI/bge-m3",
                source_scores={"bm25": 3.0, "vector": 0.8},
                section_name="4.1 PE Ratio",
                metadata={"section_name": "4.1 PE Ratio"},
            )
        ],
        strategy="hybrid",
    )


class FakeSuccessfulReranker:
    model = object()
    load_error = None

    def rerank(self, query: str, results: list[SearchResult], top_k: int) -> list[SearchResult]:
        return [
            SearchResult(
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                content=result.content,
                score=0.9,
                model_name=result.model_name,
                source_scores={**result.source_scores, "rerank": 0.9},
                section_name=result.section_name,
                metadata=result.metadata,
            )
            for result in results[:top_k]
        ]


class FakeFallbackReranker:
    model = None
    load_error = "model unavailable"

    def rerank(self, query: str, results: list[SearchResult], top_k: int) -> list[SearchResult]:
        return results[:top_k]


@pytest.fixture(autouse=True)
def disable_live_qwen(monkeypatch) -> None:
    monkeypatch.setattr(nodes.settings, "agent_decision_enabled", False)


def test_build_agentic_rag_graph_compiles() -> None:
    assert build_agentic_rag_graph() is not None


def test_graph_nodes_are_importable() -> None:
    assert callable(nodes.guardrail_node)
    assert callable(nodes.router_node)
    assert callable(nodes.direct_answer_node)
    assert callable(nodes.retrieve_node)
    assert callable(nodes.grade_node)
    assert callable(nodes.rewrite_node)
    assert callable(nodes.generate_node)
    assert callable(nodes.give_up_node)


def test_run_agentic_rag_returns_state_with_trace(monkeypatch) -> None:
    monkeypatch.setattr(nodes, "hybrid_search", lambda **kwargs: fake_search_response(kwargs["query"]))
    monkeypatch.setattr(nodes.settings, "rerank_enabled", True)
    monkeypatch.setattr(nodes, "get_bge_reranker", lambda model_name: FakeSuccessfulReranker())
    monkeypatch.setattr(
        nodes,
        "grade_documents",
        lambda query, documents, **kwargs: nodes.GradeDecision(True, True, 1.0, "ok", [1]),
    )
    monkeypatch.setattr(nodes, "build_rag_prompt", lambda query, chunks: "prompt")

    class FakeGenerator:
        def is_configured(self) -> bool:
            return True

        def generate(self, prompt: str):
            return type("Result", (), {"answer": "generated answer"})()

    monkeypatch.setattr(nodes, "get_deepseek_generator", lambda api_key, model: FakeGenerator())

    state = run_agentic_rag("LangGraph and LangChain differences")
    assert state["trace"]
    assert state["generation"] == "generated answer"
    assert state["route"] == "local_rag"
    assert state["route_reason"]
    assert state["route_confidence"] > 0
    assert state["retrieval_strategy"] == "hybrid_rerank"
    assert state["rerank_attempted"] is True
    assert state["rerank_succeeded"] is True
    assert state["used_rerank"] is True
    assert state["rerank_fallback_reason"] is None
    assert state["retrieval_results"][0]["source_scores"]["rerank"] == 0.9
    assert state["grade_reason"] == "ok"
    assert state["citations"]
    assert state["trace_summary"]["strategy"] == "hybrid_rerank"
    assert state["trace_summary"]["rerank_attempted"] is True
    assert state["trace_summary"]["rerank_succeeded"] is True
    assert state["trace_summary"]["used_rerank"] is True


def test_rerank_fallback_keeps_base_strategy(monkeypatch) -> None:
    monkeypatch.setattr(nodes.settings, "rerank_enabled", True)
    monkeypatch.setattr(nodes, "get_bge_reranker", lambda model_name: FakeFallbackReranker())
    state = nodes.rerank_node(
        {
            "current_query": "LangGraph",
            "documents": ["LangGraph context"],
            "retrieval_strategy": "hybrid_hyde",
            "retrieval_results": [
                {
                    "chunk_id": 1,
                    "document_id": 1,
                    "content": "LangGraph context",
                    "score": 0.5,
                    "model_name": "BAAI/bge-m3",
                    "source_scores": {"bm25": 1.0, "vector": 0.5},
                    "section_name": None,
                    "metadata": {},
                }
            ],
            "trace": [],
            "trace_steps": [],
        }
    )

    assert state["rerank_attempted"] is True
    assert state["rerank_succeeded"] is False
    assert state["used_rerank"] is False
    assert state["rerank_fallback_reason"] == "model unavailable"
    assert state["retrieval_strategy"] == "hybrid_hyde"


def test_rerank_no_documents_records_fallback_without_attempt(monkeypatch) -> None:
    monkeypatch.setattr(nodes.settings, "rerank_enabled", True)
    state = nodes.rerank_node(
        {
            "current_query": "LangGraph",
            "documents": [],
            "retrieval_strategy": "hybrid",
            "retrieval_results": [],
            "trace": [],
            "trace_steps": [],
        }
    )

    assert state["rerank_attempted"] is False
    assert state["rerank_succeeded"] is False
    assert state["used_rerank"] is False
    assert state["rerank_fallback_reason"] == "no documents"
    assert state["retrieval_strategy"] == "hybrid"


def test_run_agentic_rag_direct_answer_does_not_crash(monkeypatch) -> None:
    monkeypatch.setattr(nodes, "generate_direct_answer", lambda query: "hello")

    state = run_agentic_rag("hello")
    assert state["generation"]
    assert state["route"] in {"direct_answer", "local_rag"}


def test_run_agentic_rag_unknown_query_does_not_crash(monkeypatch) -> None:
    monkeypatch.setattr(nodes, "hybrid_search", lambda **kwargs: fake_search_response(kwargs["query"]))
    monkeypatch.setattr(
        nodes,
        "grade_documents",
        lambda query, documents, **kwargs: nodes.GradeDecision(False, False, 0.0, "missing evidence", [], query),
    )
    monkeypatch.setattr(nodes, "rewrite_query", lambda *args, **kwargs: kwargs.get("current_query") or args[0])

    state = run_agentic_rag("abcxyz123")
    assert state["generation"]
    assert "give_up" in state["trace"]


def test_section_aware_grader_prevents_unnecessary_rewrite(monkeypatch) -> None:
    monkeypatch.setattr(nodes, "hybrid_search", lambda **kwargs: pe_ratio_search_response(kwargs["query"]))
    monkeypatch.setattr(nodes.settings, "rerank_enabled", False)
    monkeypatch.setattr(nodes, "build_rag_prompt", lambda query, chunks: "prompt")

    class FakeGenerator:
        def is_configured(self) -> bool:
            return True

        def generate(self, prompt: str):
            return type("Result", (), {"answer": "PE Ratio is discussed as a valuation metric."})()

    monkeypatch.setattr(nodes, "get_deepseek_generator", lambda api_key, model: FakeGenerator())

    state = run_agentic_rag("What is discussed in Section 4.1 PE Ratio?")

    assert state["grade"] is True
    assert state["grade_reason"] == "section_metadata_match"
    assert state["rewrite_count"] == 0
    assert "give_up" not in state["trace"]
    assert state["generation"] == "PE Ratio is discussed as a valuation metric."


def test_run_agentic_rag_guardrail_rejects_empty_query() -> None:
    state = run_agentic_rag(" ")
    assert state["guardrail_passed"] is False
    assert "guardrail_reject" in state["trace"]
    assert state["generation"]
