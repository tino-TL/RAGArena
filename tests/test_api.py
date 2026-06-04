from __future__ import annotations

from dataclasses import dataclass

from fastapi.testclient import TestClient

import app.api.v1.agent as agent_routes
import app.api.v1.ask as ask_routes
import app.api.v1.cache_helpers as cache_helpers
import app.api.v1.search as search_routes
import ragarena.cli.ask as ask_cli
from app.main import app
from ragarena.generation import service as generation_service
from ragarena.retrieval.search import SearchResponse as RetrievalSearchResponse
from ragarena.retrieval.vector_store import SearchResult


@dataclass(frozen=True)
class FakeGenerationResult:
    answer: str


class FakeGenerator:
    model = "deepseek-chat"

    def is_configured(self) -> bool:
        return True

    def generate(self, prompt: str) -> FakeGenerationResult:
        return FakeGenerationResult(answer=f"answer from {len(prompt)} chars")

    def stream_generate(self, prompt: str):
        yield "streamed "
        yield "answer"


def fake_search_response(query: str, top_k: int, mode: str) -> RetrievalSearchResponse:
    rerank_succeeded = mode.endswith("_rerank")
    return RetrievalSearchResponse(
        query=query,
        top_k=top_k,
        mode=mode,
        strategy=mode,
        rerank_attempted=rerank_succeeded,
        rerank_succeeded=rerank_succeeded,
        used_rerank=rerank_succeeded,
        results=[
            SearchResult(
                chunk_id=1,
                document_id=1,
                content="LangGraph workflow context",
                score=1.0,
                model_name="BAAI/bge-m3",
                source_scores={mode: 1.0},
            )
        ],
    )


def test_search_endpoint(monkeypatch) -> None:
    monkeypatch.setattr(
        search_routes,
        "search_with_strategy",
        lambda query, strategy, elasticsearch_url, index_name, model_name, top_k, **kwargs: fake_search_response(query, top_k, str(strategy)),
    )

    response = TestClient(app).post(
        "/api/v1/search",
        json={"query": "LangGraph workflow framework", "top_k": 1, "mode": "hybrid"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["request_id"]
    assert payload["latency_ms"] >= 0
    assert payload["mode"] == "hybrid"
    assert payload["rerank_attempted"] is False
    assert payload["rerank_succeeded"] is False
    assert payload["used_rerank"] is False
    assert payload["trace_summary"]["rerank_attempted"] is False
    assert payload["trace_summary"]["rerank_succeeded"] is False
    assert payload["results"][0]["chunk_id"] == 1
    assert "trace_summary" in payload


def test_ask_endpoint(monkeypatch) -> None:
    monkeypatch.setattr(ask_routes, "get_cache_json", lambda key: None)
    monkeypatch.setattr(ask_routes, "set_cache_json", lambda key, value: None)
    monkeypatch.setattr(
        generation_service,
        "retrieve_for_answer",
        lambda query, top_k, **kwargs: fake_search_response(query, top_k, "hybrid"),
    )
    monkeypatch.setattr(generation_service, "get_deepseek_generator", lambda api_key, model: FakeGenerator())

    response = TestClient(app).post(
        "/api/v1/ask",
        json={"query": "LangGraph和LangChain有什么区别", "top_k": 1},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["request_id"]
    assert payload["latency_ms"] >= 0
    assert payload["answer"]
    assert len(payload["retrieved_chunks"]) == 1
    assert payload["cache_hit"] is False
    assert "trace_summary" in payload


def test_ask_endpoint_cache_hit(monkeypatch) -> None:
    monkeypatch.setattr(
        ask_routes,
        "get_cache_json",
        lambda key: {
            "answer": "cached answer",
            "retrieved_chunks": [
                {
                    "score": 1.0,
                    "source_scores": {"hybrid": 1.0},
                    "chunk_id": 1,
                    "document_id": 1,
                    "model_name": "BAAI/bge-m3",
                    "content": "cached chunk",
                    "metadata": {},
                }
            ],
        },
    )

    response = TestClient(app).post(
        "/api/v1/ask",
        json={"query": "LangGraph和LangChain有什么区别", "top_k": 1},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "cached answer"
    assert payload["cache_hit"] is True


def test_ask_cli_uses_generation_service_flags(monkeypatch, capsys) -> None:
    calls = {}

    monkeypatch.setattr("sys.argv", ["ragarena-ask", "LangGraph", "--top-k", "2"])
    monkeypatch.setattr(ask_cli.generation_service, "resolve_retrieval_flags", lambda use_hyde, use_rerank: (True, True))

    def fake_retrieve_for_answer(query, top_k, use_hyde, use_rerank):
        calls["retrieve"] = {
            "query": query,
            "top_k": top_k,
            "use_hyde": use_hyde,
            "use_rerank": use_rerank,
        }
        return fake_search_response(query, top_k, "hybrid_hyde_rerank")

    monkeypatch.setattr(ask_cli.generation_service, "retrieve_for_answer", fake_retrieve_for_answer)
    monkeypatch.setattr(ask_cli.generation_service, "generate_answer", lambda query, retrieval: "answer")

    ask_cli.main()

    assert calls["retrieve"] == {
        "query": "LangGraph",
        "top_k": 2,
        "use_hyde": True,
        "use_rerank": True,
    }
    assert "answer" in capsys.readouterr().out


def test_agent_endpoint(monkeypatch) -> None:
    monkeypatch.setattr(
        agent_routes,
        "run_agentic_rag",
        lambda query, max_rewrite: {
            "original_query": query,
            "current_query": query,
            "route": "local_rag",
            "route_reason": "default_local_rag",
            "route_confidence": 0.6,
            "guardrail_passed": True,
            "guardrail_reason": "passed",
            "documents": ["doc"],
            "generation": "final answer",
            "grade": True,
            "grade_score": 1.0,
            "grade_reason": "ok",
            "rewrite_reason": None,
            "rewrite_count": 0,
            "max_rewrite": max_rewrite,
            "trace": ["router: local_rag", "retrieve: 1 docs", "grade: True", "generate"],
        },
    )

    response = TestClient(app).post(
        "/api/v1/agent",
        json={"query": "LangGraph和LangChain有什么区别", "max_rewrite": 3},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["request_id"]
    assert payload["latency_ms"] >= 0
    assert payload["route"] == "local_rag"
    assert payload["generation"] == "final answer"
    assert "trace_summary" in payload


def test_stream_endpoint(monkeypatch) -> None:
    monkeypatch.setattr(
        generation_service,
        "retrieve_for_answer",
        lambda query, top_k, **kwargs: fake_search_response(query, top_k, "hybrid"),
    )
    monkeypatch.setattr(generation_service, "get_deepseek_generator", lambda api_key, model: FakeGenerator())

    with TestClient(app).stream(
        "POST",
        "/api/v1/stream",
        json={"query": "LangGraph和LangChain有什么区别", "top_k": 1},
    ) as response:
        body = response.read().decode("utf-8")

    assert response.status_code == 200
    assert "event: metadata" in body
    assert "event: retrieval" in body
    assert "event: chunk" in body
    assert "event: done" in body


def test_feedback_endpoint(monkeypatch) -> None:
    monkeypatch.setattr(cache_helpers, "append_feedback", lambda value: True)
    monkeypatch.setattr("app.api.v1.feedback.append_feedback", lambda value: True)

    response = TestClient(app).post(
        "/api/v1/feedback",
        json={
            "request_id": "request-1",
            "score": 1,
            "comment": "useful",
            "endpoint": "/api/v1/agent",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["stored"] is True
    assert payload["request_id"]


def test_search_endpoint_error_response(monkeypatch) -> None:
    def raise_search_error(*args, **kwargs):
        raise RuntimeError("Elasticsearch unavailable")

    monkeypatch.setattr(search_routes, "search_with_strategy", raise_search_error)

    response = TestClient(app).post(
        "/api/v1/search",
        json={"query": "LangGraph workflow framework", "top_k": 1, "mode": "hybrid"},
    )

    assert response.status_code == 503
    payload = response.json()["detail"]
    assert payload["request_id"]
    assert payload["error"]["code"] == "search_failed"
    assert payload["error"]["retryable"] is True
