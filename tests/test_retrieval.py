from __future__ import annotations

import ragarena.retrieval.search as search_module
from ragarena.reranking.reranker import BGEReranker
from ragarena.retrieval.hyde import build_hyde_search_text
from ragarena.retrieval.search import SearchResponse, fuse_rrf_sources
from ragarena.retrieval.indexer import build_indexing_result
from ragarena.retrieval.vector_store import ElasticsearchVectorStore, SearchResult, dedupe_search_results


def test_hyde_falls_back_to_original_query_without_local_model(monkeypatch) -> None:
    class FakeGenerator:
        def generate(self, prompt: str, system_prompt: str, *, json_mode: bool):
            raise RuntimeError("ollama unavailable")

    monkeypatch.setattr(
        "ragarena.retrieval.hyde.get_ollama_decision_generator",
        lambda url, model, timeout, keep_alive: FakeGenerator(),
    )

    assert build_hyde_search_text("LangGraph") == "LangGraph"


def test_hyde_uses_local_model_without_json_mode(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeGenerator:
        def generate(self, prompt: str, system_prompt: str, *, json_mode: bool):
            captured["prompt"] = prompt
            captured["system_prompt"] = system_prompt
            captured["json_mode"] = json_mode

            class Result:
                answer = "LangGraph uses graph state and retrieval nodes."

            return Result()

    monkeypatch.setattr(
        "ragarena.retrieval.hyde.get_ollama_decision_generator",
        lambda url, model, timeout, keep_alive: FakeGenerator(),
    )

    assert build_hyde_search_text("LangGraph") == "LangGraph\n\nLangGraph uses graph state and retrieval nodes."
    assert captured["json_mode"] is False
    assert "LangGraph" in str(captured["prompt"])


def test_rrf_supports_hyde_source_scores() -> None:
    result = SearchResult(
        chunk_id=1,
        document_id=1,
        content="LangGraph context",
        score=0.9,
        model_name="BAAI/bge-m3",
        source_scores={"hyde_vector": 0.9},
        metadata={"arxiv_id": "2401.00001v1"},
    )

    fused = fuse_rrf_sources([("hyde_vector", [result])])

    assert fused[0].chunk_id == 1
    assert "hyde_vector" in fused[0].source_scores
    assert fused[0].metadata["arxiv_id"] == "2401.00001v1"


def test_reranker_falls_back_when_model_unavailable() -> None:
    reranker = BGEReranker.__new__(BGEReranker)
    reranker.model_name = "missing/local-reranker"
    reranker.model = None
    reranker.load_error = "not loaded"
    results = [
        SearchResult(
            chunk_id=1,
            document_id=1,
            content="LangGraph context",
            score=0.5,
            model_name="BAAI/bge-m3",
            source_scores={"hybrid": 0.5},
        )
    ]

    assert reranker.rerank("LangGraph", results, top_k=1) == results


def test_hybrid_search_rerank_fallback_keeps_base_strategy(monkeypatch) -> None:
    base_result = SearchResult(
        chunk_id=1,
        document_id=1,
        content="LangGraph context",
        score=0.5,
        model_name="BAAI/bge-m3",
        source_scores={"hybrid": 0.5},
    )

    def fake_bm25_search(**kwargs):
        return SearchResponse(
            query=str(kwargs["query"]),
            top_k=int(kwargs["top_k"]),
            mode="bm25",
            strategy="bm25",
            results=[base_result],
        )

    def fake_vector_search(**kwargs):
        return SearchResponse(
            query=str(kwargs["query"]),
            top_k=int(kwargs["top_k"]),
            mode="dense",
            strategy="dense",
            results=[base_result],
        )

    class FakeFallbackReranker:
        model = None
        load_error = "model unavailable"

        def rerank(self, query: str, results: list[SearchResult], top_k: int) -> list[SearchResult]:
            return results[:top_k]

    monkeypatch.setattr(search_module, "bm25_search", fake_bm25_search)
    monkeypatch.setattr(search_module, "vector_search", fake_vector_search)
    monkeypatch.setattr(search_module, "get_bge_reranker", lambda model_name, max_content_chars=None: FakeFallbackReranker())

    response = search_module.hybrid_search(
        query="LangGraph",
        elasticsearch_url="http://localhost:9200",
        use_rerank=True,
    )

    assert response.rerank_attempted is True
    assert response.rerank_succeeded is False
    assert response.used_rerank is False
    assert response.rerank_fallback_reason == "model unavailable"
    assert response.mode == "hybrid+rerank"
    assert response.strategy == "hybrid"


def test_hybrid_search_rerank_success_reports_attempted_mode_and_effective_strategy(monkeypatch) -> None:
    base_result = SearchResult(
        chunk_id=1,
        document_id=1,
        content="LangGraph context",
        score=0.5,
        model_name="BAAI/bge-m3",
        source_scores={"hybrid": 0.5},
    )

    def fake_bm25_search(**kwargs):
        return SearchResponse(
            query=str(kwargs["query"]),
            top_k=int(kwargs["top_k"]),
            mode="bm25",
            strategy="bm25",
            results=[base_result],
        )

    def fake_vector_search(**kwargs):
        return SearchResponse(
            query=str(kwargs["query"]),
            top_k=int(kwargs["top_k"]),
            mode="dense",
            strategy="dense",
            results=[base_result],
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

    monkeypatch.setattr(search_module, "bm25_search", fake_bm25_search)
    monkeypatch.setattr(search_module, "vector_search", fake_vector_search)
    monkeypatch.setattr(search_module, "get_bge_reranker", lambda model_name, max_content_chars=None: FakeSuccessfulReranker())

    response = search_module.hybrid_search(
        query="LangGraph",
        elasticsearch_url="http://localhost:9200",
        use_rerank=True,
    )

    assert response.rerank_attempted is True
    assert response.rerank_succeeded is True
    assert response.used_rerank is True
    assert response.mode == "hybrid+rerank"
    assert response.strategy == "hybrid_rerank"


def test_hybrid_search_limits_rerank_candidates(monkeypatch) -> None:
    bm25_results = [
        SearchResult(
            chunk_id=index,
            document_id=1,
            content=f"BM25 candidate {index}",
            score=float(100 - index),
            model_name="BAAI/bge-m3",
            source_scores={"bm25": float(100 - index)},
        )
        for index in range(1, 31)
    ]
    vector_results = [
        SearchResult(
            chunk_id=index,
            document_id=1,
            content=f"Vector candidate {index}",
            score=float(100 - index),
            model_name="BAAI/bge-m3",
            source_scores={"vector": float(100 - index)},
        )
        for index in range(31, 61)
    ]
    captured: dict[str, int] = {}

    def fake_bm25_search(**kwargs):
        return SearchResponse(
            query=str(kwargs["query"]),
            top_k=int(kwargs["top_k"]),
            mode="bm25",
            strategy="bm25",
            results=bm25_results,
        )

    def fake_vector_search(**kwargs):
        return SearchResponse(
            query=str(kwargs["query"]),
            top_k=int(kwargs["top_k"]),
            mode="dense",
            strategy="dense",
            results=vector_results,
        )

    class FakeSuccessfulReranker:
        model = object()
        load_error = None

        def rerank(self, query: str, results: list[SearchResult], top_k: int) -> list[SearchResult]:
            captured["candidate_count"] = len(results)
            return results[:top_k]

    monkeypatch.setattr(search_module, "bm25_search", fake_bm25_search)
    monkeypatch.setattr(search_module, "vector_search", fake_vector_search)
    monkeypatch.setattr(search_module, "get_bge_reranker", lambda model_name, max_content_chars=None: FakeSuccessfulReranker())

    response = search_module.hybrid_search(
        query="LangGraph",
        elasticsearch_url="http://localhost:9200",
        top_k=10,
        use_rerank=True,
        rerank_candidate_limit=12,
    )

    assert captured["candidate_count"] == 12
    assert len(response.results) == 10
    assert response.rerank_succeeded is True


def test_hybrid_search_falls_back_when_reranker_raises(monkeypatch) -> None:
    base_results = [
        SearchResult(
            chunk_id=index,
            document_id=1,
            content=f"candidate {index}",
            score=float(100 - index),
            model_name="BAAI/bge-m3",
            source_scores={"bm25": float(100 - index)},
        )
        for index in range(1, 4)
    ]

    def fake_bm25_search(**kwargs):
        return SearchResponse(
            query=str(kwargs["query"]),
            top_k=int(kwargs["top_k"]),
            mode="bm25",
            strategy="bm25",
            results=base_results,
        )

    def fake_vector_search(**kwargs):
        return SearchResponse(
            query=str(kwargs["query"]),
            top_k=int(kwargs["top_k"]),
            mode="dense",
            strategy="dense",
            results=[],
        )

    class FakeBrokenReranker:
        model = object()
        load_error = None

        def rerank(self, query: str, results: list[SearchResult], top_k: int) -> list[SearchResult]:
            raise RuntimeError("cuda out of memory")

    monkeypatch.setattr(search_module, "bm25_search", fake_bm25_search)
    monkeypatch.setattr(search_module, "vector_search", fake_vector_search)
    monkeypatch.setattr(search_module, "get_bge_reranker", lambda model_name, max_content_chars=None: FakeBrokenReranker())

    response = search_module.hybrid_search(
        query="LangGraph",
        elasticsearch_url="http://localhost:9200",
        top_k=2,
        use_rerank=True,
    )

    assert [result.chunk_id for result in response.results] == [1, 2]
    assert response.rerank_attempted is True
    assert response.rerank_succeeded is False
    assert response.used_rerank is False
    assert response.rerank_fallback_reason == "rerank failed: cuda out of memory"
    assert response.strategy == "hybrid"


def test_elasticsearch_document_id_uses_stable_chunk_id() -> None:
    assert ElasticsearchVectorStore.document_id(42, "BAAI/bge-m3") == "42"


def test_search_results_dedupe_by_chunk_id_keeps_highest_score() -> None:
    low = SearchResult(
        chunk_id=1,
        document_id=1,
        content="old duplicate",
        score=0.1,
        model_name="BAAI/bge-m3",
        source_scores={"vector": 0.1},
    )
    high = SearchResult(
        chunk_id=1,
        document_id=1,
        content="new duplicate",
        score=0.9,
        model_name="BAAI/bge-m3",
        source_scores={"vector": 0.9},
    )
    other = SearchResult(
        chunk_id=2,
        document_id=1,
        content="other",
        score=0.5,
        model_name="BAAI/bge-m3",
        source_scores={"vector": 0.5},
    )

    deduped = dedupe_search_results([low, other, high])

    assert [result.chunk_id for result in deduped] == [1, 2]
    assert deduped[0].content == "new duplicate"


def test_indexing_result_reports_count_mismatch() -> None:
    result = build_indexing_result(
        deleted_existing_index=True,
        loaded_embeddings=19,
        indexed_chunks=19,
        final_es_count=1291,
        postgres_embedding_count=19,
    )

    assert result.count_matches is False
