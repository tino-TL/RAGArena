from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Literal

from ragarena.config import settings
from ragarena.retrieval.hyde import build_hyde_search_text
from ragarena.retrieval.vector_store import SearchResult, dedupe_search_results
from ragarena.runtime import get_bge_encoder, get_bge_reranker, get_elasticsearch_vector_store

DEFAULT_TOP_K = 5
RRF_K = settings.retrieval_rrf_k
DEFAULT_CANDIDATE_MULTIPLIER = settings.retrieval_candidate_multiplier
SearchStrategy = Literal[
    "bm25",
    "dense",
    "vector",
    "hybrid",
    "hybrid_hyde",
    "hybrid_rerank",
    "hybrid_hyde_rerank",
]


@dataclass(frozen=True)
class SearchResponse:
    query: str
    top_k: int
    # Requested/attempted retrieval pipeline, including requested optional stages.
    mode: str
    results: list[SearchResult]
    latency_ms: float = 0.0
    candidate_count: int = 0
    # Effective/successful retrieval pipeline after fallbacks are applied.
    strategy: str = ""
    rrf_k: int | None = None
    used_hyde: bool = False
    rerank_attempted: bool = False
    rerank_succeeded: bool = False
    used_rerank: bool = False
    rerank_fallback_reason: str | None = None

def bm25_search(
    query: str,
    elasticsearch_url: str,
    index_name: str = "ragarena_chunks",
    model_name: str = "BAAI/bge-m3",
    top_k: int = DEFAULT_TOP_K,
    chunking_strategy: str | None = None,
    paper_id: int | None = None,
    arxiv_id: str | None = None,
) -> SearchResponse:
    started_at = perf_counter()
    vector_store = get_elasticsearch_vector_store(elasticsearch_url, index_name)
    results = vector_store.bm25_search(
        query=query,
        top_k=top_k,
        model_name=model_name,
        chunking_strategy=chunking_strategy,
        paper_id=paper_id,
        arxiv_id=arxiv_id,
    )
    return SearchResponse(
        query=query,
        top_k=top_k,
        mode="bm25",
        results=results,
        latency_ms=_elapsed_ms(started_at),
        candidate_count=len(results),
        strategy="bm25",
    )


def vector_search(
    query: str,
    elasticsearch_url: str,
    index_name: str = "ragarena_chunks",
    model_name: str = "BAAI/bge-m3",
    top_k: int = DEFAULT_TOP_K,
    source_name: str = "vector",
    chunking_strategy: str | None = None,
    paper_id: int | None = None,
    arxiv_id: str | None = None,
) -> SearchResponse:
    started_at = perf_counter()
    encoder = get_bge_encoder(model_name)
    query_vector = encoder.encode([query], batch_size=1)[0]

    vector_store = get_elasticsearch_vector_store(elasticsearch_url, index_name)
    results = vector_store.knn_search(
        query_vector=query_vector,
        top_k=top_k,
        model_name=model_name,
        chunking_strategy=chunking_strategy,
        paper_id=paper_id,
        arxiv_id=arxiv_id,
    )
    if source_name != "vector":
        results = [
            SearchResult(
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                content=result.content,
                score=result.score,
                model_name=result.model_name,
                source_scores={source_name: result.score},
                section_name=result.section_name,
                metadata=result.metadata,
            )
            for result in results
        ]
    mode = "dense" if source_name == "vector" else source_name
    return SearchResponse(
        query=query,
        top_k=top_k,
        mode=mode,
        results=results,
        latency_ms=_elapsed_ms(started_at),
        candidate_count=len(results),
        strategy=mode,
    )


def hybrid_search(
    query: str,
    elasticsearch_url: str,
    index_name: str = "ragarena_chunks",
    model_name: str = "BAAI/bge-m3",
    top_k: int = DEFAULT_TOP_K,
    rrf_k: int = RRF_K,
    use_hyde: bool = False,
    use_rerank: bool = False,
    reranker_model: str = "BAAI/bge-reranker-v2-m3",
    rerank_candidate_limit: int | None = None,
    chunking_strategy: str | None = None,
    paper_id: int | None = None,
    arxiv_id: str | None = None,
) -> SearchResponse:
    started_at = perf_counter()
    candidate_top_k = max(top_k * DEFAULT_CANDIDATE_MULTIPLIER, 10)
    bm25_response = bm25_search(
        query=query,
        elasticsearch_url=elasticsearch_url,
        index_name=index_name,
        model_name=model_name,
        top_k=candidate_top_k,
        chunking_strategy=chunking_strategy,
        paper_id=paper_id,
        arxiv_id=arxiv_id,
    )
    vector_response = vector_search(
        query=query,
        elasticsearch_url=elasticsearch_url,
        index_name=index_name,
        model_name=model_name,
        top_k=candidate_top_k,
        chunking_strategy=chunking_strategy,
        paper_id=paper_id,
        arxiv_id=arxiv_id,
    )

    sources = [
        ("bm25", bm25_response.results),
        ("vector", vector_response.results),
    ]
    if use_hyde:
        hyde_text = build_hyde_search_text(query)
        hyde_response = vector_search(
            query=hyde_text,
            elasticsearch_url=elasticsearch_url,
            index_name=index_name,
            model_name=model_name,
            top_k=candidate_top_k,
            source_name="hyde_vector",
            chunking_strategy=chunking_strategy,
            paper_id=paper_id,
            arxiv_id=arxiv_id,
        )
        sources.append(("hyde_vector", hyde_response.results))

    fused = fuse_rrf_sources(sources, rrf_k=rrf_k)
    rerank_attempted = False
    rerank_succeeded = False
    rerank_fallback_reason = None
    if use_rerank and fused:
        rerank_attempted = True
        reranker = get_bge_reranker(
            reranker_model,
            max_content_chars=settings.rerank_max_content_chars,
        )
        if reranker.model is None:
            rerank_fallback_reason = reranker.load_error
            fused = fused[:top_k]
        else:
            rerank_limit = rerank_candidate_limit or settings.rerank_candidate_limit
            rerank_candidates = fused[: max(top_k, rerank_limit)]
            try:
                fused = reranker.rerank(query, rerank_candidates, top_k=top_k)
                rerank_succeeded = True
            except Exception as exc:
                rerank_fallback_reason = f"rerank failed: {exc}"
                fused = fused[:top_k]
    elif use_rerank:
        rerank_fallback_reason = "no documents"
        fused = fused[:top_k]
    else:
        fused = fused[:top_k]

    return SearchResponse(
        query=query,
        top_k=top_k,
        mode=_hybrid_mode(use_hyde=use_hyde, use_rerank=use_rerank),
        results=dedupe_search_results(fused, top_k=top_k),
        latency_ms=_elapsed_ms(started_at),
        candidate_count=sum(len(results) for _, results in sources),
        strategy=_hybrid_strategy(use_hyde=use_hyde, use_rerank=rerank_succeeded),
        rrf_k=rrf_k,
        used_hyde=use_hyde,
        rerank_attempted=rerank_attempted,
        rerank_succeeded=rerank_succeeded,
        used_rerank=rerank_succeeded,
        rerank_fallback_reason=rerank_fallback_reason,
    )


def search_chunks(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    mode: SearchStrategy = "dense",
    chunking_strategy: str | None = None,
    paper_id: int | None = None,
    arxiv_id: str | None = None,
    elasticsearch_url: str | None = None,
    index_name: str | None = None,
    model_name: str | None = None,
) -> SearchResponse:
    return search_with_strategy(
        query=query,
        strategy=mode,
        elasticsearch_url=elasticsearch_url or settings.elasticsearch_url,
        index_name=index_name or settings.elasticsearch_index,
        model_name=model_name or settings.embedding_model,
        top_k=top_k,
        chunking_strategy=chunking_strategy,
        paper_id=paper_id,
        arxiv_id=arxiv_id,
    )


def search_with_strategy(
    *,
    query: str,
    strategy: SearchStrategy,
    elasticsearch_url: str,
    index_name: str = "ragarena_chunks",
    model_name: str = "BAAI/bge-m3",
    top_k: int = DEFAULT_TOP_K,
    rrf_k: int = RRF_K,
    reranker_model: str = "BAAI/bge-reranker-v2-m3",
    rerank_candidate_limit: int | None = None,
    chunking_strategy: str | None = None,
    paper_id: int | None = None,
    arxiv_id: str | None = None,
) -> SearchResponse:
    normalized = "dense" if strategy == "vector" else strategy
    if normalized == "bm25":
        return bm25_search(query, elasticsearch_url, index_name, model_name, top_k, chunking_strategy, paper_id, arxiv_id)
    if normalized == "dense":
        return vector_search(query, elasticsearch_url, index_name, model_name, top_k, chunking_strategy=chunking_strategy, paper_id=paper_id, arxiv_id=arxiv_id)
    if normalized == "hybrid":
        return hybrid_search(query, elasticsearch_url, index_name, model_name, top_k, rrf_k, chunking_strategy=chunking_strategy, paper_id=paper_id, arxiv_id=arxiv_id)
    if normalized == "hybrid_hyde":
        return hybrid_search(query, elasticsearch_url, index_name, model_name, top_k, rrf_k, use_hyde=True, chunking_strategy=chunking_strategy, paper_id=paper_id, arxiv_id=arxiv_id)
    if normalized == "hybrid_rerank":
        return hybrid_search(
            query,
            elasticsearch_url,
            index_name,
            model_name,
            top_k,
            rrf_k,
            use_rerank=True,
            reranker_model=reranker_model,
            rerank_candidate_limit=rerank_candidate_limit,
            chunking_strategy=chunking_strategy,
            paper_id=paper_id,
            arxiv_id=arxiv_id,
        )
    if normalized == "hybrid_hyde_rerank":
        return hybrid_search(
            query,
            elasticsearch_url,
            index_name,
            model_name,
            top_k,
            rrf_k,
            use_hyde=True,
            use_rerank=True,
            reranker_model=reranker_model,
            rerank_candidate_limit=rerank_candidate_limit,
            chunking_strategy=chunking_strategy,
            paper_id=paper_id,
            arxiv_id=arxiv_id,
        )
    raise ValueError(f"Unsupported retrieval strategy: {strategy}")


def fuse_rrf(
    bm25_results: list[SearchResult],
    vector_results: list[SearchResult],
    rrf_k: int = RRF_K,
) -> list[SearchResult]:
    return fuse_rrf_sources(
        [
            ("bm25", bm25_results),
            ("vector", vector_results),
        ],
        rrf_k=rrf_k,
    )


def fuse_rrf_sources(
    sources: list[tuple[str, list[SearchResult]]],
    rrf_k: int = RRF_K,
) -> list[SearchResult]:
    by_chunk_id: dict[int, SearchResult] = {}
    rrf_scores: dict[int, float] = {}
    source_scores: dict[int, dict[str, float]] = {}

    for source_name, results in sources:
        for rank, result in enumerate(results, start=1):
            by_chunk_id.setdefault(result.chunk_id, result)
            rrf_scores[result.chunk_id] = rrf_scores.get(result.chunk_id, 0.0) + (
                1.0 / (rrf_k + rank)
            )
            source_scores.setdefault(result.chunk_id, {})[source_name] = result.score

    fused = []
    for chunk_id, result in by_chunk_id.items():
        fused.append(
            SearchResult(
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                content=result.content,
                score=rrf_scores[chunk_id],
                model_name=result.model_name,
                source_scores=source_scores[chunk_id],
                section_name=result.section_name,
                metadata=result.metadata,
            )
        )

    return dedupe_search_results(fused)


def _hybrid_mode(*, use_hyde: bool, use_rerank: bool) -> str:
    suffixes = []
    if use_hyde:
        suffixes.append("hyde")
    if use_rerank:
        suffixes.append("rerank")
    return "hybrid" if not suffixes else f"hybrid+{'+'.join(suffixes)}"


def _hybrid_strategy(*, use_hyde: bool, use_rerank: bool) -> str:
    if use_hyde and use_rerank:
        return "hybrid_hyde_rerank"
    if use_hyde:
        return "hybrid_hyde"
    if use_rerank:
        return "hybrid_rerank"
    return "hybrid"


def _elapsed_ms(started_at: float) -> float:
    return round((perf_counter() - started_at) * 1000, 3)
