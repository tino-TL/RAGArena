from __future__ import annotations

from dataclasses import dataclass

from ragarena.config import settings
from ragarena.retrieval.search import SearchResponse, SearchStrategy, search_with_strategy


@dataclass(frozen=True)
class RetrievalRequest:
    query: str
    strategy: SearchStrategy = "hybrid"
    top_k: int = 5


class RetrievalService:
    def search(self, request: RetrievalRequest) -> SearchResponse:
        return search_with_strategy(
            query=request.query,
            strategy=request.strategy,
            elasticsearch_url=settings.elasticsearch_url,
            index_name=settings.elasticsearch_index,
            model_name=settings.embedding_model,
            top_k=request.top_k,
            rrf_k=settings.retrieval_rrf_k,
            reranker_model=settings.reranker_model,
        )


def normalize_strategy(
    mode: str,
    *,
    use_hyde: bool = False,
    use_rerank: bool = False,
) -> SearchStrategy:
    normalized_mode = "dense" if mode == "vector" else mode
    if normalized_mode == "bm25":
        return "bm25"
    if normalized_mode == "dense":
        return "dense"
    if normalized_mode != "hybrid":
        raise ValueError(f"Unsupported retrieval mode: {mode}")
    if use_hyde and use_rerank:
        return "hybrid_hyde_rerank"
    if use_hyde:
        return "hybrid_hyde"
    if use_rerank:
        return "hybrid_rerank"
    return "hybrid"
