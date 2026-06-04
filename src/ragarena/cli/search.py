from __future__ import annotations

import argparse

from dotenv import load_dotenv

from ragarena.cli.formatters import print_search_response, print_trace_summary
from ragarena.config import settings
from ragarena.observability.trace_summary import TraceSummary
from ragarena.retrieval.search import search_with_strategy
from ragarena.retrieval.service import normalize_strategy


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Search RAGArena chunks")
    parser.add_argument("query")
    parser.add_argument("top_k", nargs="?", type=int, default=5)
    parser.add_argument(
        "--mode",
        choices=["bm25", "vector", "hybrid"],
        default="vector",
    )
    parser.add_argument("--hyde", action="store_true", help="Enable HyDE vector recall for hybrid mode")
    parser.add_argument("--rerank", action="store_true", help="Enable BGE reranker for hybrid mode")
    args = parser.parse_args()

    strategy = normalize_strategy(args.mode, use_hyde=args.hyde, use_rerank=args.rerank)
    response = search_with_strategy(
        query=args.query,
        strategy=strategy,
        elasticsearch_url=settings.elasticsearch_url,
        index_name=settings.elasticsearch_index,
        model_name=settings.embedding_model,
        top_k=args.top_k,
        rrf_k=settings.retrieval_rrf_k,
        reranker_model=settings.reranker_model,
    )

    print_search_response(response)
    print_trace_summary(TraceSummary.from_search_response(response, route="search").to_dict())
