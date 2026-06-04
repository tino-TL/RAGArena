from __future__ import annotations

import argparse

from dotenv import load_dotenv

from ragarena.cli.formatters import print_retrieved_chunks, print_trace_summary
from ragarena.generation import service as generation_service
from ragarena.observability.trace_summary import TraceSummary


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Ask RAGArena with hybrid retrieval and DeepSeek generation")
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    use_hyde, use_rerank = generation_service.resolve_retrieval_flags(
        use_hyde=None,
        use_rerank=None,
    )
    retrieval = generation_service.retrieve_for_answer(
        query=args.query,
        top_k=args.top_k,
        use_hyde=use_hyde,
        use_rerank=use_rerank,
    )

    print_retrieved_chunks(retrieval, title="Retrieved chunks")

    answer = generation_service.generate_answer(args.query, retrieval)

    print("Answer")
    print("======")
    print(answer)
    print()
    print_trace_summary(TraceSummary.from_search_response(retrieval, route="ask").to_dict())
