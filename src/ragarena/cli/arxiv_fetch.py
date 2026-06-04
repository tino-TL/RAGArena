from __future__ import annotations

import argparse
import asyncio

from dotenv import load_dotenv

from ragarena.pipeline.steps import fetch_and_store_papers, store_papers

__all__ = ["fetch_and_store_papers", "store_papers"]


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Fetch arXiv paper metadata into RAGArena")
    parser.add_argument(
        "--query",
        default="cat:cs.AI OR cat:cs.CL OR cat:cs.LG",
        help="arXiv search query, for example: all:retrieval augmented generation",
    )
    parser.add_argument("--max-results", type=int, default=20)
    parser.add_argument(
        "--no-sync-documents",
        action="store_true",
        help="Only store papers metadata; do not sync abstracts into documents.",
    )
    args = parser.parse_args()

    asyncio.run(
        fetch_and_store_papers(
            query=args.query,
            max_results=args.max_results,
            sync_documents=not args.no_sync_documents,
        )
    )
