from __future__ import annotations

import argparse
import asyncio

from dotenv import load_dotenv

from ragarena.pipeline.steps import parse_papers

__all__ = ["parse_papers"]


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Parse downloaded arXiv PDFs into paper sections")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--arxiv-id")
    parser.add_argument("--parser", choices=["simple", "pymupdf", "docling"], default="docling")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--no-sync-documents", action="store_true")
    args = parser.parse_args()

    asyncio.run(
        parse_papers(
            limit=args.limit,
            arxiv_id=args.arxiv_id,
            sync_documents=not args.no_sync_documents,
            parser=args.parser,
            reset=args.reset,
        )
    )
