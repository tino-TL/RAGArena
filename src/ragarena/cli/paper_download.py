from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from dotenv import load_dotenv

from ragarena.papers.downloader import DEFAULT_PAPER_DIR
from ragarena.pipeline.steps import download_papers

__all__ = ["download_papers"]


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Download arXiv PDFs for stored papers")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--arxiv-id")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PAPER_DIR)
    args = parser.parse_args()

    asyncio.run(
        download_papers(
            limit=args.limit,
            output_dir=args.output_dir,
            arxiv_id=args.arxiv_id,
        )
    )
