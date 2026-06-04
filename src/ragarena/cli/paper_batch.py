from __future__ import annotations

import argparse
import asyncio

import asyncpg
from dotenv import load_dotenv

from ragarena.cli.rebuild_paper import rebuild_paper
from ragarena.config import settings
from ragarena.papers.repository import fetch_paper_blocks_for_paper, fetch_paper_files
from ragarena.pipeline.steps import embed_chunks, index_embeddings


async def run_paper_batch(
    *,
    limit: int,
    parser: str,
    chunk_strategy: str,
    planner_provider: str | None,
    planner_model: str | None,
    validate_chunks: bool,
    skip_existing: bool,
) -> None:
    paper_files = await fetch_paper_files(settings.postgres_dsn, limit=limit)
    parsed_papers = 0
    failed_papers = 0
    paper_blocks_count = 0
    document_chunks_count = 0

    for paper_file in paper_files:
        if skip_existing and await fetch_paper_blocks_for_paper(settings.postgres_dsn, paper_file.paper_id):
            continue
        try:
            summary = await rebuild_paper(
                paper_id=paper_file.paper_id,
                parser=parser,
                reset=True,
                planner_provider=planner_provider,
                planner_model=planner_model,
                validate_chunks=validate_chunks,
                chunk_strategy=chunk_strategy,
            )
        except Exception as exc:
            failed_papers += 1
            print(f"failed paper_id={paper_file.paper_id} arxiv_id={paper_file.arxiv_id}: {exc}")
            continue
        parsed_papers += 1
        paper_blocks_count += summary.inserted_blocks
        document_chunks_count += summary.inserted_chunks

    await embed_chunks()
    await index_embeddings(recreate=True)
    avg_tokens = await avg_chunk_tokens_for_strategy(chunk_strategy)
    print(f"parsed_papers: {parsed_papers}")
    print(f"failed_papers: {failed_papers}")
    print(f"paper_blocks_count: {paper_blocks_count}")
    print(f"document_chunks_count: {document_chunks_count}")
    print(f"avg_chunk_tokens: {avg_tokens:.2f}")


async def avg_chunk_tokens_for_strategy(chunk_strategy: str) -> float:
    conn = await asyncpg.connect(settings.postgres_dsn)
    try:
        value = await conn.fetchval(
            "SELECT AVG(token_count)::float FROM document_chunks WHERE chunking_strategy = $1",
            chunk_strategy,
        )
        return float(value or 0.0)
    finally:
        await conn.close()


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Batch parse, chunk, embed, and index downloaded papers")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--parser", choices=["docling", "pymupdf"], default="docling")
    parser.add_argument("--chunk-strategy", choices=["fixed", "agentic"], default="agentic")
    parser.add_argument("--planner-provider", choices=["ollama"])
    parser.add_argument("--planner-model")
    parser.add_argument("--validate-chunks", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    asyncio.run(
        run_paper_batch(
            limit=args.limit,
            parser=args.parser,
            chunk_strategy=args.chunk_strategy,
            planner_provider=args.planner_provider,
            planner_model=args.planner_model,
            validate_chunks=args.validate_chunks,
            skip_existing=args.skip_existing,
        )
    )
