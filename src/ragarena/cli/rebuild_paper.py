from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from dataclasses import dataclass, replace
import re

from dotenv import load_dotenv

from ragarena.chunking.agentic_chunker import avg_chunk_tokens, chunk_agentic_documents, optimize_retrieval_units
from ragarena.chunking.boundary_validator import validate_chunk_boundaries
from ragarena.chunking.fixed_chunker import Chunk, chunk_document
from ragarena.chunking.repository import (
    delete_chunks_for_documents,
    ensure_document_chunks_table,
    fetch_documents,
    insert_chunks,
)
from ragarena.config import settings
from ragarena.ingestion.repository import (
    delete_documents_by_source_prefix,
    ensure_documents_table,
    fetch_document_ids_by_source,
    fetch_document_ids_by_source_prefix,
    upsert_documents_by_source,
)
from ragarena.papers.docling_parser import parse_pdf_with_docling
from ragarena.papers.models import PaperBlock, structured_paper_to_document
from ragarena.papers.repository import (
    delete_paper_blocks,
    ensure_papers_table,
    fetch_paper_blocks_for_paper,
    fetch_paper_by_arxiv_id,
    fetch_paper_by_id,
    fetch_paper_file_for_paper,
    insert_paper_blocks,
)
from ragarena.papers.structured_parser import parse_pdf_to_blocks


@dataclass(frozen=True)
class RebuildSummary:
    paper_id: int
    document_id: int | None
    deleted_chunks: int
    deleted_blocks: int
    deleted_block_documents: int
    inserted_blocks: int
    inserted_chunks: int


async def rebuild_paper(
    *,
    paper_id: int | None = None,
    arxiv_id: str | None = None,
    parser: str = "docling",
    reset: bool = False,
    planner_provider: str | None = None,
    planner_model: str | None = None,
    validate_chunks: bool = False,
    chunk_strategy: str = "agentic",
) -> RebuildSummary:
    await ensure_papers_table(settings.postgres_dsn)
    await ensure_documents_table(settings.postgres_dsn)
    await ensure_document_chunks_table(settings.postgres_dsn)

    paper = await resolve_paper(paper_id=paper_id, arxiv_id=arxiv_id)
    paper_file = await fetch_paper_file_for_paper(settings.postgres_dsn, paper.id)
    if paper_file is None:
        raise ValueError(f"no downloaded PDF found for paper_id={paper.id}")

    deleted_chunks = 0
    deleted_block_documents = 0
    deleted_blocks = 0
    if reset:
        document_ids = await fetch_document_ids_by_source(settings.postgres_dsn, paper.source_url)
        block_document_ids = await fetch_document_ids_by_source_prefix(
            settings.postgres_dsn,
            f"{paper.source_url}#block:",
        )
        deleted_chunks = await delete_chunks_for_documents(
            settings.postgres_dsn,
            sorted(set(document_ids + block_document_ids)),
        )
        deleted_block_documents = await delete_documents_by_source_prefix(
            settings.postgres_dsn,
            f"{paper.source_url}#block:",
        )
        deleted_blocks = await delete_paper_blocks(settings.postgres_dsn, paper.id)

    blocks = parse_blocks(parser, paper_file)
    inserted_blocks = await insert_paper_blocks(settings.postgres_dsn, blocks)
    await upsert_documents_by_source(settings.postgres_dsn, [structured_paper_to_document(paper, blocks)])

    refreshed_blocks = await fetch_paper_blocks_for_paper(settings.postgres_dsn, paper.id)
    documents = await fetch_documents(settings.postgres_dsn)
    paper_documents = [document for document in documents if document.source == paper.source_url]
    if not paper_documents:
        raise ValueError(f"no document row found for paper source={paper.source_url}")

    if chunk_strategy == "fixed":
        chunks = chunk_fixed_documents(paper_documents)
    elif chunk_strategy == "agentic":
        chunk_result = chunk_agentic_documents(
            paper_documents,
            refreshed_blocks,
            planner_provider=planner_provider,
            planner_model=planner_model,
        )
        chunks = chunk_result.chunks
    else:
        raise ValueError("rebuild_paper supports --chunk-strategy fixed or --chunk-strategy agentic")

    if validate_chunks and chunk_strategy == "agentic":
        validation = validate_chunk_boundaries(chunks, refreshed_blocks)
        chunks = optimize_retrieval_units(validation.chunks, refreshed_blocks)
        print(
            "Chunk boundary validation stats: "
            f"boundary_issues_found={validation.stats.boundary_issues_found} "
            f"rule_fixes={validation.stats.rule_fixes} "
            f"model_fixes={validation.stats.model_fixes} "
            f"dropped_chunks={validation.stats.dropped_chunks}"
        )
    elif validate_chunks:
        print("Chunk boundary validation skipped for fixed chunk strategy")
    target_document_ids = sorted({document.id for document in paper_documents})
    deleted_chunks += await delete_chunks_for_documents(settings.postgres_dsn, target_document_ids)
    inserted_chunks = await insert_chunks(settings.postgres_dsn, chunks)

    print_rebuild_stats(blocks, chunks)
    return RebuildSummary(
        paper_id=paper.id,
        document_id=paper_documents[0].id if paper_documents else None,
        deleted_chunks=deleted_chunks,
        deleted_blocks=deleted_blocks,
        deleted_block_documents=deleted_block_documents,
        inserted_blocks=inserted_blocks,
        inserted_chunks=inserted_chunks,
    )


def chunk_fixed_documents(documents) -> list[Chunk]:
    chunks = [
        chunk
        for document in documents
        for chunk in chunk_document(document.id, document.content)
    ]
    return annotate_fixed_chunk_sections(chunks)


def annotate_fixed_chunk_sections(chunks: list[Chunk]) -> list[Chunk]:
    current_section_by_document: dict[int, str | None] = {}
    annotated: list[Chunk] = []
    for chunk in sorted(chunks, key=lambda item: (item.document_id, item.chunk_index)):
        section = current_section_by_document.get(chunk.document_id)
        headings = re.findall(r"(?m)^##\s+(.+)$", chunk.content)
        if headings:
            section = headings[-1].strip()
        current_section_by_document[chunk.document_id] = section
        annotated.append(replace(chunk, section_name=section))
    return annotated


async def resolve_paper(*, paper_id: int | None, arxiv_id: str | None):
    if paper_id is None and arxiv_id is None:
        raise ValueError("provide --paper-id or --arxiv-id")
    if paper_id is not None:
        paper = await fetch_paper_by_id(settings.postgres_dsn, paper_id)
    else:
        paper = await fetch_paper_by_arxiv_id(settings.postgres_dsn, arxiv_id or "")
    if paper is None:
        raise ValueError("paper not found")
    return paper


def parse_blocks(parser: str, paper_file) -> list[PaperBlock]:
    if parser == "docling":
        try:
            return parse_pdf_with_docling(paper_file)
        except Exception as exc:
            print(f"Docling parser failed; falling back to PyMuPDF parser: {exc}")
            return parse_pdf_to_blocks(paper_file)
    if parser == "pymupdf":
        return parse_pdf_to_blocks(paper_file)
    raise ValueError("rebuild_paper supports --parser docling or --parser pymupdf")


def print_rebuild_stats(blocks: list[PaperBlock], chunks) -> None:
    block_counts = Counter(block.block_type for block in blocks)
    should_embed_counts = Counter(str(block.should_embed).lower() for block in blocks)
    print(f"Paper blocks: {len(blocks)}")
    print(f"Block type counts: {dict(block_counts)}")
    print(f"Block should_embed counts: {dict(should_embed_counts)}")
    print(f"Document chunks: {len(chunks)}")
    print(f"Chunk type counts: {dict(Counter(chunk.chunk_type for chunk in chunks))}")
    print(f"Average chunk tokens: {avg_chunk_tokens(chunks)}")
    print(f"Min chunk tokens: {min((chunk.token_count for chunk in chunks), default=0)}")
    print(f"Max chunk tokens: {max((chunk.token_count for chunk in chunks), default=0)}")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Reset and rebuild one paper's Docling blocks and retrieval units")
    parser.add_argument("--paper-id", type=int)
    parser.add_argument("--arxiv-id")
    parser.add_argument("--parser", choices=["docling", "pymupdf"], default="docling")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--planner-provider", choices=["ollama"])
    parser.add_argument("--planner-model")
    parser.add_argument("--validate-chunks", action="store_true")
    parser.add_argument("--chunk-strategy", choices=["fixed", "agentic"], default="agentic")
    args = parser.parse_args()

    summary = asyncio.run(
        rebuild_paper(
            paper_id=args.paper_id,
            arxiv_id=args.arxiv_id,
            parser=args.parser,
            reset=args.reset,
            planner_provider=args.planner_provider,
            planner_model=args.planner_model,
            validate_chunks=args.validate_chunks,
            chunk_strategy=args.chunk_strategy,
        )
    )
    print(f"Deleted chunks: {summary.deleted_chunks}")
    print(f"Deleted paper_blocks: {summary.deleted_blocks}")
    print(f"Deleted block documents: {summary.deleted_block_documents}")
    print(f"Inserted paper_blocks: {summary.inserted_blocks}")
    print(f"Inserted document_chunks: {summary.inserted_chunks}")
    print(f"Paper ID: {summary.paper_id}")
    print(f"Document ID: {summary.document_id}")


if __name__ == "__main__":
    main()
