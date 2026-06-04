from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from collections import Counter
from pathlib import Path
from urllib.parse import urlencode

import requests
from dotenv import load_dotenv

from ragarena.chunking.agentic_chunker import chunk_agentic_documents
from ragarena.chunking.repository import (
    delete_chunks_for_documents,
    ensure_document_chunks_table,
    fetch_documents,
    insert_chunks,
)
from ragarena.cli.rebuild_paper import annotate_fixed_chunk_sections
from ragarena.config import settings
from ragarena.evaluation.qa_generator import qa_from_chunk
from ragarena.ingestion.repository import (
    delete_documents_by_source_prefix,
    ensure_documents_table,
    fetch_document_ids_by_source,
    fetch_document_ids_by_source_prefix,
    upsert_documents_by_source,
)
from ragarena.papers.arxiv_client import ARXIV_API_URL, parse_arxiv_feed
from ragarena.papers.docling_parser import parse_pdf_with_docling
from ragarena.papers.models import PaperFile, PaperMetadata, structured_paper_to_document
from ragarena.papers.repository import (
    delete_paper_blocks,
    ensure_papers_table,
    fetch_paper_by_arxiv_id,
    fetch_paper_blocks_for_paper,
    insert_paper_blocks,
    insert_paper_file,
    insert_papers,
)
from ragarena.papers.structured_parser import parse_pdf_to_blocks
from ragarena.pipeline.steps import embed_chunks, index_embeddings
from ragarena.chunking.fixed_chunker import chunk_document


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Prepare 10-paper ablation corpus with fixed and agentic chunks"
    )
    parser.add_argument("--papers-dir", type=Path, default=Path("E:/ragarena-data/papers"))
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--qa-per-paper", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("data/eval/qa_ablation_100.json"))
    parser.add_argument("--plan-output", type=Path, default=Path("data/eval/ablation_10paper_plan.json"))
    parser.add_argument("--parser", choices=["docling", "pymupdf"], default="docling")
    parser.add_argument("--planner-provider", choices=["ollama"], default="ollama")
    parser.add_argument("--planner-model", default=settings.agentic_chunk_model)
    parser.add_argument("--skip-embed", action="store_true")
    parser.add_argument("--skip-index", action="store_true")
    args = parser.parse_args()

    summary = asyncio.run(
        prepare_corpus(
            papers_dir=args.papers_dir,
            limit=args.limit,
            qa_per_paper=args.qa_per_paper,
            output=args.output,
            plan_output=args.plan_output,
            parser=args.parser,
            planner_provider=args.planner_provider,
            planner_model=args.planner_model,
            skip_embed=args.skip_embed,
            skip_index=args.skip_index,
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


async def prepare_corpus(
    *,
    papers_dir: Path,
    limit: int,
    qa_per_paper: int,
    output: Path,
    plan_output: Path,
    parser: str,
    planner_provider: str,
    planner_model: str,
    skip_embed: bool,
    skip_index: bool,
) -> dict[str, object]:
    await ensure_papers_table(settings.postgres_dsn)
    await ensure_documents_table(settings.postgres_dsn)
    await ensure_document_chunks_table(settings.postgres_dsn)

    pdfs = select_pdf_files(papers_dir, limit)
    arxiv_ids = [path.stem for path in pdfs]
    papers = fetch_metadata_for_ids(arxiv_ids)
    await insert_papers(settings.postgres_dsn, papers)

    qa_items: list[dict[str, object]] = []
    paper_summaries: list[dict[str, object]] = []
    for pdf_path in pdfs:
        arxiv_id = pdf_path.stem
        paper = await fetch_paper_by_arxiv_id(settings.postgres_dsn, arxiv_id)
        if paper is None:
            raise RuntimeError(f"paper metadata missing after insert: {arxiv_id}")

        paper_file = PaperFile(
            paper_id=paper.id,
            arxiv_id=paper.arxiv_id,
            pdf_url=paper.pdf_url,
            file_path=pdf_path,
            file_sha256=sha256_file(pdf_path),
            file_size=pdf_path.stat().st_size,
        )
        await insert_paper_file(settings.postgres_dsn, paper_file)

        document_ids = await fetch_document_ids_by_source(settings.postgres_dsn, paper.source_url)
        block_document_ids = await fetch_document_ids_by_source_prefix(
            settings.postgres_dsn,
            f"{paper.source_url}#block:",
        )
        await delete_chunks_for_documents(settings.postgres_dsn, sorted(set(document_ids + block_document_ids)))
        await delete_documents_by_source_prefix(settings.postgres_dsn, f"{paper.source_url}#block:")
        await delete_paper_blocks(settings.postgres_dsn, paper.id)

        blocks = parse_blocks(parser, paper_file)
        await insert_paper_blocks(settings.postgres_dsn, blocks)
        await upsert_documents_by_source(settings.postgres_dsn, [structured_paper_to_document(paper, blocks)])

        refreshed_blocks = await fetch_paper_blocks_for_paper(settings.postgres_dsn, paper.id)
        documents = await fetch_documents(settings.postgres_dsn)
        paper_documents = [document for document in documents if document.source == paper.source_url]
        if not paper_documents:
            raise RuntimeError(f"document missing for {arxiv_id}")

        fixed_chunks = annotate_fixed_chunk_sections(
            [
                chunk
                for document in paper_documents
                for chunk in chunk_document(document.id, document.content)
            ]
        )
        agentic_result = chunk_agentic_documents(
            paper_documents,
            refreshed_blocks,
            planner_provider=planner_provider,
            planner_model=planner_model,
        )
        agentic_chunks = agentic_result.chunks

        inserted_fixed = await insert_chunks(settings.postgres_dsn, fixed_chunks)
        inserted_agentic = await insert_chunks(settings.postgres_dsn, agentic_chunks)

        qa_items.extend(await build_qa_items(paper.id, arxiv_id, qa_per_paper))
        paper_summaries.append(
            {
                "arxiv_id": arxiv_id,
                "paper_id": paper.id,
                "file_size": paper_file.file_size,
                "blocks": len(blocks),
                "block_types": dict(Counter(block.block_type for block in blocks)),
                "fixed_chunks": len(fixed_chunks),
                "inserted_fixed_chunks": inserted_fixed,
                "agentic_chunks": len(agentic_chunks),
                "inserted_agentic_chunks": inserted_agentic,
                "agentic_chunk_types": dict(Counter(chunk.chunk_type for chunk in agentic_chunks)),
            }
        )
        print(
            f"prepared {arxiv_id}: blocks={len(blocks)} "
            f"fixed={inserted_fixed}/{len(fixed_chunks)} "
            f"agentic={inserted_agentic}/{len(agentic_chunks)}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(qa_items, ensure_ascii=False, indent=2), encoding="utf-8")
    plan_output.parent.mkdir(parents=True, exist_ok=True)
    plan_output.write_text(json.dumps(build_ablation_plan(output), ensure_ascii=False, indent=2), encoding="utf-8")

    embedded = 0 if skip_embed else await embed_chunks()
    indexed = 0 if skip_index else await index_embeddings(recreate=True)
    counts = await count_prepared_rows()
    return {
        "papers": len(pdfs),
        "qa_items": len(qa_items),
        "qa_path": str(output),
        "plan_path": str(plan_output),
        "embedded_new_chunks": embedded,
        "indexed_chunks": indexed,
        "counts": counts,
        "paper_summaries": paper_summaries,
    }


def select_pdf_files(papers_dir: Path, limit: int) -> list[Path]:
    pdfs = sorted((path for path in papers_dir.glob("*.pdf") if path.is_file()), key=lambda path: path.stat().st_size)
    if len(pdfs) < limit:
        raise RuntimeError(f"only found {len(pdfs)} PDFs in {papers_dir}, need {limit}")
    return pdfs[:limit]


def fetch_metadata_for_ids(arxiv_ids: list[str]) -> list[PaperMetadata]:
    params = urlencode({"id_list": ",".join(arxiv_ids)})
    response = requests.get(f"{ARXIV_API_URL}?{params}", timeout=60)
    response.raise_for_status()
    papers_by_id = {paper.arxiv_id: paper for paper in parse_arxiv_feed(response.text)}
    missing = [arxiv_id for arxiv_id in arxiv_ids if arxiv_id not in papers_by_id]
    if missing:
        raise RuntimeError(f"arXiv metadata missing for: {', '.join(missing)}")
    return [papers_by_id[arxiv_id] for arxiv_id in arxiv_ids]


def parse_blocks(parser: str, paper_file: PaperFile):
    if parser == "pymupdf":
        return parse_pdf_to_blocks(paper_file)
    try:
        return parse_pdf_with_docling(paper_file)
    except Exception as exc:
        print(f"Docling parser failed for {paper_file.arxiv_id}; falling back to PyMuPDF: {exc}")
        return parse_pdf_to_blocks(paper_file)


async def build_qa_items(paper_id: int, arxiv_id: str, qa_per_paper: int) -> list[dict[str, object]]:
    import asyncpg

    conn = await asyncpg.connect(settings.postgres_dsn)
    try:
        rows = await conn.fetch(
            """
            SELECT
                c.id AS source_chunk_id,
                c.document_id,
                c.content,
                c.chunk_type,
                c.section_name,
                c.source_block_ids,
                c.metadata,
                c.chunking_strategy,
                p.id AS paper_id,
                p.arxiv_id
            FROM document_chunks c
            JOIN documents d ON d.id = c.document_id
            JOIN papers p ON p.source_url = d.source
            WHERE p.id = $1
              AND c.chunking_strategy IN ('agentic', 'agentic_fusion')
              AND c.section_name IS NOT NULL
            ORDER BY
                CASE
                    WHEN c.chunk_type IN ('table', 'figure_caption', 'fused') THEN 0
                    ELSE 1
                END,
                c.chunk_index
            LIMIT $2
            """,
            paper_id,
            qa_per_paper,
        )
    finally:
        await conn.close()

    items = []
    for offset, row in enumerate(rows, start=1):
        item = qa_from_chunk(dict(row), offset)
        item["id"] = f"ablation-{arxiv_id}-{offset:02d}"
        item["paper_id"] = paper_id
        item["arxiv_id"] = arxiv_id
        item["gold_chunk_ids"] = [int(row["source_chunk_id"])]
        item["gold_document_ids"] = [int(row["document_id"])]
        item["category"] = "figure_table" if str(row["chunk_type"]) in {"table", "figure_caption", "fused"} else "method"
        item["tags"] = sorted(set(item.get("tags", []) + ["ablation_10paper"]))
        item["notes"] = "Generated from agentic/fused chunks for fixed-vs-agentic and visual/fused ablation."
        items.append(item)
    return items


def build_ablation_plan(dataset_path: Path) -> dict[str, object]:
    return {
        "dataset": str(dataset_path),
        "experiments": [
            {
                "id": "A_chunk_strategy_ablation",
                "subset_filter": {"tags": ["ablation_10paper"]},
                "variants": [
                    {"name": "fixed_hybrid", "strategy": "hybrid", "chunk_strategy": "fixed"},
                    {"name": "agentic_hybrid", "strategy": "hybrid", "chunk_strategy": "agentic"},
                    {"name": "fixed_hybrid_hyde_rerank", "strategy": "hybrid_hyde_rerank", "chunk_strategy": "fixed"},
                    {"name": "agentic_hybrid_hyde_rerank", "strategy": "hybrid_hyde_rerank", "chunk_strategy": "agentic"},
                ],
            },
            {
                "id": "B_visual_fused_ablation",
                "subset_filter": {"categories": ["figure_table"]},
                "variants": [
                    {"name": "fixed_visual_subset", "strategy": "hybrid", "chunk_strategy": "fixed"},
                    {"name": "agentic_visual_subset", "strategy": "hybrid", "chunk_strategy": "agentic"},
                    {"name": "agentic_visual_rerank", "strategy": "hybrid_hyde_rerank", "chunk_strategy": "agentic"},
                ],
            },
        ],
    }


async def count_prepared_rows() -> dict[str, object]:
    import asyncpg

    conn = await asyncpg.connect(settings.postgres_dsn)
    try:
        counts = {}
        for table in ["papers", "paper_files", "paper_blocks", "documents", "document_chunks", "chunk_embeddings"]:
            counts[table] = await conn.fetchval(f"SELECT COUNT(*) FROM {table}")
        rows = await conn.fetch(
            """
            SELECT chunking_strategy, chunk_type, COUNT(*) AS count
            FROM document_chunks
            GROUP BY chunking_strategy, chunk_type
            ORDER BY chunking_strategy, chunk_type
            """
        )
        counts["chunks_by_strategy_type"] = [
            {
                "chunking_strategy": row["chunking_strategy"],
                "chunk_type": row["chunk_type"],
                "count": row["count"],
            }
            for row in rows
        ]
        return counts
    finally:
        await conn.close()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
