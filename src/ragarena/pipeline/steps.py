from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from ragarena.chunking.agentic_chunker import (
    cleanup_retrieval_chunks,
    chunk_agentic_documents,
    optimize_retrieval_units,
    print_chunk_quality_report,
)
from ragarena.chunking.boundary_validator import validate_chunk_boundaries
from ragarena.chunking.block_chunker import chunk_block_documents
from ragarena.chunking.fixed_chunker import Chunk, chunk_document
from ragarena.chunking.repository import (
    delete_chunks_for_documents,
    ensure_document_chunks_table,
    fetch_documents,
    insert_chunks,
)
from ragarena.config import settings
from ragarena.embedding.repository import (
    ensure_chunk_embeddings_table,
    fetch_chunks_without_embeddings,
    insert_embeddings,
)
from ragarena.ingestion.repository import (
    delete_documents_by_source_prefix,
    ensure_documents_table,
    fetch_document_ids_by_source,
    fetch_document_ids_by_source_prefix,
    insert_documents,
    insert_documents_missing_sources,
    upsert_documents_by_source,
)
from ragarena.papers.arxiv_client import fetch_arxiv_papers
from ragarena.papers.docling_parser import parse_pdf_with_docling
from ragarena.papers.downloader import download_paper_pdf
from ragarena.papers.models import PaperMetadata, paper_to_document, structured_paper_to_document
from ragarena.papers.repository import (
    delete_paper_blocks,
    ensure_papers_table,
    fetch_paper_by_arxiv_id,
    fetch_paper_files,
    fetch_papers,
    fetch_paper_blocks,
    insert_paper_blocks,
    insert_paper_file,
    insert_papers,
)
from ragarena.papers.structured_parser import parse_pdf_to_blocks
from ragarena.papers.text_parser import parse_pdf_to_text_blocks
from ragarena.retrieval.indexer import build_indexing_result, fetch_embedded_chunks, index_embedded_chunks
from ragarena.runtime import get_bge_encoder, get_elasticsearch_vector_store

logger = logging.getLogger(__name__)


async def fetch_and_store_papers(
    *,
    query: str,
    max_results: int,
    sync_documents: bool = True,
) -> int:
    papers = fetch_arxiv_papers(query, max_results=max_results)
    return await store_papers(papers, sync_documents=sync_documents)


async def store_papers(papers: list[PaperMetadata], *, sync_documents: bool = True) -> int:
    await ensure_papers_table(settings.postgres_dsn)
    stored_count = await insert_papers(settings.postgres_dsn, papers)

    print(f"Fetched papers: {len(papers)}")
    print(f"Stored papers: {stored_count}")

    if sync_documents:
        documents = [paper_to_document(paper) for paper in papers]
        await ensure_documents_table(settings.postgres_dsn)
        inserted_documents = await insert_documents(settings.postgres_dsn, documents)
        print(f"Synced paper abstracts to documents: {inserted_documents}")

    return stored_count


async def download_papers(
    *,
    limit: int,
    output_dir: Path,
    arxiv_id: str | None = None,
) -> int:
    await ensure_papers_table(settings.postgres_dsn)
    if arxiv_id:
        paper = await fetch_paper_by_arxiv_id(settings.postgres_dsn, arxiv_id)
        papers = [paper] if paper else []
    else:
        papers = await fetch_papers(
            settings.postgres_dsn,
            limit=limit,
            missing_files_only=True,
        )

    downloaded_count = 0
    for paper in papers:
        paper_file = download_paper_pdf(paper, output_dir=output_dir)
        downloaded_count += await insert_paper_file(settings.postgres_dsn, paper_file)
        print(f"Downloaded {paper.arxiv_id}: {paper_file.file_path}")

    print(f"Downloaded papers: {downloaded_count}")
    return downloaded_count


async def parse_papers(
    *,
    limit: int,
    arxiv_id: str | None = None,
    sync_documents: bool = True,
    parser: str = "docling",
    reset: bool = False,
) -> int:
    await ensure_papers_table(settings.postgres_dsn)
    paper_files = await fetch_paper_files(settings.postgres_dsn, limit=limit)
    if arxiv_id:
        paper_files = [paper_file for paper_file in paper_files if paper_file.arxiv_id == arxiv_id]

    inserted_blocks_total = 0
    synced_documents = 0
    structured_papers = 0
    structured_block_counts: list[int] = []
    structured_type_counts: Counter[str] = Counter()
    for paper_file in paper_files:
        paper = await fetch_paper_by_arxiv_id(settings.postgres_dsn, paper_file.arxiv_id)
        if paper is None:
            continue

        parser_backend = normalize_parser_backend(parser)
        if parser_backend in {"docling", "pymupdf"}:
            try:
                if parser_backend == "docling":
                    try:
                        blocks = parse_pdf_with_docling(paper_file)
                    except Exception as exc:
                        print(f"Docling parser failed for {paper_file.arxiv_id}; falling back to PyMuPDF parser: {exc}")
                        blocks = parse_pdf_to_blocks(paper_file)
                else:
                    blocks = parse_pdf_to_blocks(paper_file)
                if sync_documents:
                    if reset:
                        await ensure_document_chunks_table(settings.postgres_dsn)
                        document_ids = await fetch_document_ids_by_source(settings.postgres_dsn, paper.source_url)
                        block_document_ids = await fetch_document_ids_by_source_prefix(
                            settings.postgres_dsn,
                            f"{paper.source_url}#block:",
                        )
                        await delete_chunks_for_documents(
                            settings.postgres_dsn,
                            sorted(set(document_ids + block_document_ids)),
                        )
                    await delete_documents_by_source_prefix(settings.postgres_dsn, f"{paper.source_url}#block:")
                await delete_paper_blocks(settings.postgres_dsn, paper.id)
                inserted_blocks = await insert_paper_blocks(settings.postgres_dsn, blocks)
                print_structured_parser_stats(paper_file.arxiv_id, blocks)
                structured_papers += 1
                structured_block_counts.append(len(blocks))
                structured_type_counts.update(block.block_type for block in blocks)
                if sync_documents:
                    documents = [structured_paper_to_document(paper, blocks)]
                    await ensure_documents_table(settings.postgres_dsn)
                    if reset:
                        synced_documents += await upsert_documents_by_source(settings.postgres_dsn, documents)
                    else:
                        synced_documents += await insert_documents_missing_sources(settings.postgres_dsn, documents)
                inserted_blocks_total += inserted_blocks
                continue
            except Exception as exc:
                print(f"{parser_backend} parser failed for {paper_file.arxiv_id}; falling back to simple parser: {exc}")

        if sync_documents:
            if reset:
                await ensure_document_chunks_table(settings.postgres_dsn)
                document_ids = await fetch_document_ids_by_source(settings.postgres_dsn, paper.source_url)
                block_document_ids = await fetch_document_ids_by_source_prefix(
                    settings.postgres_dsn,
                    f"{paper.source_url}#block:",
                )
                await delete_chunks_for_documents(
                    settings.postgres_dsn,
                    sorted(set(document_ids + block_document_ids)),
                )
            await delete_documents_by_source_prefix(settings.postgres_dsn, f"{paper.source_url}#block:")
        await delete_paper_blocks(settings.postgres_dsn, paper.id)
        blocks = parse_pdf_to_text_blocks(paper_file)
        inserted_blocks_total += await insert_paper_blocks(settings.postgres_dsn, blocks)
        print(f"Parsed {paper_file.arxiv_id}: {len(blocks)} text blocks")

        if sync_documents:
            documents = [structured_paper_to_document(paper, blocks)]
            await ensure_documents_table(settings.postgres_dsn)
            if reset:
                synced_documents += await upsert_documents_by_source(settings.postgres_dsn, documents)
            else:
                synced_documents += await insert_documents_missing_sources(settings.postgres_dsn, documents)

    print(f"Inserted paper blocks: {inserted_blocks_total}")
    if structured_papers:
        print(
            "Parser stats: "
            f"papers={structured_papers} "
            f"blocks_per_paper={structured_block_counts} "
            f"block_type_counts={dict(structured_type_counts)}"
        )
    if sync_documents:
        print(f"Synced full-text papers to documents: {synced_documents}")
    return inserted_blocks_total


def print_structured_parser_stats(arxiv_id: str, blocks: Sequence[object]) -> None:
    block_types = [getattr(block, "block_type", "unknown") for block in blocks]
    counts = Counter(block_types)
    print(f"Structured parsed {arxiv_id}: {len(blocks)} blocks")
    print(f"Parser stats: papers=1 blocks_per_paper={len(blocks)} block_type_counts={dict(counts)}")
    if len(blocks) > 200:
        print(f"WARNING: {arxiv_id} produced {len(blocks)} structured blocks; expected <= 200 for most papers")


def normalize_parser_backend(parser: str) -> str:
    return "pymupdf" if parser == "structured" else parser


def avg_chunk_tokens(chunks: list[Chunk]) -> float:
    if not chunks:
        return 0.0
    return round(sum(chunk.token_count for chunk in chunks) / len(chunks), 2)


async def chunk_documents(
    *,
    chunk_strategy: str = "agentic",
    debug_planner: bool = False,
    planner_provider: str | None = None,
    planner_model: str | None = None,
    validate_chunks: bool = False,
) -> int:
    await ensure_documents_table(settings.postgres_dsn)
    await ensure_document_chunks_table(settings.postgres_dsn)

    documents = await fetch_documents(settings.postgres_dsn)
    try:
        effective_strategy = chunk_strategy
        if chunk_strategy in {"block", "agentic"}:
            blocks = await fetch_paper_blocks(settings.postgres_dsn, limit=100000)
            if chunk_strategy == "agentic":
                result = chunk_agentic_documents(
                    documents,
                    blocks,
                    debug_planner=debug_planner,
                    planner_provider=planner_provider,
                    planner_model=planner_model,
                )
                chunks = result.chunks
                for trace in result.traces:
                    print(
                        "Agentic chunk trace: "
                        f"section_name={trace.section_name} "
                        f"input_block_count={trace.input_block_count} "
                        f"generated_chunk_count={trace.generated_chunk_count} "
                        f"total_duration={trace.total_duration} "
                        f"load_duration={trace.load_duration} "
                        f"eval_duration={trace.eval_duration} "
                        f"load_warning={trace.load_warning} "
                        f"fallback_reason={trace.fallback_reason}"
                    )
                print(
                    "Agentic planner stats: "
                    f"before_merge_chunks={result.stats.get('before_merge_chunks', 0)} "
                    f"after_merge_chunks={result.stats.get('after_merge_chunks', 0)} "
                    f"skipped={result.stats.get('skipped', 0)} "
                    f"avg_chunk_tokens={result.stats.get('avg_chunk_tokens', 0)} "
                    f"dropped_tiny_chunks={result.stats.get('dropped_tiny_chunks', 0)} "
                    f"merged_tiny_chunks={result.stats.get('merged_tiny_chunks', 0)} "
                    f"final_avg_chunk_tokens={result.stats.get('final_avg_chunk_tokens', 0)} "
                    f"min_chunk_tokens={result.stats.get('min_chunk_tokens', 0)}"
                )
            else:
                cleanup = cleanup_retrieval_chunks(chunk_block_documents(documents, blocks))
                chunks = cleanup.chunks
                print(
                    "Chunk cleanup stats: "
                    f"dropped_tiny_chunks={cleanup.dropped_tiny_chunks} "
                    f"merged_tiny_chunks={cleanup.merged_tiny_chunks} "
                    f"final_avg_chunk_tokens={avg_chunk_tokens(chunks)} "
                    f"min_chunk_tokens={min((chunk.token_count for chunk in chunks), default=0)}"
                )
            if validate_chunks:
                validation = validate_chunk_boundaries(chunks, blocks)
                chunks = optimize_retrieval_units(validation.chunks, blocks)
                print(
                    "Chunk boundary validation stats: "
                    f"boundary_issues_found={validation.stats.boundary_issues_found} "
                    f"rule_fixes={validation.stats.rule_fixes} "
                    f"model_fixes={validation.stats.model_fixes} "
                    f"dropped_chunks={validation.stats.dropped_chunks}"
                )
            if not chunks:
                raise ValueError(f"no {chunk_strategy} chunks produced")
            print_chunk_quality_report(chunks)
        else:
            chunks = []
    except Exception as exc:
        logger.warning("block_chunking_fallback", extra={"error": str(exc)})
        chunks = []

    if not chunks:
        chunks = [
            chunk
            for document in documents
            for chunk in chunk_document(document.id, document.content)
        ]
        effective_strategy = "fixed"
    deleted_count = await delete_chunks_for_documents(
        settings.postgres_dsn,
        sorted({chunk.document_id for chunk in chunks}),
    )
    inserted_count = await insert_chunks(settings.postgres_dsn, chunks)

    print(f"Loaded documents: {len(documents)}")
    print(f"Generated chunks: {len(chunks)}")
    print(f"Deleted existing chunks: {deleted_count}")
    print(f"Inserted chunks: {inserted_count}")
    print(f"Skipped duplicates: {len(chunks) - inserted_count}")
    print(f"Chunking strategy: {effective_strategy}")

    return inserted_count


async def embed_chunks() -> int:
    await ensure_documents_table(settings.postgres_dsn)
    await ensure_document_chunks_table(settings.postgres_dsn)
    await ensure_chunk_embeddings_table(settings.postgres_dsn)

    chunks = await fetch_chunks_without_embeddings(
        settings.postgres_dsn,
        settings.embedding_model,
    )
    if not chunks:
        print("Loaded chunks: 0")
        print("Inserted embeddings: 0")
        print("Skipped existing embeddings: all chunks already embedded")
        return 0

    encoder = get_bge_encoder(settings.embedding_model)
    print(f"before encode: chunks={len(chunks)} model={settings.embedding_model}")
    embeddings = encoder.encode([chunk.content for chunk in chunks])
    print(f"after encode: embeddings={len(embeddings)} model={settings.embedding_model}")
    print(f"before insert embeddings: chunks={len(chunks)} embeddings={len(embeddings)}")
    inserted_count = await insert_embeddings(
        settings.postgres_dsn,
        settings.embedding_model,
        [chunk.id for chunk in chunks],
        embeddings,
    )
    print(f"after insert embeddings: inserted={inserted_count}")

    print(f"Loaded chunks: {len(chunks)}")
    print(f"Inserted embeddings: {inserted_count}")
    print(f"Skipped existing embeddings: {len(chunks) - inserted_count}")

    return inserted_count


async def index_embeddings(*, recreate: bool = True) -> int:
    vector_store = get_elasticsearch_vector_store(
        settings.elasticsearch_url,
        settings.elasticsearch_index,
    )
    embedded_chunks = await fetch_embedded_chunks(
        settings.postgres_dsn,
        settings.embedding_model,
    )
    deleted_existing_index = False
    if recreate:
        deleted_existing_index = vector_store.recreate_index()
        indexed_count = index_embedded_chunks(vector_store, embedded_chunks, recreate=False)
    else:
        indexed_count = index_embedded_chunks(vector_store, embedded_chunks, recreate=False)
    final_es_count = vector_store.count(settings.embedding_model)
    result = build_indexing_result(
        deleted_existing_index=deleted_existing_index,
        loaded_embeddings=len(embedded_chunks),
        indexed_chunks=indexed_count,
        final_es_count=final_es_count,
        postgres_embedding_count=len(embedded_chunks),
    )

    print(f"deleted_existing_index: {str(result.deleted_existing_index).lower()}")
    print(f"loaded_embeddings: {result.loaded_embeddings}")
    print(f"indexed_chunks: {result.indexed_chunks}")
    print(f"final_es_count: {result.final_es_count}")
    print(f"postgres_embedding_count: {result.postgres_embedding_count}")
    print(f"Elasticsearch index: {settings.elasticsearch_index}")
    if not result.count_matches:
        print(
            "warning: final_es_count does not match postgres_embedding_count "
            f"({result.final_es_count} != {result.postgres_embedding_count})"
        )

    return indexed_count
