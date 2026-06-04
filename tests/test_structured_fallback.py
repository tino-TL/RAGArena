from __future__ import annotations

import pytest

from pathlib import Path

from ragarena.chunking.repository import DocumentRecord
from ragarena.papers.models import PaperFile, StoredPaper
from ragarena.pipeline import steps as pipeline_steps


@pytest.mark.anyio
async def test_structured_parser_falls_back_to_simple(monkeypatch) -> None:
    async def noop(*args, **kwargs):
        return None

    async def fake_fetch_paper_files(dsn, *, limit):
        return [
            PaperFile(
                paper_id=1,
                arxiv_id="2401.00001v1",
                pdf_url="https://arxiv.org/pdf/2401.00001v1",
                file_path=Path("paper.pdf"),
                file_sha256="sha",
                file_size=123,
            )
        ]

    async def fake_fetch_paper_by_arxiv_id(dsn, arxiv_id):
        return StoredPaper(
            id=1,
            arxiv_id=arxiv_id,
            title="Paper",
            authors=[],
            abstract="Abstract",
            categories=[],
            published_at=None,
            updated_at=None,
            pdf_url="https://arxiv.org/pdf/2401.00001v1",
            source_url="https://arxiv.org/abs/2401.00001v1",
        )

    monkeypatch.setattr(pipeline_steps, "ensure_papers_table", noop)
    monkeypatch.setattr(pipeline_steps, "fetch_paper_files", fake_fetch_paper_files)
    monkeypatch.setattr(pipeline_steps, "fetch_paper_by_arxiv_id", fake_fetch_paper_by_arxiv_id)
    monkeypatch.setattr(pipeline_steps, "parse_pdf_to_blocks", lambda paper_file: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(pipeline_steps, "parse_pdf_to_text_blocks", lambda paper_file: ["block"])
    monkeypatch.setattr(pipeline_steps, "delete_paper_blocks", noop)

    async def fake_insert_paper_blocks(dsn, blocks):
        return 1

    monkeypatch.setattr(pipeline_steps, "insert_paper_blocks", fake_insert_paper_blocks)

    count = await pipeline_steps.parse_papers(limit=1, sync_documents=False, parser="pymupdf")

    assert count == 1


@pytest.mark.anyio
async def test_docling_failure_falls_back_to_pymupdf(monkeypatch) -> None:
    async def noop(*args, **kwargs):
        return None

    async def fake_fetch_paper_files(dsn, *, limit):
        return [
            PaperFile(
                paper_id=1,
                arxiv_id="2401.00001v1",
                pdf_url="https://arxiv.org/pdf/2401.00001v1",
                file_path=Path("paper.pdf"),
                file_sha256="sha",
                file_size=123,
            )
        ]

    async def fake_fetch_paper_by_arxiv_id(dsn, arxiv_id):
        return StoredPaper(
            id=1,
            arxiv_id=arxiv_id,
            title="Paper",
            authors=[],
            abstract="Abstract",
            categories=[],
            published_at=None,
            updated_at=None,
            pdf_url="https://arxiv.org/pdf/2401.00001v1",
            source_url="https://arxiv.org/abs/2401.00001v1",
        )

    monkeypatch.setattr(pipeline_steps, "ensure_papers_table", noop)
    monkeypatch.setattr(pipeline_steps, "fetch_paper_files", fake_fetch_paper_files)
    monkeypatch.setattr(pipeline_steps, "fetch_paper_by_arxiv_id", fake_fetch_paper_by_arxiv_id)
    monkeypatch.setattr(pipeline_steps, "parse_pdf_with_docling", lambda paper_file: (_ for _ in ()).throw(RuntimeError("docling boom")))
    monkeypatch.setattr(pipeline_steps, "parse_pdf_to_blocks", lambda paper_file: [])
    monkeypatch.setattr(pipeline_steps, "delete_paper_blocks", noop)

    async def fake_insert_paper_blocks(dsn, blocks):
        return 0

    monkeypatch.setattr(pipeline_steps, "insert_paper_blocks", fake_insert_paper_blocks)

    count = await pipeline_steps.parse_papers(limit=1, sync_documents=False, parser="docling")

    assert count == 0


@pytest.mark.anyio
async def test_block_chunking_falls_back_to_fixed(monkeypatch) -> None:
    async def noop(*args, **kwargs):
        return None

    async def fake_fetch_documents(dsn):
        return [
            DocumentRecord(
                id=1,
                title="Doc",
                source="source",
                content="fixed chunk fallback content",
                content_hash="hash",
            )
        ]

    async def fake_fetch_paper_blocks(dsn, limit):
        return []

    async def fake_delete_chunks_for_documents(dsn, document_ids):
        return 0

    monkeypatch.setattr(pipeline_steps, "ensure_documents_table", noop)
    monkeypatch.setattr(pipeline_steps, "ensure_document_chunks_table", noop)
    monkeypatch.setattr(pipeline_steps, "fetch_documents", fake_fetch_documents)
    monkeypatch.setattr(pipeline_steps, "fetch_paper_blocks", fake_fetch_paper_blocks)
    monkeypatch.setattr(pipeline_steps, "chunk_block_documents", lambda documents, blocks: (_ for _ in ()).throw(RuntimeError("boom")))

    inserted = {}

    async def fake_insert_chunks(dsn, chunks):
        inserted["chunks"] = chunks
        return len(chunks)

    monkeypatch.setattr(pipeline_steps, "delete_chunks_for_documents", fake_delete_chunks_for_documents)
    monkeypatch.setattr(pipeline_steps, "insert_chunks", fake_insert_chunks)

    count = await pipeline_steps.chunk_documents(chunk_strategy="block")

    assert count == 1
    assert inserted["chunks"][0].chunking_strategy == "fixed"
