from __future__ import annotations

from pathlib import Path

import asyncpg

from ragarena.papers.models import PaperBlock, PaperFile, PaperMetadata, StoredPaper
from ragarena.utils.text import sanitize_text

CREATE_PAPERS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS papers (
    id BIGSERIAL PRIMARY KEY,
    arxiv_id TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    authors JSONB NOT NULL,
    abstract TEXT NOT NULL,
    categories JSONB NOT NULL,
    published_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ,
    pdf_url TEXT NOT NULL,
    source_url TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

CREATE_PAPER_FILES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS paper_files (
    id BIGSERIAL PRIMARY KEY,
    paper_id BIGINT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    file_path TEXT NOT NULL,
    file_sha256 TEXT NOT NULL,
    file_size BIGINT NOT NULL,
    downloaded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (paper_id, file_sha256)
);
"""

CREATE_PAPER_BLOCKS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS paper_blocks (
    id BIGSERIAL PRIMARY KEY,
    paper_id BIGINT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    arxiv_id TEXT NOT NULL,
    block_type TEXT NOT NULL,
    section_name TEXT,
    page_number INTEGER,
    content TEXT NOT NULL,
    markdown_content TEXT,
    image_path TEXT,
    order_index INTEGER NOT NULL,
    should_embed BOOLEAN NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    content_hash TEXT NOT NULL UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""


async def ensure_papers_table(dsn: str) -> None:
    conn = await asyncpg.connect(dsn)
    try:
        await conn.execute(CREATE_PAPERS_TABLE_SQL)
        await conn.execute(CREATE_PAPER_FILES_TABLE_SQL)
        await conn.execute(CREATE_PAPER_BLOCKS_TABLE_SQL)
    finally:
        await conn.close()


async def insert_papers(dsn: str, papers: list[PaperMetadata]) -> int:
    if not papers:
        return 0

    conn = await asyncpg.connect(dsn)
    try:
        inserted_count = 0
        for paper in papers:
            result = await conn.execute(
                """
                INSERT INTO papers (
                    arxiv_id,
                    title,
                    authors,
                    abstract,
                    categories,
                    published_at,
                    updated_at,
                    pdf_url,
                    source_url
                )
                VALUES ($1, $2, $3::jsonb, $4, $5::jsonb, $6, $7, $8, $9)
                ON CONFLICT (arxiv_id) DO UPDATE SET
                    title = EXCLUDED.title,
                    authors = EXCLUDED.authors,
                    abstract = EXCLUDED.abstract,
                    categories = EXCLUDED.categories,
                    published_at = EXCLUDED.published_at,
                    updated_at = EXCLUDED.updated_at,
                    pdf_url = EXCLUDED.pdf_url,
                    source_url = EXCLUDED.source_url
                """,
                paper.arxiv_id,
                paper.title,
                _json_array(paper.authors),
                paper.abstract,
                _json_array(paper.categories),
                paper.published_at,
                paper.updated_at,
                paper.pdf_url,
                paper.source_url,
            )
            inserted_count += _inserted_count(result)

        return inserted_count
    finally:
        await conn.close()


async def count_papers_by_arxiv_ids(dsn: str, arxiv_ids: list[str]) -> int:
    if not arxiv_ids:
        return 0

    conn = await asyncpg.connect(dsn)
    try:
        return await conn.fetchval(
            "SELECT COUNT(*) FROM papers WHERE arxiv_id = ANY($1::text[])",
            arxiv_ids,
        )
    finally:
        await conn.close()


async def fetch_papers(
    dsn: str,
    *,
    limit: int = 20,
    missing_files_only: bool = False,
) -> list[StoredPaper]:
    conn = await asyncpg.connect(dsn)
    try:
        where_clause = ""
        if missing_files_only:
            where_clause = """
            WHERE NOT EXISTS (
                SELECT 1 FROM paper_files pf WHERE pf.paper_id = p.id
            )
            """
        rows = await conn.fetch(
            f"""
            SELECT
                p.id,
                p.arxiv_id,
                p.title,
                p.authors,
                p.abstract,
                p.categories,
                p.published_at,
                p.updated_at,
                p.pdf_url,
                p.source_url
            FROM papers p
            {where_clause}
            ORDER BY p.published_at DESC NULLS LAST, p.id DESC
            LIMIT $1
            """,
            limit,
        )
        return [_row_to_paper(row) for row in rows]
    finally:
        await conn.close()


async def fetch_paper_by_arxiv_id(dsn: str, arxiv_id: str) -> StoredPaper | None:
    conn = await asyncpg.connect(dsn)
    try:
        row = await conn.fetchrow(
            """
            SELECT
                id,
                arxiv_id,
                title,
                authors,
                abstract,
                categories,
                published_at,
                updated_at,
                pdf_url,
                source_url
            FROM papers
            WHERE arxiv_id = $1
            """,
            arxiv_id,
        )
        return _row_to_paper(row) if row else None
    finally:
        await conn.close()


async def fetch_paper_by_id(dsn: str, paper_id: int) -> StoredPaper | None:
    conn = await asyncpg.connect(dsn)
    try:
        row = await conn.fetchrow(
            """
            SELECT
                id,
                arxiv_id,
                title,
                authors,
                abstract,
                categories,
                published_at,
                updated_at,
                pdf_url,
                source_url
            FROM papers
            WHERE id = $1
            """,
            paper_id,
        )
        return _row_to_paper(row) if row else None
    finally:
        await conn.close()


async def insert_paper_file(dsn: str, paper_file: PaperFile) -> int:
    conn = await asyncpg.connect(dsn)
    try:
        result = await conn.execute(
            """
            INSERT INTO paper_files (paper_id, file_path, file_sha256, file_size)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (paper_id, file_sha256) DO NOTHING
            """,
            paper_file.paper_id,
            str(paper_file.file_path),
            paper_file.file_sha256,
            paper_file.file_size,
        )
        return _inserted_count(result)
    finally:
        await conn.close()


async def fetch_paper_files(dsn: str, *, limit: int = 20) -> list[PaperFile]:
    conn = await asyncpg.connect(dsn)
    try:
        rows = await conn.fetch(
            """
            SELECT
                p.id AS paper_id,
                p.arxiv_id,
                p.pdf_url,
                pf.file_path,
                pf.file_sha256,
                pf.file_size
            FROM paper_files pf
            JOIN papers p ON p.id = pf.paper_id
            ORDER BY pf.downloaded_at DESC
            LIMIT $1
            """,
            limit,
        )
        return [
            PaperFile(
                paper_id=row["paper_id"],
                arxiv_id=row["arxiv_id"],
                pdf_url=row["pdf_url"],
                file_path=Path(row["file_path"]),
                file_sha256=row["file_sha256"],
                file_size=row["file_size"],
            )
            for row in rows
        ]
    finally:
        await conn.close()


async def fetch_paper_file_for_paper(dsn: str, paper_id: int) -> PaperFile | None:
    conn = await asyncpg.connect(dsn)
    try:
        row = await conn.fetchrow(
            """
            SELECT
                p.id AS paper_id,
                p.arxiv_id,
                p.pdf_url,
                pf.file_path,
                pf.file_sha256,
                pf.file_size
            FROM paper_files pf
            JOIN papers p ON p.id = pf.paper_id
            WHERE p.id = $1
            ORDER BY pf.downloaded_at DESC
            LIMIT 1
            """,
            paper_id,
        )
        if row is None:
            return None
        return PaperFile(
            paper_id=row["paper_id"],
            arxiv_id=row["arxiv_id"],
            pdf_url=row["pdf_url"],
            file_path=Path(row["file_path"]),
            file_sha256=row["file_sha256"],
            file_size=row["file_size"],
        )
    finally:
        await conn.close()


async def insert_paper_blocks(dsn: str, blocks: list[PaperBlock]) -> int:
    if not blocks:
        return 0

    conn = await asyncpg.connect(dsn)
    try:
        inserted_count = 0
        for block in blocks:
            result = await conn.execute(
                """
                INSERT INTO paper_blocks (
                    paper_id,
                    arxiv_id,
                    block_type,
                    section_name,
                    page_number,
                    content,
                    markdown_content,
                    image_path,
                    order_index,
                    should_embed,
                    metadata,
                    content_hash
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11::jsonb, $12)
                ON CONFLICT (content_hash) DO NOTHING
                """,
                block.paper_id,
                sanitize_text(block.arxiv_id),
                sanitize_text(block.block_type),
                sanitize_text(block.section_name) if block.section_name else None,
                block.page_number,
                sanitize_text(block.content),
                sanitize_text(block.markdown_content) if block.markdown_content else None,
                sanitize_text(block.image_path) if block.image_path else None,
                block.order_index,
                block.should_embed,
                _json_object(block.metadata),
                block.content_hash,
            )
            inserted_count += _inserted_count(result)
        return inserted_count
    finally:
        await conn.close()


async def delete_paper_blocks(dsn: str, paper_id: int) -> int:
    conn = await asyncpg.connect(dsn)
    try:
        result = await conn.execute("DELETE FROM paper_blocks WHERE paper_id = $1", paper_id)
        return _inserted_count(result)
    finally:
        await conn.close()


async def fetch_paper_blocks(
    dsn: str,
    *,
    limit: int = 1000,
    should_embed_only: bool = False,
) -> list[PaperBlock]:
    conn = await asyncpg.connect(dsn)
    try:
        where_clause = "WHERE should_embed IS TRUE" if should_embed_only else ""
        rows = await conn.fetch(
            f"""
            SELECT
                id,
                paper_id,
                arxiv_id,
                block_type,
                section_name,
                page_number,
                content,
                markdown_content,
                image_path,
                order_index,
                should_embed,
                metadata,
                content_hash
            FROM paper_blocks
            {where_clause}
            ORDER BY paper_id, order_index
            LIMIT $1
            """,
            limit,
        )
        return [_row_to_block(row) for row in rows]
    finally:
        await conn.close()


async def fetch_paper_blocks_for_paper(dsn: str, paper_id: int) -> list[PaperBlock]:
    conn = await asyncpg.connect(dsn)
    try:
        rows = await conn.fetch(
            """
            SELECT
                id,
                paper_id,
                arxiv_id,
                block_type,
                section_name,
                page_number,
                content,
                markdown_content,
                image_path,
                order_index,
                should_embed,
                metadata,
                content_hash
            FROM paper_blocks
            WHERE paper_id = $1
            ORDER BY order_index
            """,
            paper_id,
        )
        return [_row_to_block(row) for row in rows]
    finally:
        await conn.close()


async def fetch_paper_blocks_by_hashes(dsn: str, hashes: list[str]) -> list[PaperBlock]:
    if not hashes:
        return []

    conn = await asyncpg.connect(dsn)
    try:
        rows = await conn.fetch(
            """
            SELECT
                id,
                paper_id,
                arxiv_id,
                block_type,
                section_name,
                page_number,
                content,
                markdown_content,
                image_path,
                order_index,
                should_embed,
                metadata,
                content_hash
            FROM paper_blocks
            WHERE content_hash = ANY($1::text[])
            ORDER BY paper_id, order_index
            """,
            hashes,
        )
        return [_row_to_block(row) for row in rows]
    finally:
        await conn.close()


def _json_array(values: list[str]) -> str:
    import json

    return json.dumps(values, ensure_ascii=False)


def _json_object(value: dict[str, object]) -> str:
    import json

    return json.dumps(value, ensure_ascii=False)


def _load_json_array(value: object) -> list[str]:
    import json

    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        return [str(item) for item in json.loads(value)]
    return []


def _row_to_paper(row: asyncpg.Record) -> StoredPaper:
    return StoredPaper(
        id=row["id"],
        arxiv_id=row["arxiv_id"],
        title=row["title"],
        authors=_load_json_array(row["authors"]),
        abstract=row["abstract"],
        categories=_load_json_array(row["categories"]),
        published_at=row["published_at"],
        updated_at=row["updated_at"],
        pdf_url=row["pdf_url"],
        source_url=row["source_url"],
    )


def _row_to_block(row: asyncpg.Record) -> PaperBlock:
    return PaperBlock(
        id=row["id"],
        paper_id=row["paper_id"],
        arxiv_id=row["arxiv_id"],
        block_type=row["block_type"],
        section_name=row["section_name"],
        page_number=row["page_number"],
        content=row["content"],
        markdown_content=row["markdown_content"],
        image_path=row["image_path"],
        order_index=row["order_index"],
        should_embed=row["should_embed"],
        metadata=_load_json_object(row["metadata"]),
        content_hash=row["content_hash"],
    )


def _load_json_object(value: object) -> dict[str, object]:
    import json

    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        return dict(json.loads(value))
    return {}


def _inserted_count(command: str) -> int:
    try:
        return int(command.rsplit(" ", 1)[-1])
    except (IndexError, ValueError):
        return 0
