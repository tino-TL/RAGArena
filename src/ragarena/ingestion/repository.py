from __future__ import annotations

import asyncpg

from ragarena.ingestion.loaders import LoadedDocument


CREATE_DOCUMENTS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    id BIGSERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    source TEXT NOT NULL,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""


async def ensure_documents_table(dsn: str) -> None:
    conn = await asyncpg.connect(dsn)
    try:
        await conn.execute(CREATE_DOCUMENTS_TABLE_SQL)
    finally:
        await conn.close()


async def insert_documents(dsn: str, documents: list[LoadedDocument]) -> int:
    if not documents:
        return 0

    conn = await asyncpg.connect(dsn)
    try:
        inserted_count = 0
        for doc in documents:
            result = await conn.execute(
                """
                INSERT INTO documents (title, source, content, content_hash)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (content_hash) DO NOTHING
                """,
                doc.title,
                doc.source,
                doc.content,
                doc.content_hash,
            )
            inserted_count += _inserted_count(result)

        return inserted_count
    finally:
        await conn.close()


async def source_exists(dsn: str, source: str) -> bool:
    conn = await asyncpg.connect(dsn)
    try:
        value = await conn.fetchval("SELECT 1 FROM documents WHERE source = $1 LIMIT 1", source)
        return value is not None
    finally:
        await conn.close()


async def insert_documents_missing_sources(dsn: str, documents: list[LoadedDocument]) -> int:
    if not documents:
        return 0

    inserted_count = 0
    for document in documents:
        if await source_exists(dsn, document.source):
            continue
        inserted_count += await insert_documents(dsn, [document])
    return inserted_count


async def upsert_documents_by_source(dsn: str, documents: list[LoadedDocument]) -> int:
    """Insert or update documents by source while preserving existing document ids."""
    if not documents:
        return 0

    conn = await asyncpg.connect(dsn)
    try:
        changed_count = 0
        for document in documents:
            updated = await conn.execute(
                """
                UPDATE documents
                SET title = $1,
                    content = $2,
                    content_hash = $3
                WHERE id = (
                    SELECT id FROM documents WHERE source = $4 ORDER BY id LIMIT 1
                )
                """,
                document.title,
                document.content,
                document.content_hash,
                document.source,
            )
            if _inserted_count(updated):
                changed_count += 1
                continue
            changed_count += await insert_documents(dsn, [document])
        return changed_count
    finally:
        await conn.close()


async def fetch_document_ids_by_source(dsn: str, source: str) -> list[int]:
    conn = await asyncpg.connect(dsn)
    try:
        rows = await conn.fetch("SELECT id FROM documents WHERE source = $1 ORDER BY id", source)
        return [row["id"] for row in rows]
    finally:
        await conn.close()


async def fetch_document_ids_by_source_prefix(dsn: str, source_prefix: str) -> list[int]:
    conn = await asyncpg.connect(dsn)
    try:
        rows = await conn.fetch(
            "SELECT id FROM documents WHERE source LIKE $1 ORDER BY id",
            f"{source_prefix}%",
        )
        return [row["id"] for row in rows]
    finally:
        await conn.close()


async def delete_documents_by_source_prefix(dsn: str, source_prefix: str) -> int:
    conn = await asyncpg.connect(dsn)
    try:
        result = await conn.execute(
            "DELETE FROM documents WHERE source LIKE $1",
            f"{source_prefix}%",
        )
        return _inserted_count(result)
    finally:
        await conn.close()


async def count_documents_by_hash(dsn: str, hashes: list[str]) -> int:
    if not hashes:
        return 0

    conn = await asyncpg.connect(dsn)
    try:
        return await conn.fetchval(
            "SELECT COUNT(*) FROM documents WHERE content_hash = ANY($1::text[])",
            hashes,
        )
    finally:
        await conn.close()


def _inserted_count(command: str) -> int:
    try:
        return int(command.rsplit(" ", 1)[-1])
    except (IndexError, ValueError):
        return 0
