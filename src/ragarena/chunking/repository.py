from __future__ import annotations

from dataclasses import dataclass
import json

import asyncpg

from ragarena.chunking.fixed_chunker import Chunk


CREATE_DOCUMENT_CHUNKS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS document_chunks (
    id BIGSERIAL PRIMARY KEY,
    document_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    content_hash TEXT NOT NULL UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

ALTER_DOCUMENT_CHUNKS_METADATA_SQL = """
ALTER TABLE document_chunks
    ADD COLUMN IF NOT EXISTS chunk_type TEXT NOT NULL DEFAULT 'fixed',
    ADD COLUMN IF NOT EXISTS section_name TEXT,
    ADD COLUMN IF NOT EXISTS source_block_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    ADD COLUMN IF NOT EXISTS chunking_strategy TEXT NOT NULL DEFAULT 'fixed',
    ADD COLUMN IF NOT EXISTS retrieval_value TEXT,
    ADD COLUMN IF NOT EXISTS query_intents JSONB NOT NULL DEFAULT '[]'::jsonb,
    ADD COLUMN IF NOT EXISTS keywords JSONB NOT NULL DEFAULT '[]'::jsonb,
    ADD COLUMN IF NOT EXISTS planner_reason TEXT,
    ADD COLUMN IF NOT EXISTS metadata JSONB NOT NULL DEFAULT '{}'::jsonb;
"""


@dataclass(frozen=True)
class DocumentRecord:
    id: int
    title: str
    source: str
    content: str
    content_hash: str


async def ensure_document_chunks_table(dsn: str) -> None:
    conn = await asyncpg.connect(dsn)
    try:
        await conn.execute(CREATE_DOCUMENT_CHUNKS_TABLE_SQL)
        await conn.execute(ALTER_DOCUMENT_CHUNKS_METADATA_SQL)
    finally:
        await conn.close()


async def fetch_documents(dsn: str) -> list[DocumentRecord]:
    conn = await asyncpg.connect(dsn)
    try:
        rows = await conn.fetch(
            """
            SELECT id, title, source, content, content_hash
            FROM documents
            ORDER BY id
            """
        )
        return [
            DocumentRecord(
                id=row["id"],
                title=row["title"],
                source=row["source"],
                content=row["content"],
                content_hash=row["content_hash"],
            )
            for row in rows
        ]
    finally:
        await conn.close()


async def insert_chunks(dsn: str, chunks: list[Chunk]) -> int:
    if not chunks:
        return 0

    conn = await asyncpg.connect(dsn)
    try:
        inserted_count = 0
        for chunk in chunks:
            result = await conn.execute(
                """
                INSERT INTO document_chunks (
                    document_id,
                    chunk_index,
                    content,
                    token_count,
                    content_hash
                    ,
                    chunk_type,
                    section_name,
                    source_block_ids,
                    chunking_strategy,
                    retrieval_value,
                    query_intents,
                    keywords,
                    planner_reason,
                    metadata
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9, $10, $11::jsonb, $12::jsonb, $13, $14::jsonb)
                ON CONFLICT (content_hash) DO NOTHING
                """,
                chunk.document_id,
                chunk.chunk_index,
                chunk.content,
                chunk.token_count,
                chunk.content_hash,
                chunk.chunk_type,
                chunk.section_name,
                json.dumps(chunk.source_block_ids or []),
                chunk.chunking_strategy,
                chunk.retrieval_value,
                json.dumps(chunk.query_intents or []),
                json.dumps(chunk.keywords or []),
                chunk.planner_reason,
                json.dumps(chunk.metadata or {}),
            )
            inserted_count += _inserted_count(result)

        return inserted_count
    finally:
        await conn.close()


async def delete_chunks_for_documents(dsn: str, document_ids: list[int]) -> int:
    if not document_ids:
        return 0

    conn = await asyncpg.connect(dsn)
    try:
        result = await conn.execute(
            "DELETE FROM document_chunks WHERE document_id = ANY($1::bigint[])",
            document_ids,
        )
        return _inserted_count(result)
    finally:
        await conn.close()


async def count_chunks_by_hash(dsn: str, hashes: list[str]) -> int:
    if not hashes:
        return 0

    conn = await asyncpg.connect(dsn)
    try:
        return await conn.fetchval(
            "SELECT COUNT(*) FROM document_chunks WHERE content_hash = ANY($1::text[])",
            hashes,
        )
    finally:
        await conn.close()


def _inserted_count(command: str) -> int:
    try:
        return int(command.rsplit(" ", 1)[-1])
    except (IndexError, ValueError):
        return 0
