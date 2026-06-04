from __future__ import annotations

from dataclasses import dataclass
import json

import asyncpg


CREATE_CHUNK_EMBEDDINGS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS chunk_embeddings (
    id BIGSERIAL PRIMARY KEY,
    chunk_id BIGINT NOT NULL REFERENCES document_chunks(id) ON DELETE CASCADE,
    model_name TEXT NOT NULL,
    embedding JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chunk_embeddings_chunk_model_unique UNIQUE (chunk_id, model_name)
);
"""


@dataclass(frozen=True)
class ChunkRecord:
    id: int
    document_id: int
    chunk_index: int
    content: str
    token_count: int
    content_hash: str


async def ensure_chunk_embeddings_table(dsn: str) -> None:
    conn = await asyncpg.connect(dsn)
    try:
        await conn.execute(CREATE_CHUNK_EMBEDDINGS_TABLE_SQL)
    finally:
        await conn.close()


async def fetch_chunks(dsn: str) -> list[ChunkRecord]:
    conn = await asyncpg.connect(dsn)
    try:
        rows = await conn.fetch(
            """
            SELECT id, document_id, chunk_index, content, token_count, content_hash
            FROM document_chunks
            ORDER BY id
            """
        )
        return [
            ChunkRecord(
                id=row["id"],
                document_id=row["document_id"],
                chunk_index=row["chunk_index"],
                content=row["content"],
                token_count=row["token_count"],
                content_hash=row["content_hash"],
            )
            for row in rows
        ]
    finally:
        await conn.close()


async def fetch_chunks_without_embeddings(
    dsn: str,
    model_name: str,
) -> list[ChunkRecord]:
    conn = await asyncpg.connect(dsn)
    try:
        rows = await conn.fetch(
            """
            SELECT c.id, c.document_id, c.chunk_index, c.content, c.token_count, c.content_hash
            FROM document_chunks c
            LEFT JOIN chunk_embeddings e
                ON e.chunk_id = c.id
                AND e.model_name = $1
            WHERE e.id IS NULL
            ORDER BY c.id
            """,
            model_name,
        )
        return [
            ChunkRecord(
                id=row["id"],
                document_id=row["document_id"],
                chunk_index=row["chunk_index"],
                content=row["content"],
                token_count=row["token_count"],
                content_hash=row["content_hash"],
            )
            for row in rows
        ]
    finally:
        await conn.close()


async def insert_embeddings(
    dsn: str,
    model_name: str,
    chunk_ids: list[int],
    embeddings: list[list[float]],
) -> int:
    if len(chunk_ids) != len(embeddings):
        raise ValueError("chunk_ids and embeddings must have the same length")
    if not chunk_ids:
        return 0

    conn = await asyncpg.connect(dsn)
    try:
        inserted_count = 0
        for chunk_id, embedding in zip(chunk_ids, embeddings, strict=True):
            result = await conn.execute(
                """
                INSERT INTO chunk_embeddings (chunk_id, model_name, embedding)
                VALUES ($1, $2, $3::jsonb)
                ON CONFLICT (chunk_id, model_name) DO NOTHING
                """,
                chunk_id,
                model_name,
                json.dumps(embedding),
            )
            inserted_count += _inserted_count(result)

        return inserted_count
    finally:
        await conn.close()


async def count_embeddings_for_model(
    dsn: str,
    model_name: str,
    chunk_ids: list[int],
) -> int:
    if not chunk_ids:
        return 0

    conn = await asyncpg.connect(dsn)
    try:
        return await conn.fetchval(
            """
            SELECT COUNT(*)
            FROM chunk_embeddings
            WHERE model_name = $1
              AND chunk_id = ANY($2::bigint[])
            """,
            model_name,
            chunk_ids,
        )
    finally:
        await conn.close()


def _inserted_count(command: str) -> int:
    try:
        return int(command.rsplit(" ", 1)[-1])
    except (IndexError, ValueError):
        return 0
