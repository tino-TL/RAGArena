from __future__ import annotations

from dataclasses import dataclass
import json

import asyncpg

from ragarena.retrieval.vector_store import ElasticsearchVectorStore


@dataclass(frozen=True)
class EmbeddedChunk:
    chunk_id: int
    document_id: int
    model_name: str
    content: str
    embedding: list[float]
    metadata: dict[str, object]


@dataclass(frozen=True)
class IndexingResult:
    deleted_existing_index: bool
    loaded_embeddings: int
    indexed_chunks: int
    final_es_count: int
    postgres_embedding_count: int

    @property
    def count_matches(self) -> bool:
        return self.final_es_count == self.postgres_embedding_count


async def fetch_embedded_chunks(dsn: str, model_name: str) -> list[EmbeddedChunk]:
    conn = await asyncpg.connect(dsn)
    try:
        rows = await conn.fetch(
            """
            SELECT
                c.id AS chunk_id,
                c.document_id,
                c.content,
                c.section_name,
                c.chunking_strategy,
                c.chunk_type,
                c.metadata AS chunk_metadata,
                e.model_name,
                e.embedding,
                d.title AS document_title,
                d.source AS document_source,
                p.arxiv_id,
                p.id AS paper_id,
                p.title AS paper_title,
                p.authors AS paper_authors,
                p.categories AS paper_categories,
                p.pdf_url AS paper_pdf_url,
                p.source_url AS paper_source_url,
                CASE
                    WHEN POSITION('#' IN d.source) > 0 THEN SPLIT_PART(d.source, '#', 2)
                    WHEN p.arxiv_id IS NOT NULL THEN 'abstract'
                    ELSE NULL
                END AS paper_section
            FROM chunk_embeddings e
            JOIN document_chunks c ON c.id = e.chunk_id
            JOIN documents d ON d.id = c.document_id
            LEFT JOIN papers p ON p.source_url = SPLIT_PART(d.source, '#', 1)
            WHERE e.model_name = $1
            ORDER BY c.id
            """,
            model_name,
        )
        return [
            EmbeddedChunk(
                chunk_id=row["chunk_id"],
                document_id=row["document_id"],
                model_name=row["model_name"],
                content=row["content"],
                embedding=parse_embedding(row["embedding"]),
                metadata=build_chunk_metadata(row),
            )
            for row in rows
        ]
    finally:
        await conn.close()


def index_embedded_chunks(
    vector_store: ElasticsearchVectorStore,
    embedded_chunks: list[EmbeddedChunk],
    *,
    recreate: bool = True,
) -> int:
    if recreate:
        vector_store.recreate_index()
    else:
        vector_store.create_index()

    for chunk in embedded_chunks:
        vector_store.index_chunk(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            model_name=chunk.model_name,
            content=chunk.content,
            embedding=chunk.embedding,
            metadata=chunk.metadata,
        )

    if embedded_chunks:
        vector_store.refresh()

    return len(embedded_chunks)


def build_indexing_result(
    *,
    deleted_existing_index: bool,
    loaded_embeddings: int,
    indexed_chunks: int,
    final_es_count: int,
    postgres_embedding_count: int,
) -> IndexingResult:
    return IndexingResult(
        deleted_existing_index=deleted_existing_index,
        loaded_embeddings=loaded_embeddings,
        indexed_chunks=indexed_chunks,
        final_es_count=final_es_count,
        postgres_embedding_count=postgres_embedding_count,
    )


def parse_embedding(value: object) -> list[float]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list):
        raise TypeError(f"Unsupported embedding value type: {type(value)!r}")

    return [float(item) for item in value]


def build_chunk_metadata(row: asyncpg.Record) -> dict[str, object]:
    metadata: dict[str, object] = {
        "document_title": row["document_title"],
        "document_source": row["document_source"],
        "section_name": row["section_name"],
        "chunk_type": row["chunk_type"],
        "chunking_strategy": row["chunking_strategy"],
        "paper_id": row["paper_id"],
    }
    optional_fields = {
        "arxiv_id": row["arxiv_id"],
        "paper_title": row["paper_title"],
        "paper_authors": parse_json_array(row["paper_authors"]),
        "paper_categories": parse_json_array(row["paper_categories"]),
        "paper_pdf_url": row["paper_pdf_url"],
        "paper_source_url": row["paper_source_url"],
        "paper_section": row["paper_section"],
    }
    for key, value in optional_fields.items():
        if value:
            metadata[key] = value
    chunk_metadata = parse_json_object(row["chunk_metadata"])
    metadata.update(chunk_metadata)
    return metadata


def parse_json_array(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        return [str(item) for item in json.loads(value)]
    return []


def parse_json_object(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        return dict(json.loads(value))
    return {}
