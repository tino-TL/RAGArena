from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ragarena.ingestion.hashing import content_hash


@dataclass(frozen=True)
class Chunk:
    document_id: int
    chunk_index: int
    content: str
    token_count: int
    content_hash: str
    # Best-effort parser/planner metadata for debugging and analysis only.
    # Retrieval, embedding, and evaluation must not treat it as ground truth.
    chunk_type: str = "fixed"
    section_name: str | None = None
    source_block_ids: list[int] | None = None
    chunking_strategy: str = "fixed"
    retrieval_value: str | None = None
    query_intents: list[str] | None = None
    keywords: list[str] | None = None
    planner_reason: str | None = None
    metadata: dict[str, Any] | None = None


def chunk_document(
    document_id: int,
    content: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be greater than or equal to 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    normalized = content.replace("\r\n", "\n").strip()
    if not normalized:
        return []

    chunks = []
    start = 0
    chunk_index = 0

    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        chunk_content = normalized[start:end].strip()
        if chunk_content:
            chunks.append(
                Chunk(
                    document_id=document_id,
                    chunk_index=chunk_index,
                    content=chunk_content,
                    token_count=estimate_token_count(chunk_content),
                    content_hash=content_hash(chunk_content),
                )
            )
            chunk_index += 1

        if end == len(normalized):
            break

        start = end - chunk_overlap

    return chunks


def estimate_token_count(content: str) -> int:
    words = content.split()
    if words:
        return len(words)

    return max(1, len(content) // 4)
