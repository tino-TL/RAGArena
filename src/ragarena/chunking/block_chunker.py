from __future__ import annotations

import re
from collections import Counter, defaultdict

from ragarena.chunking.fixed_chunker import Chunk, estimate_token_count
from ragarena.chunking.repository import DocumentRecord
from ragarena.ingestion.hashing import content_hash
from ragarena.papers.models import PaperBlock
from ragarena.utils.text import sanitize_text

SINGLE_CHUNK_TYPES = {
    "abstract",
    "figure_caption",
    "table",
    "table_caption",
    "code",
    "algorithm",
    "title",
    "section_title",
}

PARAGRAPH_ATTACH_TYPES = {"paragraph", "formula", "list"}
SKIP_CHUNK_TYPES = {"reference", "footnote", "figure", "unknown"}


def chunk_block_documents(
    documents: list[DocumentRecord],
    blocks: list[PaperBlock],
) -> list[Chunk]:
    document_by_arxiv_id = map_documents_by_arxiv_id(documents)
    chunks: list[Chunk] = []
    paragraph_groups: dict[tuple[int, str], list[PaperBlock]] = defaultdict(list)

    for block in blocks:
        document = document_by_arxiv_id.get(block.arxiv_id)
        if document is None:
            continue
        if block.block_type in SKIP_CHUNK_TYPES:
            continue
        section_name = block.section_name or "unknown"
        if block.block_type in PARAGRAPH_ATTACH_TYPES:
            paragraph_groups[(document.id, section_name)].append(block)
            continue
        if block.block_type in SINGLE_CHUNK_TYPES:
            chunks.append(build_block_chunk(document, [block], block.block_type, section_name, block.content))

    for (document_id, section_name), grouped_blocks in paragraph_groups.items():
        document = next(document for document in documents if document.id == document_id)
        content = "\n\n".join(block.markdown_content or block.content for block in grouped_blocks)
        chunks.append(build_block_chunk(document, grouped_blocks, "paragraph", section_name, content))

    print_block_chunk_stats(chunks, documents)
    return chunks


def build_block_chunk(
    document: DocumentRecord,
    blocks: list[PaperBlock],
    chunk_type: str,
    section_name: str,
    content: str,
) -> Chunk:
    clean_content = sanitize_text(content).strip()
    source_block_ids = [block.id for block in blocks if block.id is not None]
    return Chunk(
        document_id=document.id,
        chunk_index=min(block.order_index for block in blocks),
        content=clean_content,
        token_count=estimate_token_count(clean_content),
        content_hash=content_hash(
                f"block:{document.id}:{chunk_type}:{section_name}:"
                f"{','.join(map(str, source_block_ids))}:{clean_content}"
        ),
        chunk_type=chunk_type,
        section_name=section_name,
        source_block_ids=source_block_ids,
        chunking_strategy="block",
    )


def map_documents_by_arxiv_id(documents: list[DocumentRecord]) -> dict[str, DocumentRecord]:
    mapping: dict[str, DocumentRecord] = {}
    for document in documents:
        arxiv_id = extract_field(document.content, "arXiv ID")
        if arxiv_id and "#block:" not in document.source:
            mapping.setdefault(arxiv_id, document)
    return mapping


def print_block_chunk_stats(chunks: list[Chunk], documents: list[DocumentRecord]) -> None:
    counts = Counter(chunk.document_id for chunk in chunks)
    arxiv_by_document_id = {
        document.id: extract_field(document.content, "arXiv ID") or str(document.id)
        for document in documents
    }
    for document_id, count in sorted(counts.items()):
        arxiv_id = arxiv_by_document_id.get(document_id, str(document_id))
        print(f"Parser stats: chunks_per_paper {arxiv_id}={count}")
        if count > 80:
            print(f"WARNING: {arxiv_id} produced {count} block chunks; expected <= 80 for most papers")


def extract_field(content: str, name: str) -> str | None:
    match = re.search(rf"^{re.escape(name)}:\s*(.+)$", content, re.MULTILINE)
    return match.group(1).strip() if match else None
