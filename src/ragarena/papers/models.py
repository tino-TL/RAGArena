from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ragarena.ingestion.hashing import content_hash
from ragarena.ingestion.loaders import LoadedDocument
from ragarena.utils.text import sanitize_text


@dataclass(frozen=True)
class PaperMetadata:
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    categories: list[str]
    published_at: datetime | None
    updated_at: datetime | None
    pdf_url: str
    source_url: str


@dataclass(frozen=True)
class StoredPaper:
    id: int
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    categories: list[str]
    published_at: datetime | None
    updated_at: datetime | None
    pdf_url: str
    source_url: str


@dataclass(frozen=True)
class PaperFile:
    paper_id: int
    arxiv_id: str
    pdf_url: str
    file_path: Path
    file_sha256: str
    file_size: int


SUPPORTED_BLOCK_TYPES = {
    "title",
    "abstract",
    "section_title",
    "paragraph",
    "content_block",
    "figure",
    "image_reference",
    "figure_caption",
    "table",
    "table_caption",
    "formula",
    "code",
    "algorithm",
    "list",
    "footnote",
    "reference",
    "appendix",
    "authors",
    "affiliation",
    "table_like",
    "date",
    "page_number",
    "metadata",
    "unknown",
}


EMBEDDABLE_BLOCK_TYPES = {
    "title",
    "abstract",
    "section_title",
    "paragraph",
    "content_block",
    "figure_caption",
    "table_caption",
    "table",
    "code",
    "algorithm",
}


@dataclass(frozen=True)
class PaperBlock:
    id: int | None
    paper_id: int
    arxiv_id: str
    block_type: str
    section_name: str | None
    page_number: int | None
    content: str
    markdown_content: str | None
    image_path: str | None
    order_index: int
    should_embed: bool
    metadata: dict[str, Any]
    content_hash: str


def paper_to_document(paper: PaperMetadata) -> LoadedDocument:
    content = (
        f"# {paper.title}\n\n"
        f"arXiv ID: {paper.arxiv_id}\n"
        f"Authors: {', '.join(paper.authors)}\n"
        f"Categories: {', '.join(paper.categories)}\n"
        f"PDF: {paper.pdf_url}\n\n"
        f"Abstract\n\n{paper.abstract}"
    )
    return LoadedDocument(
        title=paper.title,
        source=paper.source_url,
        content=content,
        content_hash=content_hash(content),
    )


def structured_paper_to_document(paper: StoredPaper, blocks: list[PaperBlock]) -> LoadedDocument:
    body = "\n\n".join(
        block.markdown_content or block.content
        for block in blocks
        if block.should_embed and (block.markdown_content or block.content).strip()
    )
    content = sanitize_text(
        f"# {paper.title}\n\n"
        f"arXiv ID: {paper.arxiv_id}\n"
        f"Authors: {', '.join(paper.authors)}\n"
        f"Categories: {', '.join(paper.categories)}\n"
        f"PDF: {paper.pdf_url}\n\n"
        f"{body or paper.abstract}"
    )
    return LoadedDocument(
        title=sanitize_text(paper.title),
        source=sanitize_text(paper.source_url),
        content=content,
        content_hash=content_hash(content),
    )


def normalize_block_type(value: str) -> str:
    normalized = sanitize_text(value).strip().lower()
    return normalized if normalized in SUPPORTED_BLOCK_TYPES else "unknown"


def should_embed_block(block_type: str) -> bool:
    return normalize_block_type(block_type) in EMBEDDABLE_BLOCK_TYPES
