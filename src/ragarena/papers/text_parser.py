from __future__ import annotations

from pathlib import Path

from ragarena.ingestion.hashing import content_hash
from ragarena.papers.models import PaperBlock, PaperFile
from ragarena.utils.text import sanitize_text


def parse_pdf_to_text_blocks(paper_file: PaperFile) -> list[PaperBlock]:
    section_name = sanitize_text("full_text")
    text = sanitize_text(extract_pdf_text(paper_file.file_path))
    if not text.strip():
        return []

    return [
        PaperBlock(
            id=None,
            paper_id=paper_file.paper_id,
            arxiv_id=paper_file.arxiv_id,
            block_type="content_block",
            section_name=section_name,
            page_number=None,
            content=text,
            markdown_content=text,
            image_path=None,
            order_index=0,
            should_embed=True,
            metadata={"parser": "simple"},
            content_hash=content_hash(f"{paper_file.arxiv_id}\n{section_name}\n{text}"),
        )
    ]


def extract_pdf_text(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = []
    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        normalized = normalize_pdf_text(text)
        if normalized:
            pages.append(f"Page {index}\n\n{normalized}")
    return "\n\n".join(pages)


def normalize_pdf_text(text: str) -> str:
    text = sanitize_text(text)
    lines = [" ".join(line.split()) for line in text.replace("\r\n", "\n").split("\n")]
    return "\n".join(line for line in lines if line).strip()
