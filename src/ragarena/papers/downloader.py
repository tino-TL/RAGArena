from __future__ import annotations

import hashlib
from pathlib import Path

import requests

from ragarena.papers.models import PaperFile, StoredPaper

DEFAULT_PAPER_DIR = Path("data/papers")


def download_paper_pdf(
    paper: StoredPaper,
    *,
    output_dir: Path = DEFAULT_PAPER_DIR,
    timeout: int = 60,
) -> PaperFile:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{safe_arxiv_id(paper.arxiv_id)}.pdf"

    response = requests.get(paper.pdf_url, timeout=timeout)
    response.raise_for_status()
    content = response.content
    output_path.write_bytes(content)

    return PaperFile(
        paper_id=paper.id,
        arxiv_id=paper.arxiv_id,
        pdf_url=paper.pdf_url,
        file_path=output_path,
        file_sha256=hashlib.sha256(content).hexdigest(),
        file_size=len(content),
    )


def safe_arxiv_id(arxiv_id: str) -> str:
    return arxiv_id.replace("/", "_").replace("\\", "_")
