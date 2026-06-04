from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ragarena.ingestion.hashing import content_hash


SUPPORTED_EXTENSIONS = {".md", ".txt"}


@dataclass(frozen=True)
class LoadedDocument:
    title: str
    source: str
    content: str
    content_hash: str


def load_documents(path: str | Path) -> list[LoadedDocument]:
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"Document path does not exist: {root}")

    files = [root] if root.is_file() else sorted(
        file for file in root.rglob("*") if file.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    documents = []
    for file in files:
        if file.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        content = file.read_text(encoding="utf-8").strip()
        if not content:
            continue

        documents.append(
            LoadedDocument(
                title=extract_title(file, content),
                source=str(file),
                content=content,
                content_hash=content_hash(content),
            )
        )

    return documents


def extract_title(file: Path, content: str) -> str:
    for line in content.splitlines():
        text = line.strip()
        if not text:
            continue
        if file.suffix.lower() == ".md" and text.startswith("#"):
            return text.lstrip("#").strip() or file.stem
        return text[:120]

    return file.stem
