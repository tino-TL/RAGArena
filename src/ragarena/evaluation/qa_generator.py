from __future__ import annotations

import json
import re
from pathlib import Path

import asyncpg

from ragarena.config import settings


async def generate_eval_qa(*, paper_id: int, num_questions: int, output: Path) -> int:
    chunks = await fetch_eval_candidate_chunks(paper_id)
    items = [qa_from_chunk(chunk, offset) for offset, chunk in enumerate(chunks[:num_questions])]
    existing = load_existing_items(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(existing + items, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"generated_questions: {len(items)}")
    print(f"output: {output}")
    return len(items)


async def fetch_eval_candidate_chunks(paper_id: int) -> list[dict[str, object]]:
    conn = await asyncpg.connect(settings.postgres_dsn)
    try:
        rows = await conn.fetch(
            """
            SELECT
                c.id AS source_chunk_id,
                c.section_name,
                c.content,
                p.id AS paper_id,
                p.arxiv_id
            FROM document_chunks c
            JOIN documents d ON d.id = c.document_id
            JOIN papers p ON p.source_url = split_part(d.source, '#', 1)
            WHERE p.id = $1
            ORDER BY c.chunk_index
            """,
            paper_id,
        )
        return [dict(row) for row in rows if row["section_name"]]
    finally:
        await conn.close()


def qa_from_chunk(chunk: dict[str, object], offset: int) -> dict[str, object]:
    section = str(chunk["section_name"])
    keywords = extract_keywords(str(chunk["content"]))
    return {
        "paper_id": int(str(chunk["paper_id"])),
        "arxiv_id": str(chunk["arxiv_id"]),
        "query": query_for_section(section, offset),
        "gold_sections": [section],
        "gold_keywords": keywords,
        "source_chunk_id": int(str(chunk["source_chunk_id"])),
    }


def query_for_section(section: str, offset: int) -> str:
    lowered = section.lower()
    if "abstract" in lowered:
        return "What is the main contribution of the paper?"
    if "method" in lowered or "model" in lowered:
        return f"What method is described in {section}?"
    if "result" in lowered or "performance" in lowered or "calculate" in lowered:
        return f"What result is reported in {section}?"
    if "conclusion" in lowered or "consideration" in lowered or "limitation" in lowered:
        return "What limitations or conclusions does the paper discuss?"
    if "table" in lowered:
        return f"What does the table in {section} show?"
    return f"What information is discussed in {section}?"


def extract_keywords(content: str) -> list[str]:
    candidates = re.findall(r"\b(?:[A-Z][A-Za-z/&-]{2,}|[A-Z]{2,}\s*\d*[A-Z]?|\d+(?:\.\d+)?%?)\b", content)
    cleaned: list[str] = []
    for value in candidates:
        item = " ".join(value.split())
        if len(item) < 2 or item.lower() in {"the", "this", "that"}:
            continue
        if item not in cleaned:
            cleaned.append(item)
        if len(cleaned) >= 4:
            break
    if cleaned:
        return cleaned
    words = [word.strip(".,;:()[]") for word in content.split() if len(word.strip(".,;:()[]")) > 5]
    return list(dict.fromkeys(words[:3]))


def load_existing_items(path: Path) -> list[object]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("existing QA file must contain a JSON list")
    return payload
