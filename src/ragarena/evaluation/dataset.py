from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class EvaluationCase:
    query: str
    relevant_chunk_ids: set[int] = field(default_factory=set)
    relevant_document_ids: set[int] = field(default_factory=set)
    category: str | None = None
    tags: list[str] = field(default_factory=list)
    expected_answer: str | None = None
    gold_sections: list[str] = field(default_factory=list)
    gold_page_numbers: list[int] = field(default_factory=list)
    gold_visual_refs: list[str] = field(default_factory=list)
    notes: str | None = None


def load_evaluation_dataset(path: Path) -> list[EvaluationCase]:
    cases: list[EvaluationCase] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not payload.get("relevant_chunk_ids") and not payload.get("relevant_document_ids"):
            raise ValueError(f"Evaluation row {line_number} must define relevant ids")
        cases.append(
            EvaluationCase(
                query=str(payload["query"]),
                relevant_chunk_ids={int(value) for value in payload.get("relevant_chunk_ids", [])},
                relevant_document_ids={int(value) for value in payload.get("relevant_document_ids", [])},
                category=payload.get("category"),
                tags=[str(value) for value in payload.get("tags", [])],
                expected_answer=payload.get("expected_answer"),
                gold_sections=[str(value) for value in payload.get("gold_sections", [])],
                gold_page_numbers=[int(value) for value in payload.get("gold_page_numbers", [])],
                gold_visual_refs=[str(value) for value in payload.get("gold_visual_refs", [])],
                notes=payload.get("notes"),
            )
        )
    return cases
