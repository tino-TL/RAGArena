from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ragarena.retrieval.search import SearchStrategy, search_chunks
from ragarena.retrieval.vector_store import SearchResult


@dataclass(frozen=True)
class QAGoldItem:
    query: str
    gold_sections: list[str]
    gold_keywords: list[str]
    paper_id: int | None = None
    arxiv_id: str | None = None
    source_chunk_id: int | None = None


@dataclass(frozen=True)
class QueryEvalResult:
    query: str
    gold_sections: list[str]
    hit_rank: int | None
    top_sections: list[str | None]
    paper_id: int | None = None
    arxiv_id: str | None = None

    @property
    def hit_at_1(self) -> bool:
        return self.hit_rank is not None and self.hit_rank <= 1

    @property
    def hit_at_3(self) -> bool:
        return self.hit_rank is not None and self.hit_rank <= 3

    @property
    def hit_at_5(self) -> bool:
        return self.hit_rank is not None and self.hit_rank <= 5


@dataclass(frozen=True)
class RetrievalEvalReport:
    per_query: list[QueryEvalResult]
    total_queries: int
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mrr: float
    per_paper: dict[str, dict[str, float | int]]


SearchFn = Callable[[str, int, SearchStrategy, str | None, int | None, str | None], list[SearchResult]]


def load_eval_dataset(path: str | Path) -> list[QAGoldItem]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("QA eval dataset must be a JSON list")
    return [
        QAGoldItem(
            query=str(item["query"]),
            gold_sections=[str(value) for value in item.get("gold_sections", [])],
            gold_keywords=[str(value) for value in item.get("gold_keywords", [])],
            paper_id=int(item["paper_id"]) if item.get("paper_id") is not None else None,
            arxiv_id=str(item["arxiv_id"]) if item.get("arxiv_id") is not None else None,
            source_chunk_id=int(item["source_chunk_id"]) if item.get("source_chunk_id") is not None else None,
        )
        for item in payload
    ]


def evaluate_retrieval(
    gold_items: list[QAGoldItem],
    *,
    top_k: int = 5,
    mode: SearchStrategy = "dense",
    chunking_strategy: str | None = None,
    search_fn: SearchFn | None = None,
) -> RetrievalEvalReport:
    searcher = search_fn or default_search
    rows: list[QueryEvalResult] = []
    for item in gold_items:
        results = searcher(item.query, top_k, mode, chunking_strategy, item.paper_id, item.arxiv_id)
        hit_rank = first_hit_rank(results, item)
        rows.append(
            QueryEvalResult(
                query=item.query,
                gold_sections=item.gold_sections,
                hit_rank=hit_rank,
                top_sections=[result.section_name or result.metadata.get("section_name") for result in results],
                paper_id=item.paper_id,
                arxiv_id=item.arxiv_id,
            )
        )
    return RetrievalEvalReport(
        per_query=rows,
        total_queries=len(rows),
        recall_at_1=compute_recall_at_k(rows, 1),
        recall_at_3=compute_recall_at_k(rows, 3),
        recall_at_5=compute_recall_at_k(rows, 5),
        mrr=compute_mrr(rows),
        per_paper=compute_per_paper_metrics(rows),
    )


def default_search(
    query: str,
    top_k: int,
    mode: SearchStrategy,
    chunking_strategy: str | None,
    paper_id: int | None,
    arxiv_id: str | None,
) -> list[SearchResult]:
    return search_chunks(
        query=query,
        top_k=top_k,
        mode=mode,
        chunking_strategy=chunking_strategy,
        paper_id=paper_id,
        arxiv_id=arxiv_id,
    ).results


def compute_recall_at_k(results: list[QueryEvalResult], k: int) -> float:
    if not results:
        return 0.0
    hits = sum(1 for result in results if result.hit_rank is not None and result.hit_rank <= k)
    return hits / len(results)


def compute_mrr(results: list[QueryEvalResult]) -> float:
    if not results:
        return 0.0
    return sum(0.0 if result.hit_rank is None else 1.0 / result.hit_rank for result in results) / len(results)


def first_hit_rank(results: list[SearchResult], gold_item: QAGoldItem) -> int | None:
    for rank, result in enumerate(results, start=1):
        if match_gold(result, gold_item):
            return rank
    return None


def match_gold(result: SearchResult, gold_item: QAGoldItem) -> bool:
    if gold_item.source_chunk_id is not None and result.chunk_id == gold_item.source_chunk_id:
        return True
    section_name = result.section_name or str(result.metadata.get("section_name") or "")
    normalized_section = normalize_text(section_name)
    gold_sections = {normalize_text(section) for section in gold_item.gold_sections}
    if normalized_section in gold_sections:
        return True

    content = normalize_text(result.content)
    if any(normalize_text(keyword) in content for keyword in gold_item.gold_keywords):
        return True
    return any(normalize_text(f"## {section}") in content for section in gold_item.gold_sections)


def compute_per_paper_metrics(results: list[QueryEvalResult]) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[QueryEvalResult]] = {}
    for result in results:
        key = str(result.paper_id or result.arxiv_id or "unknown")
        grouped.setdefault(key, []).append(result)
    return {
        key: {
            "total_queries": len(items),
            "recall@1": compute_recall_at_k(items, 1),
            "recall@3": compute_recall_at_k(items, 3),
            "recall@5": compute_recall_at_k(items, 5),
            "mrr": compute_mrr(items),
        }
        for key, items in sorted(grouped.items())
    }


def normalize_text(value: str) -> str:
    return " ".join(value.lower().split())
