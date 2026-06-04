from __future__ import annotations

from ragarena.evaluation.retrieval_eval import (
    QAGoldItem,
    QueryEvalResult,
    compute_per_paper_metrics,
    compute_mrr,
    compute_recall_at_k,
    evaluate_retrieval,
    match_gold,
)
from ragarena.retrieval.vector_store import SearchResult


def result(
    content: str,
    *,
    section_name: str | None = None,
    chunk_id: int = 1,
) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        document_id=10,
        content=content,
        score=1.0,
        model_name="test",
        source_scores={"dense": 1.0},
        section_name=section_name,
        metadata={},
    )


def test_gold_section_hit() -> None:
    gold = QAGoldItem(
        query="q",
        gold_sections=["3.1.2 Calculate Factor Returns"],
        gold_keywords=[],
    )

    assert match_gold(result("body", section_name="3.1.2 Calculate Factor Returns"), gold)


def test_gold_keyword_hit() -> None:
    gold = QAGoldItem(query="q", gold_sections=["Results"], gold_keywords=["MOM 7M"])

    assert match_gold(result("The best factor is MOM 7M."), gold)


def test_recall_at_k_computes_hit_rate() -> None:
    rows = [
        QueryEvalResult("q1", ["A"], 1, ["A"]),
        QueryEvalResult("q2", ["B"], 3, ["X", "Y", "B"]),
        QueryEvalResult("q3", ["C"], None, []),
    ]

    assert compute_recall_at_k(rows, 1) == 1 / 3
    assert compute_recall_at_k(rows, 3) == 2 / 3


def test_mrr_computes_mean_reciprocal_rank() -> None:
    rows = [
        QueryEvalResult("q1", ["A"], 1, []),
        QueryEvalResult("q2", ["B"], 4, []),
        QueryEvalResult("q3", ["C"], None, []),
    ]

    assert compute_mrr(rows) == (1.0 + 0.25 + 0.0) / 3


def test_empty_results_do_not_error() -> None:
    gold = [QAGoldItem(query="q", gold_sections=["A"], gold_keywords=["keyword"])]

    report = evaluate_retrieval(gold, search_fn=lambda query, top_k, mode, chunking_strategy, paper_id, arxiv_id: [])

    assert report.total_queries == 1
    assert report.recall_at_1 == 0.0
    assert report.mrr == 0.0
    assert report.per_query[0].hit_rank is None


def test_multiple_gold_sections_supported() -> None:
    gold = QAGoldItem(query="q", gold_sections=["A", "B"], gold_keywords=[])

    assert match_gold(result("body", section_name="B"), gold)


def test_source_chunk_id_hit() -> None:
    gold = QAGoldItem(query="q", gold_sections=["A"], gold_keywords=[], source_chunk_id=42)

    assert match_gold(result("body", section_name="X", chunk_id=42), gold)


def test_multi_paper_search_receives_paper_filter() -> None:
    seen: dict[str, object] = {}
    gold = [QAGoldItem(query="q", gold_sections=["A"], gold_keywords=[], paper_id=7, arxiv_id="2401.1")]

    def fake_search(query, top_k, mode, chunking_strategy, paper_id, arxiv_id):
        seen["paper_id"] = paper_id
        seen["arxiv_id"] = arxiv_id
        return [result("body", section_name="A")]

    report = evaluate_retrieval(gold, search_fn=fake_search)

    assert report.recall_at_1 == 1.0
    assert seen == {"paper_id": 7, "arxiv_id": "2401.1"}


def test_per_paper_metric_aggregation() -> None:
    rows = [
        QueryEvalResult("q1", ["A"], 1, ["A"], paper_id=1),
        QueryEvalResult("q2", ["B"], None, [], paper_id=1),
        QueryEvalResult("q3", ["C"], 2, ["X", "C"], paper_id=2),
    ]

    metrics = compute_per_paper_metrics(rows)

    assert metrics["1"]["total_queries"] == 2
    assert metrics["1"]["recall@1"] == 0.5
    assert metrics["2"]["mrr"] == 0.5
