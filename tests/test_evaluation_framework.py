from __future__ import annotations

import json

from ragarena.evaluation.framework import (
    compute_answer_metrics,
    compute_retrieval_metrics,
    compute_source_metrics,
    load_gold_qa,
    run_benchmark,
    summarize_case_results,
)
from ragarena.retrieval.vector_store import SearchResult


def test_load_gold_qa_parses_resume_grade_fields(tmp_path) -> None:
    dataset = tmp_path / "qa_gold.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "id": "qa-1",
                    "query": "What does Figure 2 show?",
                    "expected_answer": "Recall improves.",
                    "category": "figure_table",
                    "tags": ["visual"],
                    "gold_sections": ["Results"],
                    "gold_keywords": ["Figure 2"],
                    "gold_chunk_ids": [7],
                    "gold_page_numbers": [5],
                    "gold_visual_refs": ["Figure 2"],
                    "answer_must_include": ["Recall improves"],
                }
            ]
        ),
        encoding="utf-8",
    )

    items = load_gold_qa(dataset)

    assert items[0].id == "qa-1"
    assert items[0].gold_chunk_ids == {7}
    assert items[0].gold_page_numbers == {5}
    assert items[0].tags == ["visual"]


def test_source_and_retrieval_metrics_use_gold_evidence() -> None:
    item = load_gold_qa_item()
    results = [
        SearchResult(
            chunk_id=7,
            document_id=3,
            content="Figure 2 reports that recall improves with context length.",
            score=1.0,
            model_name="test",
            source_scores={"bm25": 1.0},
            section_name="Results",
            metadata={
                "chunk_type": "fused",
                "page_number": 5,
                "visual_refs": ["Figure 2"],
            },
        )
    ]

    retrieval = compute_retrieval_metrics(item, results, [1, 3])
    source = compute_source_metrics(item, results)

    assert retrieval["1"]["recall"] == 1.0
    assert source["chunk_hit"] is True
    assert source["section_hit"] is True
    assert source["page_hit"] is True
    assert source["visual_ref_hit"] is True
    assert source["citation_source_hit"] is True


def test_answer_metrics_use_deterministic_and_judge_labels() -> None:
    item = load_gold_qa_item()
    metrics = compute_answer_metrics(
        item,
        "variant-a",
        "The paper shows that recall improves with context length.",
        {},
    )

    assert metrics["deterministic_answer_correct"] is True
    assert metrics["judge_answer_correct"] is None


def test_run_benchmark_writes_variant_report(tmp_path, monkeypatch) -> None:
    dataset = tmp_path / "qa_gold.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "id": "qa-1",
                    "query": "What does Figure 2 show?",
                    "category": "figure_table",
                    "gold_chunk_ids": [7],
                    "gold_sections": ["Results"],
                    "gold_keywords": ["Figure 2"],
                }
            ]
        ),
        encoding="utf-8",
    )
    plan = tmp_path / "ablation_plan.json"
    plan.write_text(
        json.dumps(
            {
                "experiments": [
                    {
                        "id": "D_retrieval_stack",
                        "variants": [{"name": "bm25", "strategy": "bm25"}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "report.json"

    def fake_search_chunks(**kwargs):
        return type(
            "Response",
            (),
            {
                "results": [
                    SearchResult(
                        chunk_id=7,
                        document_id=3,
                        content="Figure 2 reports recall.",
                        score=1.0,
                        model_name="test",
                        source_scores={"bm25": 1.0},
                        section_name="Results",
                        metadata={},
                    )
                ]
            },
        )()

    monkeypatch.setattr("ragarena.evaluation.framework.search_chunks", fake_search_chunks)

    report = run_benchmark(
        dataset_path=dataset,
        plan_path=plan,
        output_path=output,
        top_k_values=[1],
    )

    assert report["case_count"] == 1
    assert report["variants"][0]["summary"]["recall@1"] == 1.0
    assert output.exists()
    assert output.with_suffix(".md").exists()


def test_summary_reports_error_cases() -> None:
    summary = summarize_case_results([], [1])

    assert summary["ok_cases"] == 0
    assert summary["error_cases"] == 0
    assert summary["recall@1"] == 0.0


def load_gold_qa_item():
    return load_gold_qa_from_payload(
        [
            {
                "id": "qa-1",
                "query": "What does Figure 2 show?",
                "expected_answer": "Recall improves",
                "category": "figure_table",
                "gold_sections": ["Results"],
                "gold_keywords": ["Figure 2"],
                "gold_chunk_ids": [7],
                "gold_page_numbers": [5],
                "gold_visual_refs": ["Figure 2"],
                "answer_must_include": ["recall improves"],
            }
        ]
    )[0]


def load_gold_qa_from_payload(payload):
    from ragarena.evaluation.framework import parse_gold_item

    return [parse_gold_item(item, index) for index, item in enumerate(payload, start=1)]
