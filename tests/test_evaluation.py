from __future__ import annotations

from pathlib import Path

from ragarena.evaluation.metrics import evaluate_ranked_results
from ragarena.evaluation.runner import run_retrieval_evaluation
from ragarena.retrieval.search import SearchResponse
from ragarena.retrieval.vector_store import SearchResult


def test_retrieval_metrics() -> None:
    metrics = evaluate_ranked_results([10, 20, 30], {20, 30}, top_k=3)

    assert metrics.recall == 1.0
    assert metrics.mrr == 0.5
    assert metrics.hit_rate == 1.0
    assert 0.0 < metrics.ndcg <= 1.0


def test_retrieval_evaluation_writes_json_and_markdown(tmp_path, monkeypatch) -> None:
    dataset = tmp_path / "retrieval.jsonl"
    dataset.write_text(
        '{"query":"LangGraph","relevant_chunk_ids":[1],"category":"workflow"}\n',
        encoding="utf-8",
    )
    output = tmp_path / "report.json"

    def fake_search(self, request):
        return SearchResponse(
            query=request.query,
            top_k=request.top_k,
            mode=request.strategy,
            strategy=request.strategy,
            latency_ms=12.5,
            results=[
                SearchResult(
                    chunk_id=1,
                    document_id=10,
                    content="LangGraph workflow",
                    score=1.0,
                    model_name="BAAI/bge-m3",
                    source_scores={request.strategy: 1.0},
                )
            ],
        )

    monkeypatch.setattr("ragarena.evaluation.runner.RetrievalService.search", fake_search)

    report = run_retrieval_evaluation(
        dataset_path=dataset,
        strategies=["bm25", "hybrid"],
        top_k_values=[1],
        output_path=output,
    )

    assert report["case_count"] == 1
    assert output.exists()
    assert Path(output.with_suffix(".md")).exists()
    assert report["strategies"][0]["summary"]["recall@1"] == 1.0
