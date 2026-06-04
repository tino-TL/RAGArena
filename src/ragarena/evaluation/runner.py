from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any, cast

from ragarena.evaluation.dataset import EvaluationCase, load_evaluation_dataset
from ragarena.evaluation.metrics import evaluate_ranked_results
from ragarena.retrieval.service import RetrievalRequest, RetrievalService
from ragarena.retrieval.vector_store import SearchResult


def run_retrieval_evaluation(
    *,
    dataset_path: Path,
    strategies: list[str],
    top_k_values: list[int],
    output_path: Path,
) -> dict[str, object]:
    cases = load_evaluation_dataset(dataset_path)
    service = RetrievalService()
    max_top_k = max(top_k_values)
    strategy_reports = []

    for strategy in strategies:
        rows = []
        latencies = []
        for case in cases:
            response = service.search(
                RetrievalRequest(
                    query=case.query,
                    strategy=strategy,  # type: ignore[arg-type]
                    top_k=max_top_k,
                )
            )
            latencies.append(response.latency_ms)
            rows.append(evaluate_case(case, response.results, top_k_values))

        strategy_reports.append(
            {
                "strategy": strategy,
                "cases": rows,
                "summary": summarize_rows(rows, latencies, top_k_values),
            }
        )

    report = {
        "dataset": str(dataset_path),
        "case_count": len(cases),
        "top_k": top_k_values,
        "strategies": strategy_reports,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    output_path.with_suffix(".md").write_text(render_markdown_report(report), encoding="utf-8")
    return report


def evaluate_case(case: EvaluationCase, results: list[SearchResult], top_k_values: list[int]) -> dict[str, object]:
    chunk_ids = [result.chunk_id for result in results]
    document_ids = [result.document_id for result in results]
    relevant_ids = case.relevant_chunk_ids
    ranked_ids = chunk_ids
    id_type = "chunk"
    if not relevant_ids:
        relevant_ids = case.relevant_document_ids
        ranked_ids = document_ids
        id_type = "document"

    metrics = {
        str(top_k): asdict(
            evaluate_ranked_results(
                ranked_ids,
                relevant_ids,
                top_k=top_k,
            )
        )
        for top_k in top_k_values
    }
    return {
        "query": case.query,
        "category": case.category,
        "id_type": id_type,
        "metrics": metrics,
        "retrieved_chunk_ids": chunk_ids,
        "retrieved_document_ids": document_ids,
    }


def summarize_rows(
    rows: list[dict[str, object]],
    latencies: list[float],
    top_k_values: list[int],
) -> dict[str, object]:
    summary: dict[str, object] = {
        "latency_ms_avg": round(mean(latencies), 3) if latencies else 0.0,
        "latency_ms_p95": percentile(latencies, 95),
    }
    for top_k in top_k_values:
        key = str(top_k)
        metrics = [row["metrics"][key] for row in rows]  # type: ignore[index]
        summary[f"recall@{top_k}"] = average_metric(metrics, "recall")
        summary[f"mrr@{top_k}"] = average_metric(metrics, "mrr")
        summary[f"ndcg@{top_k}"] = average_metric(metrics, "ndcg")
        summary[f"hit_rate@{top_k}"] = average_metric(metrics, "hit_rate")
    return summary


def average_metric(metrics: list[object], name: str) -> float:
    values = [float(metric[name]) for metric in metrics]  # type: ignore[index]
    return round(mean(values), 4) if values else 0.0


def percentile(values: list[float], percent: int) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = min(len(sorted_values) - 1, round((percent / 100) * (len(sorted_values) - 1)))
    return round(sorted_values[index], 3)


def render_markdown_report(report: dict[str, object]) -> str:
    top_k_values = cast(list[int], report["top_k"])
    lines = [
        "# RAGArena Retrieval Evaluation",
        "",
        f"Dataset: `{report['dataset']}`",
        f"Cases: {report['case_count']}",
        "",
        "| Strategy | Latency Avg ms | Latency P95 ms | " + " | ".join(
            f"Recall@{top_k} | MRR@{top_k} | nDCG@{top_k} | Hit Rate@{top_k}"
            for top_k in top_k_values
        ) + " |",
        "|---|---:|---:|" + "---:|" * (4 * len(top_k_values)),
    ]
    for strategy in cast(list[dict[str, Any]], report["strategies"]):
        summary = cast(dict[str, Any], strategy["summary"])
        cells = [
            str(strategy["strategy"]),
            str(summary["latency_ms_avg"]),
            str(summary["latency_ms_p95"]),
        ]
        for top_k in top_k_values:
            cells.extend(
                [
                    str(summary[f"recall@{top_k}"]),
                    str(summary[f"mrr@{top_k}"]),
                    str(summary[f"ndcg@{top_k}"]),
                    str(summary[f"hit_rate@{top_k}"]),
                ]
            )
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
