from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from ragarena.evaluation.runner import run_retrieval_evaluation
from ragarena.evaluation.retrieval_eval import evaluate_retrieval, load_eval_dataset
from ragarena.evaluation.framework import run_benchmark

STRATEGY_CHOICES = [
    "bm25",
    "dense",
    "vector",
    "hybrid",
    "hybrid_hyde",
    "hybrid_rerank",
    "hybrid_hyde_rerank",
]


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Evaluate RAGArena retrieval")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run JSONL relevance evaluation")
    run_parser.add_argument("--dataset", type=Path, required=True)
    run_parser.add_argument("--strategies", default="bm25,dense,hybrid")
    run_parser.add_argument("--top-k", default="5")
    run_parser.add_argument("--output", type=Path, default=Path("reports/retrieval_eval.json"))

    benchmark_parser = subparsers.add_parser("benchmark", help="Run resume-grade benchmark and ablations")
    benchmark_parser.add_argument("--dataset", type=Path, default=Path("data/eval/qa_gold.json"))
    benchmark_parser.add_argument("--plan", type=Path, default=Path("data/eval/ablation_plan.json"))
    benchmark_parser.add_argument("--judge", type=Path, default=Path("data/eval/answer_judge.json"))
    benchmark_parser.add_argument("--top-k", default="1,3,5,10")
    benchmark_parser.add_argument("--output", type=Path, default=Path("reports/evaluation/benchmark.json"))

    parser.add_argument("--qa", type=Path, default=Path("data/eval/qa_gold.json"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--mode", choices=STRATEGY_CHOICES, default="dense")
    parser.add_argument("--chunking-strategy", choices=["agentic", "fixed"], default=None)
    args = parser.parse_args()

    if args.command == "run":
        relevance_report = run_retrieval_evaluation(
            dataset_path=args.dataset,
            strategies=parse_csv(args.strategies),
            top_k_values=[int(value) for value in parse_csv(args.top_k)],
            output_path=args.output,
        )
        print(f"dataset: {relevance_report['dataset']}")
        print(f"case_count: {relevance_report['case_count']}")
        print(f"output: {args.output}")
        print(f"markdown: {args.output.with_suffix('.md')}")
        return

    if args.command == "benchmark":
        benchmark_report = run_benchmark(
            dataset_path=args.dataset,
            plan_path=args.plan,
            judge_path=args.judge,
            top_k_values=[int(value) for value in parse_csv(args.top_k)],
            output_path=args.output,
        )
        print(f"dataset: {benchmark_report['dataset']}")
        print(f"case_count: {benchmark_report['case_count']}")
        print(f"variants: {len(benchmark_report['variants'])}")
        print(f"output: {args.output}")
        print(f"markdown: {args.output.with_suffix('.md')}")
        return

    items = load_eval_dataset(args.qa)
    qa_report = evaluate_retrieval(
        items,
        top_k=args.top_k,
        mode=args.mode,
        chunking_strategy=args.chunking_strategy,
    )

    print("Per-query result:")
    print()
    for row in qa_report.per_query:
        print(f"Query: {row.query}")
        print(f"Gold: {', '.join(row.gold_sections)}")
        print("Top sections:")
        for index, section in enumerate(row.top_sections, start=1):
            print(f"{index}. {section or 'unknown'}")
        print(f"hit@1: {row.hit_at_1}")
        print(f"hit@3: {row.hit_at_3}")
        print(f"hit@5: {row.hit_at_5}")
        print(f"rank: {row.hit_rank if row.hit_rank is not None else 'miss'}")
        print()

    print("Summary:")
    print(f"total_queries: {qa_report.total_queries}")
    print(f"recall@1: {qa_report.recall_at_1:.2f}")
    print(f"recall@3: {qa_report.recall_at_3:.2f}")
    print(f"recall@5: {qa_report.recall_at_5:.2f}")
    print(f"mrr: {qa_report.mrr:.2f}")
    print()
    print("Per-paper:")
    for paper_key, metrics in qa_report.per_paper.items():
        print(
            f"{paper_key}: total={metrics['total_queries']} "
            f"recall@1={metrics['recall@1']:.2f} "
            f"recall@3={metrics['recall@3']:.2f} "
            f"recall@5={metrics['recall@5']:.2f} "
            f"mrr={metrics['mrr']:.2f}"
        )


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]
