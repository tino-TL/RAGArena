from __future__ import annotations

import json
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any

from ragarena.evaluation.metrics import evaluate_ranked_results
from ragarena.generation.service import generate_answer
from ragarena.retrieval.search import SearchStrategy, search_chunks
from ragarena.retrieval.vector_store import SearchResult


DEFAULT_TOP_K_VALUES = [1, 3, 5, 10]


@dataclass(frozen=True)
class GoldQA:
    id: str
    query: str
    expected_answer: str = ""
    category: str = "unknown"
    tags: list[str] = field(default_factory=list)
    difficulty: str | None = None
    paper_id: int | None = None
    arxiv_id: str | None = None
    gold_sections: list[str] = field(default_factory=list)
    gold_keywords: list[str] = field(default_factory=list)
    gold_chunk_ids: set[int] = field(default_factory=set)
    gold_chunk_ids_by_strategy: dict[str, set[int]] = field(default_factory=dict)
    gold_document_ids: set[int] = field(default_factory=set)
    gold_page_numbers: set[int] = field(default_factory=set)
    gold_visual_refs: list[str] = field(default_factory=list)
    answer_must_include: list[str] = field(default_factory=list)
    answer_must_not_include: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass(frozen=True)
class JudgeRecord:
    qa_id: str
    variant: str
    answer_correct: bool
    citation_correct: bool
    unsupported_claim: bool = False
    score: float | None = None
    notes: str = ""


@dataclass(frozen=True)
class VariantConfig:
    name: str
    strategy: SearchStrategy = "hybrid_hyde_rerank"
    chunking_strategy: str | None = None
    index_name: str | None = None
    generate: bool = False
    include_categories: set[str] = field(default_factory=set)
    include_tags: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class CaseResult:
    qa_id: str
    query: str
    category: str
    tags: list[str]
    status: str
    retrieval_metrics: dict[str, dict[str, float]]
    source_metrics: dict[str, bool]
    answer_metrics: dict[str, bool | float | None]
    latency_ms: float
    retrieved_chunk_ids: list[int]
    retrieved_document_ids: list[int]
    retrieved_sections: list[str | None]
    retrieved_chunk_types: list[str | None]
    answer: str | None = None
    error: str | None = None


def load_gold_qa(path: str | Path) -> list[GoldQA]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Gold QA dataset must be a JSON list")
    return [parse_gold_item(item, index) for index, item in enumerate(payload, start=1)]


def parse_gold_item(item: dict[str, Any], index: int) -> GoldQA:
    query = str(item.get("query") or "").strip()
    if not query:
        raise ValueError(f"Gold QA item {index} is missing query")
    return GoldQA(
        id=str(item.get("id") or f"qa-{index:04d}"),
        query=query,
        expected_answer=str(item.get("expected_answer") or ""),
        category=str(item.get("category") or "unknown"),
        tags=[str(value) for value in item.get("tags", [])],
        difficulty=item.get("difficulty"),
        paper_id=int(item["paper_id"]) if item.get("paper_id") is not None else None,
        arxiv_id=str(item["arxiv_id"]) if item.get("arxiv_id") is not None else None,
        gold_sections=[str(value) for value in item.get("gold_sections", [])],
        gold_keywords=[str(value) for value in item.get("gold_keywords", [])],
        gold_chunk_ids={int(value) for value in item.get("gold_chunk_ids", [])},
        gold_chunk_ids_by_strategy={
            str(strategy): {int(value) for value in values}
            for strategy, values in dict(item.get("gold_chunk_ids_by_strategy", {})).items()
        },
        gold_document_ids={int(value) for value in item.get("gold_document_ids", [])},
        gold_page_numbers={int(value) for value in item.get("gold_page_numbers", [])},
        gold_visual_refs=[str(value) for value in item.get("gold_visual_refs", [])],
        answer_must_include=[str(value) for value in item.get("answer_must_include", [])],
        answer_must_not_include=[str(value) for value in item.get("answer_must_not_include", [])],
        notes=str(item.get("notes") or ""),
    )


def load_judge_records(path: str | Path) -> dict[tuple[str, str], JudgeRecord]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Answer judge file must be a JSON list")
    records = {}
    for item in payload:
        record = JudgeRecord(
            qa_id=str(item["qa_id"]),
            variant=str(item["variant"]),
            answer_correct=bool(item["answer_correct"]),
            citation_correct=bool(item["citation_correct"]),
            unsupported_claim=bool(item.get("unsupported_claim", False)),
            score=float(item["score"]) if item.get("score") is not None else None,
            notes=str(item.get("notes") or ""),
        )
        records[(record.qa_id, record.variant)] = record
    return records


def run_benchmark(
    *,
    dataset_path: Path,
    plan_path: Path,
    output_path: Path,
    judge_path: Path | None = None,
    top_k_values: list[int] | None = None,
) -> dict[str, Any]:
    top_k_values = top_k_values or DEFAULT_TOP_K_VALUES
    gold_items = load_gold_qa(dataset_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    judges = load_judge_records(judge_path) if judge_path and judge_path.exists() else {}
    variants = variants_from_plan(plan)

    reports = []
    for variant in variants:
        cases = [item for item in gold_items if case_matches_variant(item, variant)]
        case_results = [evaluate_variant_case(item, variant, top_k_values, judges) for item in cases]
        reports.append(
            {
                "variant": serialize_variant(variant),
                "case_count": len(case_results),
                "cases": [asdict(result) for result in case_results],
                "summary": summarize_case_results(case_results, top_k_values),
            }
        )

    report = {
        "dataset": str(dataset_path),
        "plan": str(plan_path),
        "case_count": len(gold_items),
        "top_k": top_k_values,
        "variants": reports,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    output_path.with_suffix(".md").write_text(render_benchmark_markdown(report), encoding="utf-8")
    return report


def variants_from_plan(plan: dict[str, Any]) -> list[VariantConfig]:
    variants: list[VariantConfig] = []
    for experiment in plan.get("experiments", []):
        subset = experiment.get("subset_filter", {})
        include_categories = {str(value) for value in subset.get("categories", [])}
        include_tags = {str(value) for value in subset.get("tags", [])}
        for raw_variant in experiment.get("variants", []):
            strategy = raw_variant.get("strategy") or infer_strategy(raw_variant)
            variants.append(
                VariantConfig(
                    name=f"{experiment['id']}::{raw_variant['name']}",
                    strategy=strategy,
                    chunking_strategy=raw_variant.get("chunk_strategy"),
                    index_name=raw_variant.get("index_name"),
                    generate=bool(raw_variant.get("generate", False)),
                    include_categories=include_categories,
                    include_tags=include_tags,
                )
            )
    return variants


def serialize_variant(variant: VariantConfig) -> dict[str, Any]:
    payload = asdict(variant)
    payload["include_categories"] = sorted(variant.include_categories)
    payload["include_tags"] = sorted(variant.include_tags)
    return payload


def infer_strategy(raw_variant: dict[str, Any]) -> SearchStrategy:
    if raw_variant.get("use_hyde") and raw_variant.get("use_rerank"):
        return "hybrid_hyde_rerank"
    if raw_variant.get("use_rerank"):
        return "hybrid_rerank"
    if raw_variant.get("use_hyde"):
        return "hybrid_hyde"
    return "hybrid_hyde_rerank"


def case_matches_variant(item: GoldQA, variant: VariantConfig) -> bool:
    if variant.include_categories and item.category not in variant.include_categories:
        return False
    if variant.include_tags and not variant.include_tags.intersection(item.tags):
        return False
    return True


def evaluate_variant_case(
    item: GoldQA,
    variant: VariantConfig,
    top_k_values: list[int],
    judges: dict[tuple[str, str], JudgeRecord],
) -> CaseResult:
    started_at = perf_counter()
    try:
        retrieval = search_chunks(
            query=item.query,
            top_k=max(top_k_values),
            mode=variant.strategy,
            chunking_strategy=variant.chunking_strategy,
            paper_id=item.paper_id,
            arxiv_id=item.arxiv_id,
            index_name=variant.index_name,
        )
        answer = generate_answer(item.query, retrieval) if variant.generate else None
        latency_ms = round((perf_counter() - started_at) * 1000, 3)
        return CaseResult(
            qa_id=item.id,
            query=item.query,
            category=item.category,
            tags=item.tags,
            status="ok",
            retrieval_metrics=compute_retrieval_metrics(
                item,
                retrieval.results,
                top_k_values,
                chunking_strategy=variant.chunking_strategy,
            ),
            source_metrics=compute_source_metrics(
                item,
                retrieval.results,
                chunking_strategy=variant.chunking_strategy,
            ),
            answer_metrics=compute_answer_metrics(item, variant.name, answer, judges),
            latency_ms=latency_ms,
            retrieved_chunk_ids=[result.chunk_id for result in retrieval.results],
            retrieved_document_ids=[result.document_id for result in retrieval.results],
            retrieved_sections=[result.section_name or result.metadata.get("section_name") for result in retrieval.results],
            retrieved_chunk_types=[chunk_type(result) for result in retrieval.results],
            answer=answer,
        )
    except Exception as exc:
        return CaseResult(
            qa_id=item.id,
            query=item.query,
            category=item.category,
            tags=item.tags,
            status="error",
            retrieval_metrics={},
            source_metrics={},
            answer_metrics={},
            latency_ms=round((perf_counter() - started_at) * 1000, 3),
            retrieved_chunk_ids=[],
            retrieved_document_ids=[],
            retrieved_sections=[],
            retrieved_chunk_types=[],
            error="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        )


def compute_retrieval_metrics(
    item: GoldQA,
    results: list[SearchResult],
    top_k_values: list[int],
    *,
    chunking_strategy: str | None = None,
) -> dict[str, dict[str, float]]:
    relevant_ids = gold_chunk_ids_for_strategy(item, chunking_strategy)
    ranked_ids = [result.chunk_id for result in results]
    if not relevant_ids:
        relevant_ids = item.gold_document_ids
        ranked_ids = [result.document_id for result in results]
    if not relevant_ids:
        return {}
    return {
        str(top_k): asdict(evaluate_ranked_results(ranked_ids, relevant_ids, top_k=top_k))
        for top_k in top_k_values
    }


def compute_source_metrics(
    item: GoldQA,
    results: list[SearchResult],
    *,
    chunking_strategy: str | None = None,
) -> dict[str, bool]:
    gold_chunk_ids = gold_chunk_ids_for_strategy(item, chunking_strategy)
    return {
        "chunk_hit": bool(gold_chunk_ids and any(result.chunk_id in gold_chunk_ids for result in results)),
        "document_hit": bool(item.gold_document_ids and any(result.document_id in item.gold_document_ids for result in results)),
        "section_hit": section_hit(item, results),
        "keyword_hit": keyword_hit(item, results),
        "page_hit": page_hit(item, results),
        "visual_ref_hit": visual_ref_hit(item, results),
        "citation_source_hit": citation_source_hit(
            item,
            results,
            chunking_strategy=chunking_strategy,
        ),
    }


def compute_answer_metrics(
    item: GoldQA,
    variant_name: str,
    answer: str | None,
    judges: dict[tuple[str, str], JudgeRecord],
) -> dict[str, bool | float | None]:
    judge = judges.get((item.id, variant_name))
    answer_text = normalize(answer or "")
    must_include = item.answer_must_include or ([item.expected_answer] if item.expected_answer else [])
    deterministic_correct = bool(answer_text) and all(normalize(term) in answer_text for term in must_include)
    deterministic_safe = not any(normalize(term) in answer_text for term in item.answer_must_not_include)
    return {
        "deterministic_answer_correct": deterministic_correct and deterministic_safe,
        "judge_answer_correct": judge.answer_correct if judge else None,
        "judge_citation_correct": judge.citation_correct if judge else None,
        "unsupported_claim": judge.unsupported_claim if judge else None,
        "judge_score": judge.score if judge else None,
    }


def summarize_case_results(results: list[CaseResult], top_k_values: list[int]) -> dict[str, float | int]:
    ok_results = [result for result in results if result.status == "ok"]
    summary: dict[str, float | int] = {
        "ok_cases": len(ok_results),
        "error_cases": len(results) - len(ok_results),
        "latency_ms_avg": average([result.latency_ms for result in ok_results]),
        "latency_ms_p50": percentile([result.latency_ms for result in ok_results], 50),
        "latency_ms_p95": percentile([result.latency_ms for result in ok_results], 95),
        "citation_source_hit_rate": rate(ok_results, "source_metrics", "citation_source_hit"),
        "section_hit_rate": rate(ok_results, "source_metrics", "section_hit"),
        "keyword_hit_rate": rate(ok_results, "source_metrics", "keyword_hit"),
        "page_hit_rate": rate(ok_results, "source_metrics", "page_hit"),
        "visual_ref_hit_rate": rate(ok_results, "source_metrics", "visual_ref_hit"),
        "deterministic_answer_accuracy": rate(ok_results, "answer_metrics", "deterministic_answer_correct"),
        "judge_answer_accuracy": nullable_rate(ok_results, "answer_metrics", "judge_answer_correct"),
        "judge_citation_accuracy": nullable_rate(ok_results, "answer_metrics", "judge_citation_correct"),
        "unsupported_claim_rate": nullable_rate(ok_results, "answer_metrics", "unsupported_claim"),
    }
    for top_k in top_k_values:
        for metric in ("recall", "mrr", "ndcg", "hit_rate"):
            values = [
                result.retrieval_metrics[str(top_k)][metric]
                for result in ok_results
                if str(top_k) in result.retrieval_metrics
            ]
            summary[f"{metric}@{top_k}"] = average(values)
    return summary


def render_benchmark_markdown(report: dict[str, Any]) -> str:
    top_k_values = report["top_k"]
    metric_headers = [f"Recall@{top_k}" for top_k in top_k_values]
    headers = [
        "Variant",
        "Cases",
        "Errors",
        "P50 ms",
        "P95 ms",
        "Citation Hit",
        "Answer Acc",
        *metric_headers,
    ]
    lines = [
        "# RAGArena Benchmark Report",
        "",
        f"Dataset: `{report['dataset']}`",
        f"Plan: `{report['plan']}`",
        f"Cases: {report['case_count']}",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(headers) - 1)) + " |",
    ]
    for variant in report["variants"]:
        summary = variant["summary"]
        cells = [
            variant["variant"]["name"],
            str(summary["ok_cases"]),
            str(summary["error_cases"]),
            str(summary["latency_ms_p50"]),
            str(summary["latency_ms_p95"]),
            str(summary["citation_source_hit_rate"]),
            str(summary["judge_answer_accuracy"] or summary["deterministic_answer_accuracy"]),
        ]
        cells.extend(str(summary.get(f"recall@{top_k}", 0.0)) for top_k in top_k_values)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def section_hit(item: GoldQA, results: list[SearchResult]) -> bool:
    gold_sections = {normalize(section) for section in item.gold_sections}
    return bool(gold_sections) and any(normalize(section_name(result)) in gold_sections for result in results)


def keyword_hit(item: GoldQA, results: list[SearchResult]) -> bool:
    keywords = [normalize(keyword) for keyword in item.gold_keywords]
    return bool(keywords) and any(
        any(keyword in normalize(result.content) for keyword in keywords) for result in results
    )


def page_hit(item: GoldQA, results: list[SearchResult]) -> bool:
    return bool(item.gold_page_numbers) and any(page_number(result) in item.gold_page_numbers for result in results)


def visual_ref_hit(item: GoldQA, results: list[SearchResult]) -> bool:
    refs = {normalize(ref) for ref in item.gold_visual_refs}
    return bool(refs) and any(refs.intersection({normalize(ref) for ref in visual_refs(result)}) for result in results)


def citation_source_hit(
    item: GoldQA,
    results: list[SearchResult],
    *,
    chunking_strategy: str | None = None,
) -> bool:
    gold_chunk_ids = gold_chunk_ids_for_strategy(item, chunking_strategy)
    source_checks = [
        bool(gold_chunk_ids and any(result.chunk_id in gold_chunk_ids for result in results)),
        bool(item.gold_document_ids and any(result.document_id in item.gold_document_ids for result in results)),
        section_hit(item, results),
        keyword_hit(item, results),
        page_hit(item, results),
        visual_ref_hit(item, results),
    ]
    return any(source_checks)


def gold_chunk_ids_for_strategy(item: GoldQA, chunking_strategy: str | None) -> set[int]:
    if chunking_strategy:
        strategy_ids = item.gold_chunk_ids_by_strategy.get(chunking_strategy)
        if strategy_ids:
            return strategy_ids
    return item.gold_chunk_ids


def section_name(result: SearchResult) -> str:
    return str(result.section_name or result.metadata.get("section_name") or "")


def chunk_type(result: SearchResult) -> str | None:
    value = result.metadata.get("semantic_chunk_type") or result.metadata.get("chunk_type")
    return str(value) if value is not None else None


def page_number(result: SearchResult) -> int | None:
    value = result.metadata.get("page_number")
    return int(value) if value is not None else None


def visual_refs(result: SearchResult) -> list[str]:
    value = result.metadata.get("visual_refs")
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        return [value]
    return []


def normalize(value: str) -> str:
    return " ".join(value.lower().split())


def average(values: list[float]) -> float:
    return round(mean(values), 4) if values else 0.0


def percentile(values: list[float], percent: int) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = min(len(sorted_values) - 1, round((percent / 100) * (len(sorted_values) - 1)))
    return round(sorted_values[index], 3)


def rate(results: list[CaseResult], attr: str, key: str) -> float:
    if not results:
        return 0.0
    return round(sum(1 for result in results if getattr(result, attr).get(key) is True) / len(results), 4)


def nullable_rate(results: list[CaseResult], attr: str, key: str) -> float | None:
    values = [getattr(result, attr).get(key) for result in results]
    observed = [value for value in values if value is not None]
    if not observed:
        return None
    return round(sum(1 for value in observed if value is True) / len(observed), 4)
