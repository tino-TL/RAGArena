from __future__ import annotations

import sys

from ragarena.retrieval.search import SearchResponse


def safe_print(value: object = "") -> None:
    text = str(value)
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or "utf-8"
        print(text.encode(encoding, errors="replace").decode(encoding, errors="replace"))


def print_search_response(response: SearchResponse) -> None:
    safe_print(f"query: {response.query}")
    safe_print(f"mode: {response.mode}")
    safe_print(f"top_k: {response.top_k}")
    safe_print()

    print_retrieved_chunks(response)


def print_retrieved_chunks(response: SearchResponse, *, title: str | None = None) -> None:
    if title:
        safe_print(title)
        safe_print("=" * len(title))
    for index, result in enumerate(response.results, start=1):
        safe_print(
            f"{index}. score={result.score:.4f} "
            f"source_scores={result.source_scores} "
            f"chunk_id={result.chunk_id} "
            f"document_id={result.document_id} "
            f"model_name={result.model_name}"
        )
        if result.metadata:
            safe_print(f"metadata={result.metadata}")
        safe_print(result.content)
        safe_print()


def print_trace_summary(summary: dict[str, object]) -> None:
    safe_print("Agentic Trace")
    safe_print("=============")
    safe_print(f"trace_id: {summary.get('trace_id')}")
    safe_print(f"trace_url: {summary.get('trace_url')}")
    safe_print(f"route: {summary.get('route')}")
    safe_print(f"route_confidence: {summary.get('route_confidence')}")
    safe_print(f"route_reason: {summary.get('route_reason')}")
    guardrail_reason = summary.get("guardrail_reason")
    if guardrail_reason:
        safe_print(f"guardrail_reason: {guardrail_reason}")
    safe_print(f"strategy: {summary.get('strategy')}")
    safe_print(f"used_hyde: {summary.get('used_hyde')}")
    safe_print(f"rerank_attempted: {summary.get('rerank_attempted')}")
    safe_print(f"rerank_succeeded: {summary.get('rerank_succeeded')}")
    safe_print(f"used_rerank: {summary.get('used_rerank')}")
    fallback = summary.get("rerank_fallback_reason")
    if fallback:
        safe_print(f"rerank_fallback_reason: {fallback}")
    safe_print(f"rewrite_count: {summary.get('rewrite_count')}")
    rewrite_reason = summary.get("rewrite_reason")
    if rewrite_reason:
        safe_print(f"rewrite_reason: {rewrite_reason}")
    safe_print(f"grader_decision: {summary.get('grader_decision')}")
    safe_print(f"grader_score: {summary.get('grader_score')}")
    grader_reason = summary.get("grader_reason")
    if grader_reason:
        safe_print(f"grader_reason: {grader_reason}")
    safe_print("top_retrieved_chunks:")
    chunks = summary.get("retrieved_chunks", [])
    if isinstance(chunks, list):
        for index, chunk in enumerate(chunks[:5], start=1):
            if not isinstance(chunk, dict):
                continue
            safe_print(
                f"{index}. chunk_id={chunk.get('chunk_id')} "
                f"section={chunk.get('section_name')} "
                f"score={chunk.get('score')} "
                f"source_scores={chunk.get('source_scores')} "
                f"preview={chunk.get('content_preview')}"
            )
