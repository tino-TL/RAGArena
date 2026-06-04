from __future__ import annotations

import json
import re

from ragarena.config import settings
from ragarena.runtime import get_ollama_decision_generator

REWRITER_SYSTEM_PROMPT = """You rewrite failed retrieval queries for an agentic RAG system.
Return JSON only with:
rewritten_query: string
reason: short snake_case string

The rewritten query must be concise and optimized for hybrid BM25 + vector retrieval.
Keep important entities, section numbers, paper terms, methods, metrics, and domain words.
Do not answer the question.
"""

REWRITER_PROMPT_TEMPLATE = """Original user question:
{query}
{context}
"""

REWRITE_PREFIX_PATTERN = re.compile(
    r"^(改写后的查询|改写查询|重写后的查询|rewritten query|query)\s*[:：]\s*",
    flags=re.IGNORECASE,
)
QUOTE_CHARS = "\"'`“”‘’"


def rewrite_query(
    original_query: str,
    current_query: str | None = None,
    grade_reason: str | None = None,
    retrieved_doc_previews: list[str] | None = None,
    rewrite_count: int = 0,
) -> str:
    original_query = original_query.strip()
    if not original_query:
        return current_query or "query"

    if settings.agent_decision_enabled:
        rewritten = rewrite_with_qwen(
            original_query,
            current_query=current_query,
            grade_reason=grade_reason,
            retrieved_doc_previews=retrieved_doc_previews,
            rewrite_count=rewrite_count,
        )
        if rewritten:
            return rewritten

    return current_query or original_query


def rewrite_with_qwen(
    original_query: str,
    *,
    current_query: str | None = None,
    grade_reason: str | None = None,
    retrieved_doc_previews: list[str] | None = None,
    rewrite_count: int = 0,
) -> str | None:
    generator = get_ollama_decision_generator(
        settings.ollama_url,
        settings.agent_decision_model,
        settings.agent_decision_timeout,
        settings.ollama_keep_alive,
    )
    try:
        result = generator.generate(
            build_rewrite_prompt(
                original_query,
                current_query=current_query,
                grade_reason=grade_reason,
                retrieved_doc_previews=retrieved_doc_previews,
                rewrite_count=rewrite_count,
            ),
            system_prompt=REWRITER_SYSTEM_PROMPT,
        )
    except Exception:
        return None

    rewritten = parse_rewrite(result.answer)
    if not rewritten:
        return None
    if current_query and rewritten.strip().lower() == current_query.strip().lower():
        return None
    return rewritten


def build_rewrite_prompt(
    query: str,
    *,
    current_query: str | None = None,
    grade_reason: str | None = None,
    retrieved_doc_previews: list[str] | None = None,
    rewrite_count: int = 0,
) -> str:
    context = ""
    if current_query and current_query != query:
        context += f"\nCurrent failed query:\n{current_query}\n"
    if grade_reason:
        context += f"\nWhy retrieval was insufficient:\n{grade_reason}\n"
    if retrieved_doc_previews:
        previews = "\n".join(f"- {preview}" for preview in retrieved_doc_previews[:3])
        context += f"\nRetrieved document previews:\n{previews}\n"
    context += f"\nRewrite attempt: {rewrite_count + 1}\n"
    return REWRITER_PROMPT_TEMPLATE.format(query=query, context=context)


def parse_rewrite(output: str) -> str:
    text = output.strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        value = payload.get("rewritten_query") or payload.get("query")
        if value:
            return clean_rewritten_query(str(value))
    return clean_rewritten_query(text)


def clean_rewritten_query(output: str) -> str:
    cleaned = output.strip().strip(QUOTE_CHARS)
    cleaned = REWRITE_PREFIX_PATTERN.sub("", cleaned)
    return cleaned.strip().strip(QUOTE_CHARS)
