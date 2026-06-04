from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, cast

from ragarena.agent.policies.intent import is_current_or_external_query
from ragarena.config import settings
from ragarena.runtime import get_ollama_decision_generator

QUESTION_STOPWORDS = {
    "about",
    "chapter",
    "discuss",
    "discussed",
    "does",
    "explain",
    "section",
    "what",
    "which",
}


@dataclass(frozen=True)
class GradeDecision:
    relevant: bool
    sufficient: bool
    score: float
    reason: str
    useful_chunk_ids: list[int]
    suggested_rewrite: str | None = None

    def __bool__(self) -> bool:
        return self.sufficient


GRADER_SYSTEM_PROMPT = """You are a strict but practical document grader for an agentic RAG workflow.
Return JSON only with these fields:
relevant: boolean
sufficient: boolean
score: number from 0.0 to 1.0
reason: short snake_case string
useful_chunk_ids: list of 1-based document indexes
suggested_rewrite: string or null

relevant means at least one document is related to the question.
sufficient means the documents are enough to answer the question without unsupported claims.
If the context directly contains a concise answer, mark sufficient=true. Do not demand broader background that the user did not ask for.
Use section metadata as evidence when the question asks about a specific section.
If documents are relevant but insufficient, provide a better suggested_rewrite.
"""

GRADER_PROMPT_TEMPLATE = """
Question:
{query}

Retrieved documents:
{context}
"""

def grade_documents(
    query: str,
    documents: list[str],
    retrieved_chunks: list[dict[str, object]] | None = None,
) -> GradeDecision:
    if not documents:
        return GradeDecision(False, False, 0.0, "no_documents", [], query)

    if settings.agent_decision_enabled:
        decision = grade_with_qwen(query, documents, retrieved_chunks=retrieved_chunks)
        if decision is not None:
            return decision

    return fallback_grade(query, documents, retrieved_chunks=retrieved_chunks)


def grade_with_qwen(
    query: str,
    documents: list[str],
    retrieved_chunks: list[dict[str, object]] | None = None,
) -> GradeDecision | None:
    generator = get_ollama_decision_generator(
        settings.ollama_url,
        settings.agent_decision_model,
        settings.agent_decision_timeout,
        settings.ollama_keep_alive,
    )
    try:
        result = generator.generate(
            build_grader_prompt(query, documents, retrieved_chunks=retrieved_chunks),
            system_prompt=GRADER_SYSTEM_PROMPT,
        )
    except Exception:
        return None

    parsed = parse_grade(result.answer, document_count=len(documents))
    if parsed is None:
        return None
    return GradeDecision(
        relevant=parsed.relevant,
        sufficient=parsed.sufficient,
        score=parsed.score,
        reason=f"qwen:{parsed.reason}",
        useful_chunk_ids=parsed.useful_chunk_ids,
        suggested_rewrite=parsed.suggested_rewrite,
    )


def build_grader_prompt(
    query: str,
    documents: list[str],
    retrieved_chunks: list[dict[str, object]] | None = None,
) -> str:
    context_parts = []
    for index, document in enumerate(documents, start=1):
        section_name = section_name_for_index(index, retrieved_chunks)
        header = f"[Document {index}]"
        if section_name:
            header += f"\nSection: {section_name}"
        context_parts.append(f"{header}\n{document}")
    context = "\n\n".join(context_parts)
    return GRADER_PROMPT_TEMPLATE.format(query=query, context=context)


def parse_grade(output: str, *, document_count: int | None = None) -> GradeDecision | None:
    parsed_json = parse_grade_json(output, document_count=document_count)
    if parsed_json is not None:
        return parsed_json

    match = re.match(r"^(yes|no)\b", output.strip().lower())
    if not match:
        return None

    relevant = match.group(1) == "yes"
    return GradeDecision(
        relevant=relevant,
        sufficient=relevant,
        score=1.0 if relevant else 0.0,
        reason="legacy_yes_no_output",
        useful_chunk_ids=list(range(1, (document_count or 0) + 1)) if relevant else [],
    )


def parse_grade_json(output: str, *, document_count: int | None = None) -> GradeDecision | None:
    text = output.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict):
        return None
    useful_ids = value.get("useful_chunk_ids", [])
    if not isinstance(useful_ids, list):
        useful_ids = []
    max_id = document_count or len(useful_ids)
    clean_ids = [
        chunk_id
        for chunk_id in (to_int(item) for item in useful_ids)
        if chunk_id is not None and 1 <= chunk_id <= max_id
    ]
    relevant = bool(value.get("relevant", False))
    sufficient = bool(value.get("sufficient", relevant))
    score = to_float(value.get("score"), default=1.0 if sufficient else 0.0)
    suggested_rewrite = value.get("suggested_rewrite")
    return GradeDecision(
        relevant=relevant,
        sufficient=sufficient,
        score=max(0.0, min(1.0, score)),
        reason=str(value.get("reason") or "llm_structured_grade"),
        useful_chunk_ids=clean_ids,
        suggested_rewrite=str(suggested_rewrite).strip() if suggested_rewrite else None,
    )


def fallback_grade(
    query: str,
    documents: list[str],
    retrieved_chunks: list[dict[str, object]] | None = None,
) -> GradeDecision:
    query_lower = query.lower()
    context_lower = "\n".join(documents).lower()

    if is_current_or_external_query(query):
        return GradeDecision(False, False, 0.0, "query_requires_current_information", [], query)

    section_match = section_match_grade(query, documents, retrieved_chunks)
    if section_match is not None:
        return section_match

    if "langgraph" in query_lower and "langgraph" in context_lower:
        return GradeDecision(True, True, 0.9, "langgraph_term_overlap", list(range(1, len(documents) + 1)))

    if "langchain" in query_lower and "langchain" in context_lower:
        return GradeDecision(True, True, 0.9, "langchain_term_overlap", list(range(1, len(documents) + 1)))

    query_terms = extract_query_terms(query)
    if not query_terms:
        return GradeDecision(False, False, 0.0, "no_query_terms", [], query)

    overlap = sum(1 for term in query_terms if term in context_lower)
    threshold = max(1, min(2, len(query_terms)))
    relevant = overlap >= threshold
    score = overlap / max(1, len(query_terms))
    return GradeDecision(
        relevant=relevant,
        sufficient=relevant,
        score=max(0.0, min(1.0, score)),
        reason=f"signal_overlap={overlap}/{len(query_terms)}",
        useful_chunk_ids=list(range(1, len(documents) + 1)) if relevant else [],
        suggested_rewrite=query if not relevant else None,
    )


def section_match_grade(
    query: str,
    documents: list[str],
    retrieved_chunks: list[dict[str, object]] | None,
) -> GradeDecision | None:
    candidates = extract_section_candidates(query)
    if not candidates:
        return None

    useful_ids = []
    for index, document in enumerate(documents, start=1):
        section_name = section_name_for_index(index, retrieved_chunks)
        haystacks = [section_name.lower(), document[:300].lower()]
        if any(candidate in haystack for candidate in candidates for haystack in haystacks):
            useful_ids.append(index)

    if not useful_ids:
        return None

    non_empty = any(documents[index - 1].strip() for index in useful_ids)
    return GradeDecision(
        relevant=True,
        sufficient=non_empty,
        score=0.95 if non_empty else 0.7,
        reason="section_metadata_match" if non_empty else "section_metadata_match_empty_content",
        useful_chunk_ids=useful_ids,
        suggested_rewrite=None if non_empty else query,
    )


def extract_query_terms(query: str) -> set[str]:
    terms = set()
    for match in re.finditer(r"\d+(?:\.\d+)*|[a-zA-Z][a-zA-Z0-9]*|[\u4e00-\u9fff]+", query):
        token = match.group(0)
        lowered = token.lower()
        if lowered in QUESTION_STOPWORDS:
            continue
        if re.fullmatch(r"\d+(?:\.\d+)*", lowered):
            terms.add(lowered)
        elif token.isupper() and len(token) >= 2:
            terms.add(lowered)
        elif len(lowered) >= 3:
            terms.add(lowered)
    return terms


def extract_section_candidates(query: str) -> set[str]:
    lowered = " ".join(query.lower().split())
    candidates = set()
    for match in re.finditer(
        r"\b(?:section|chapter)\s+(\d+(?:\.\d+)*)(?:\s+([a-zA-Z][a-zA-Z0-9]*(?:\s+[a-zA-Z][a-zA-Z0-9]*){0,4}))?",
        lowered,
    ):
        number = match.group(1)
        title = clean_section_tail(match.group(2) or "")
        candidates.add(number)
        if title:
            candidates.add(title)
            candidates.add(f"{number} {title}")

    for name in ("abstract", "conclusion", "introduction"):
        if name in lowered:
            candidates.add(name)
    return candidates


def clean_section_tail(value: str) -> str:
    tokens = [token for token in value.split() if token not in QUESTION_STOPWORDS]
    return " ".join(tokens)


def section_name_for_index(
    index: int,
    retrieved_chunks: list[dict[str, object]] | None,
) -> str:
    if not retrieved_chunks or index > len(retrieved_chunks):
        return ""
    chunk = retrieved_chunks[index - 1]
    section_name = chunk.get("section_name")
    if section_name:
        return str(section_name)
    metadata = chunk.get("metadata")
    if isinstance(metadata, dict) and metadata.get("section_name"):
        return str(metadata["section_name"])
    return ""


def to_int(value: object) -> int | None:
    try:
        return int(cast(Any, value))
    except (TypeError, ValueError):
        return None


def to_float(value: object, *, default: float) -> float:
    try:
        return float(cast(Any, value))
    except (TypeError, ValueError):
        return default
