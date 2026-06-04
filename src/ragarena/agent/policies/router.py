from __future__ import annotations

from dataclasses import dataclass
import json
import re

from ragarena.agent.policies.intent import (
    is_direct_answer_query,
    is_local_knowledge_query,
)
from ragarena.config import settings
from ragarena.runtime import get_ollama_decision_generator

RouteType = str
VALID_ROUTES = {"local_rag", "direct_answer"}


@dataclass(frozen=True)
class RouteDecision:
    route: RouteType
    confidence: float
    reason: str


ROUTER_SYSTEM_PROMPT = """You are the router for an agentic RAG system.
Return JSON only with:
route: one of ["local_rag", "direct_answer"]
confidence: number from 0.0 to 1.0
reason: short snake_case string

Use local_rag for research-paper questions, project knowledge, RAG, retrieval, LangGraph, LangChain, methods, citations, or anything that should consult the local knowledge base.
Use direct_answer only for greetings, small talk, and trivial stable questions that do not need retrieval.
There is no web-search route. Current-news or external realtime questions must still route to local_rag so the grader can reject insufficient local evidence.
"""

ROUTER_USER_PROMPT = """Route this user query:
{query}
"""


def route_query(query: str) -> RouteDecision:
    if settings.agent_decision_enabled:
        decision = route_with_qwen(query)
        if decision is not None:
            return decision
    return fallback_route(query)


def route_with_qwen(query: str) -> RouteDecision | None:
    generator = get_ollama_decision_generator(
        settings.ollama_url,
        settings.agent_decision_model,
        settings.agent_decision_timeout,
        settings.ollama_keep_alive,
    )
    try:
        result = generator.generate(
            ROUTER_USER_PROMPT.format(query=query),
            system_prompt=ROUTER_SYSTEM_PROMPT,
        )
    except Exception:
        return None

    parsed = parse_route(result.answer)
    if parsed.reason.startswith("llm_"):
        return None
    return RouteDecision(parsed.route, parsed.confidence, f"qwen:{parsed.reason}")


def parse_route(output: str) -> RouteDecision:
    text = output.strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = None

    if isinstance(value, dict):
        route = str(value.get("route") or "").strip().lower()
        if route not in VALID_ROUTES:
            return RouteDecision("local_rag", 0.5, "llm_unsupported_route")
        confidence = _clamp_float(value.get("confidence"), default=0.75)
        reason = str(value.get("reason") or "router_decision")
        return RouteDecision(route, confidence, reason)

    match = re.search(r"\b(local_rag|direct_answer)\b", text.lower())
    if not match:
        return RouteDecision("local_rag", 0.5, "llm_output_unparseable")
    return RouteDecision(match.group(1), 0.7, "router_text_match")


def fallback_route(query: str) -> RouteDecision:
    if is_direct_answer_query(query):
        return RouteDecision("direct_answer", 0.9, "fallback_direct_answer_pattern")

    if is_local_knowledge_query(query):
        return RouteDecision("local_rag", 0.9, "fallback_local_knowledge_terms")

    return RouteDecision("local_rag", 0.6, "fallback_default_local_rag")


def _clamp_float(value: object, *, default: float) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, number))
