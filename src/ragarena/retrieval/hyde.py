from __future__ import annotations

from ragarena.config import settings
from ragarena.runtime import get_ollama_decision_generator

HYDE_SYSTEM_PROMPT = """You are a local RAG retrieval expansion assistant.
Generate only a short hypothetical passage that could appear in a relevant knowledge-base document.
Do not answer the user, do not explain, and do not invent concrete sources, ids, authors, or dates.
"""

HYDE_PROMPT_TEMPLATE = """Generate a short HyDE hypothetical document passage for semantic retrieval.

Requirements:
- Output only the hypothetical document text.
- Keep it within 120 Chinese characters or 80 English words.
- Preserve the key terms from the user query.
- Do not output JSON or Markdown.

User query: {query}
"""


def generate_hypothetical_document(query: str) -> str | None:
    generator = get_ollama_decision_generator(
        settings.ollama_url,
        settings.agent_decision_model,
        settings.agent_decision_timeout,
        settings.ollama_keep_alive,
    )

    prompt = HYDE_PROMPT_TEMPLATE.format(query=query)
    try:
        result = generator.generate(
            prompt,
            system_prompt=HYDE_SYSTEM_PROMPT,
            json_mode=False,
        )
    except Exception:
        return None

    document = result.answer.strip().strip("\"'")
    return document or None


def build_hyde_search_text(query: str) -> str:
    hypothetical_document = generate_hypothetical_document(query)
    if not hypothetical_document:
        return query
    return f"{query}\n\n{hypothetical_document}"
