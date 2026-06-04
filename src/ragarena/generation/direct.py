from __future__ import annotations

from ragarena.config import settings
from ragarena.runtime import get_deepseek_generator

DIRECT_PROMPT_TEMPLATE = """Answer the user directly and concisely.

User question:
{query}
"""

DIRECT_SYSTEM_PROMPT = (
    "You are RAGArena's direct answer assistant. "
    "Answer concise general questions without using retrieved documents."
)
DIRECT_FALLBACK_MESSAGE = "抱歉，我暂时无法回答该问题。"


def generate_direct_answer(query: str) -> str:
    generator = get_deepseek_generator(settings.deepseek_api_key, settings.deepseek_model)
    if not generator.is_configured():
        return "DEEPSEEK_API_KEY is not configured."

    try:
        result = generator.generate(
            DIRECT_PROMPT_TEMPLATE.format(query=query),
            system_prompt=DIRECT_SYSTEM_PROMPT,
        )
    except Exception:
        return DIRECT_FALLBACK_MESSAGE

    return result.answer or DIRECT_FALLBACK_MESSAGE
