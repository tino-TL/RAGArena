from __future__ import annotations

from ragarena.retrieval.vector_store import SearchResult

RAG_PROMPT_TEMPLATE = """你是 RAGArena 的本地 RAG 助手。请只根据给定上下文回答问题。
如果上下文不足以回答，请明确说明“根据当前资料无法确定”。

问题：
{query}

上下文：
{context}

回答要求：
1. 用中文回答。
2. 优先引用上下文中的事实。
3. 不要编造上下文之外的信息。
"""


def build_rag_prompt(query: str, chunks: list[SearchResult]) -> str:
    return RAG_PROMPT_TEMPLATE.format(query=query, context=format_context(chunks))


def format_context(chunks: list[SearchResult]) -> str:
    if not chunks:
        return "无可用上下文。"

    parts = []
    for index, chunk in enumerate(chunks, start=1):
        parts.append(
            f"[Chunk {index} | chunk_id={chunk.chunk_id} | document_id={chunk.document_id}]\n"
            f"{chunk.content}"
        )

    return "\n\n".join(parts)
