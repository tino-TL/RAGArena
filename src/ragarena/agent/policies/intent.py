from __future__ import annotations

import re

LOCAL_KNOWLEDGE_TERMS = (
    "langchain",
    "langgraph",
    "rag",
    "retrieval",
    "ragarena",
    "知识库",
    "文档",
    "检索",
)

CURRENT_OR_EXTERNAL_TERMS = (
    "today",
    "current",
    "latest",
    "news",
    "weather",
    "price",
    "schedule",
    "现在",
    "今天",
    "最新",
    "新闻",
    "实时",
    "天气",
    "价格",
    "日程",
)

DIRECT_ANSWER_PATTERNS = (
    re.compile(r"^\s*你好[吗呀]?\s*$", re.IGNORECASE),
    re.compile(r"^\s*hello\s*$", re.IGNORECASE),
    re.compile(r"^\s*hi\s*$", re.IGNORECASE),
    re.compile(r"^\s*\d+\s*[+\-*/]\s*\d+\s*$"),
)


def is_local_knowledge_query(query: str) -> bool:
    query_lower = query.lower()
    return any(term in query_lower for term in LOCAL_KNOWLEDGE_TERMS)


def is_current_or_external_query(query: str) -> bool:
    query_lower = query.lower()
    return any(term in query_lower for term in CURRENT_OR_EXTERNAL_TERMS)


def is_direct_answer_query(query: str) -> bool:
    return any(pattern.match(query) for pattern in DIRECT_ANSWER_PATTERNS)
