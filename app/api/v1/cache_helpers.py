from __future__ import annotations

from typing import Any

from ragarena.config import settings
from ragarena.runtime import get_redis_cache


def get_cache_json(key: str) -> dict[str, Any] | None:
    try:
        return get_redis_cache(settings.redis_url).get_json(key)
    except Exception:
        return None


def set_cache_json(key: str, value: dict[str, Any]) -> None:
    try:
        get_redis_cache(settings.redis_url).set_json(
            key,
            value,
            ttl_seconds=settings.rag_cache_ttl_seconds,
        )
    except Exception:
        return


def append_feedback(value: dict[str, Any]) -> bool:
    try:
        get_redis_cache(settings.redis_url).append_json("ragarena:feedback", value)
    except Exception:
        return False
    return True
