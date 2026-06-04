from __future__ import annotations

import asyncpg
import requests
from fastapi import APIRouter

from ragarena.config import settings
from ragarena.runtime import get_redis_cache

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict[str, object]:
    services = {
        "postgres": await check_postgres(),
        "elasticsearch": check_http_service(settings.elasticsearch_url),
        "ollama": check_http_service(settings.ollama_url, "/api/tags"),
        "redis": check_redis(),
    }

    return {
        "status": "ok",
        "services": services,
    }


async def check_postgres() -> dict[str, object]:
    try:
        conn = await asyncpg.connect(settings.postgres_dsn, timeout=3)
        try:
            await conn.execute("SELECT 1")
        finally:
            await conn.close()
        return {"ok": True, "error": None}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def check_http_service(url: str, path: str = "/") -> dict[str, object]:
    try:
        response = requests.get(f"{url.rstrip('/')}{path}", timeout=3)
        response.raise_for_status()
        return {"ok": True, "error": None}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def check_redis() -> dict[str, object]:
    try:
        get_redis_cache(settings.redis_url).ping()
        return {"ok": True, "error": None}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
