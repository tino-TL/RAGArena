from __future__ import annotations

import json
from hashlib import sha256
from typing import Any

import redis


class RedisCache:
    def __init__(self, url: str, timeout: int = 3) -> None:
        self.client = redis.Redis.from_url(
            url,
            socket_connect_timeout=timeout,
            socket_timeout=timeout,
            decode_responses=True,
        )

    def ping(self) -> bool:
        return bool(self.client.ping())

    def get_json(self, key: str) -> dict[str, Any] | None:
        value = self.client.get(key)
        if not value:
            return None
        return json.loads(value)

    def set_json(self, key: str, value: dict[str, Any], ttl_seconds: int) -> None:
        self.client.set(key, json.dumps(value, ensure_ascii=False), ex=ttl_seconds)

    def append_json(self, key: str, value: dict[str, Any], max_length: int = 1000) -> None:
        payload = json.dumps(value, ensure_ascii=False)
        pipe = self.client.pipeline()
        pipe.lpush(key, payload)
        pipe.ltrim(key, 0, max_length - 1)
        pipe.execute()


def cache_key(namespace: str, payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    digest = sha256(encoded.encode("utf-8")).hexdigest()
    return f"ragarena:{namespace}:{digest}"
