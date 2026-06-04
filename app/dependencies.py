from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from uuid import uuid4

from fastapi import HTTPException


@dataclass(frozen=True)
class RequestContext:
    request_id: str
    started_at: float


def get_request_context() -> RequestContext:
    return RequestContext(request_id=str(uuid4()), started_at=perf_counter())


def latency_ms(context: RequestContext) -> float:
    return round((perf_counter() - context.started_at) * 1000, 2)


def raise_api_error(
    *,
    code: str,
    message: str,
    context: RequestContext,
    status_code: int = 503,
    retryable: bool = True,
) -> None:
    raise HTTPException(
        status_code=status_code,
        detail={
            "error": {
                "code": code,
                "message": message,
                "retryable": retryable,
            },
            "request_id": context.request_id,
        },
    )
