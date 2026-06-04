from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from ragarena.config import settings
from ragarena.papers.downloader import DEFAULT_PAPER_DIR
from ragarena.pipeline.steps import (
    chunk_documents,
    download_papers,
    embed_chunks,
    fetch_and_store_papers,
    index_embeddings,
    parse_papers,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineConfig:
    query: str = "all:retrieval augmented generation"
    max_results: int = 20
    output_dir: Path = DEFAULT_PAPER_DIR
    download: bool = True
    parse: bool = True
    retry_attempts: int = settings.pipeline_retry_attempts
    retry_backoff_seconds: float = settings.pipeline_retry_backoff_seconds
    parser: str = "docling"
    chunk_strategy: str = "agentic"
    debug_planner: bool = False
    planner_provider: str | None = None
    planner_model: str | None = None
    validate_chunks: bool = False


@dataclass
class PipelineStepSummary:
    name: str
    status: str = "pending"
    count: int = 0
    latency_ms: float = 0.0
    error: str | None = None


@dataclass
class PipelineRunSummary:
    run_id: str
    query: str
    max_results: int
    status: str = "running"
    latency_ms: float = 0.0
    steps: list[PipelineStepSummary] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["steps"] = [asdict(step) for step in self.steps]
        return payload


async def run_knowledge_pipeline(config: PipelineConfig) -> PipelineRunSummary:
    run_started_at = perf_counter()
    summary = PipelineRunSummary(
        run_id=str(uuid4()),
        query=config.query,
        max_results=config.max_results,
    )

    steps: list[tuple[str, Callable[[], Awaitable[int]] | None]] = [
        (
            "arxiv_fetch",
            lambda: fetch_and_store_papers(
                query=config.query,
                max_results=config.max_results,
                sync_documents=True,
            ),
        ),
        (
            "download",
            lambda: download_papers(
                limit=config.max_results,
                output_dir=config.output_dir,
            ),
        )
        if config.download
        else ("download", None),
        (
            "parse",
            lambda: parse_papers(
                limit=config.max_results,
                sync_documents=True,
                parser=config.parser,
            ),
        )
        if config.parse
        else ("parse", None),
        (
            "chunk",
            lambda: chunk_documents(
                chunk_strategy=config.chunk_strategy,
                debug_planner=config.debug_planner,
                planner_provider=config.planner_provider,
                planner_model=config.planner_model,
                validate_chunks=config.validate_chunks,
            ),
        ),
        ("embed", embed_chunks),
        ("index", index_embeddings),
    ]

    try:
        for name, operation in steps:
            step_summary = PipelineStepSummary(name=name)
            summary.steps.append(step_summary)
            if operation is None:
                step_summary.status = "skipped"
                continue
            step_summary.status = "started"
            step_started_at = perf_counter()
            step_summary.count = await run_with_retry(
                operation,
                attempts=config.retry_attempts,
                backoff_seconds=config.retry_backoff_seconds,
                step_name=name,
                run_id=summary.run_id,
            )
            step_summary.status = "succeeded"
            step_summary.latency_ms = elapsed_ms(step_started_at)
        summary.status = "succeeded"
    except Exception as exc:
        summary.status = "failed"
        summary.steps[-1].status = "failed"
        summary.steps[-1].error = str(exc)
        logger.exception("pipeline_step_failed", extra={"pipeline_run_id": summary.run_id})
    finally:
        summary.latency_ms = elapsed_ms(run_started_at)

    return summary


async def run_with_retry(
    operation: Callable[[], Awaitable[int]],
    *,
    attempts: int,
    backoff_seconds: float,
    step_name: str,
    run_id: str,
) -> int:
    last_error: Exception | None = None
    for attempt in range(1, max(attempts, 1) + 1):
        try:
            logger.info(
                "pipeline_step_started",
                extra={"pipeline_run_id": run_id, "step": step_name, "attempt": attempt},
            )
            count = await operation()
            logger.info(
                "pipeline_step_succeeded",
                extra={"pipeline_run_id": run_id, "step": step_name, "count": count},
            )
            return count
        except Exception as exc:
            last_error = exc
            logger.warning(
                "pipeline_step_retry",
                extra={
                    "pipeline_run_id": run_id,
                    "step": step_name,
                    "attempt": attempt,
                    "error": str(exc),
                },
            )
            if attempt < attempts:
                await asyncio.sleep(backoff_seconds * attempt)
    assert last_error is not None
    raise last_error


def run_scheduled_once(config: PipelineConfig) -> PipelineRunSummary:
    return asyncio.run(run_knowledge_pipeline(config))


def elapsed_ms(started_at: float) -> float:
    return round((perf_counter() - started_at) * 1000, 3)
