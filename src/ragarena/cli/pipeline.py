from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from apscheduler.schedulers.blocking import BlockingScheduler
from dotenv import load_dotenv

from ragarena.config import settings
from ragarena.logging import configure_logging
from ragarena.papers.downloader import DEFAULT_PAPER_DIR
from ragarena.pipeline.runner import (
    PipelineConfig,
    run_knowledge_pipeline,
    run_scheduled_once,
)
from ragarena.runtime import get_elasticsearch_vector_store


def main() -> None:
    load_dotenv()
    configure_logging()

    parser = argparse.ArgumentParser(description="Run the RAGArena knowledge pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    add_pipeline_args(run_parser)
    run_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")

    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--json", action="store_true")

    schedule_parser = subparsers.add_parser("schedule")
    add_pipeline_args(schedule_parser)
    schedule_parser.add_argument("--interval-minutes", type=int, default=settings.pipeline_schedule_interval_minutes)
    schedule_parser.add_argument("--cron", default=settings.pipeline_schedule_cron)

    args = parser.parse_args()
    if args.command == "run":
        summary = asyncio.run(run_knowledge_pipeline(config_from_args(args)))
        print_summary(summary.to_dict(), as_json=args.json)
        raise SystemExit(0 if summary.status == "succeeded" else 1)
    if args.command == "status":
        payload = pipeline_status()
        print_summary(payload, as_json=args.json)
        return
    if args.command == "schedule":
        schedule_pipeline(args)


def add_pipeline_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--query", default="all:retrieval augmented generation")
    parser.add_argument("--max-results", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PAPER_DIR)
    parser.add_argument("--download", dest="download", action="store_true", default=True)
    parser.add_argument("--no-download", dest="download", action="store_false")
    parser.add_argument("--parse", dest="parse", action="store_true", default=True)
    parser.add_argument("--no-parse", dest="parse", action="store_false")
    parser.add_argument("--retry-attempts", type=int, default=settings.pipeline_retry_attempts)
    parser.add_argument("--retry-backoff-seconds", type=float, default=settings.pipeline_retry_backoff_seconds)
    parser.add_argument("--parser", choices=["simple", "pymupdf", "docling"], default="docling")
    parser.add_argument("--chunk-strategy", choices=["fixed", "block", "agentic"], default="agentic")
    parser.add_argument("--planner-provider", choices=["ollama"])
    parser.add_argument("--planner-model")
    parser.add_argument("--debug-planner", action="store_true")
    parser.add_argument("--validate-chunks", action="store_true")


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        query=args.query,
        max_results=args.max_results,
        output_dir=args.output_dir,
        download=args.download,
        parse=args.parse,
        retry_attempts=args.retry_attempts,
        retry_backoff_seconds=args.retry_backoff_seconds,
        parser=args.parser,
        chunk_strategy=args.chunk_strategy,
        debug_planner=args.debug_planner,
        planner_provider=args.planner_provider,
        planner_model=args.planner_model,
        validate_chunks=args.validate_chunks,
    )


def pipeline_status() -> dict[str, object]:
    vector_store = get_elasticsearch_vector_store(
        settings.elasticsearch_url,
        settings.elasticsearch_index,
    )
    try:
        index_exists = vector_store.index_exists()
        indexed_chunks = vector_store.count(settings.embedding_model) if index_exists else 0
        elasticsearch_error = None
    except Exception as exc:
        index_exists = False
        indexed_chunks = 0
        elasticsearch_error = str(exc)
    return {
        "postgres_dsn_configured": bool(settings.postgres_dsn),
        "elasticsearch_url": settings.elasticsearch_url,
        "elasticsearch_index": settings.elasticsearch_index,
        "index_exists": index_exists,
        "indexed_chunks": indexed_chunks,
        "embedding_model": settings.embedding_model,
        "elasticsearch_error": elasticsearch_error,
    }


def schedule_pipeline(args: argparse.Namespace) -> None:
    config = config_from_args(args)
    scheduler = BlockingScheduler()
    if args.cron:
        scheduler.add_job(
            run_scheduled_once,
            "cron",
            args=[config],
            **parse_cron(args.cron),
        )
    else:
        scheduler.add_job(
            run_scheduled_once,
            "interval",
            minutes=args.interval_minutes,
            args=[config],
            next_run_time=None,
        )
    print("RAGArena pipeline scheduler started")
    scheduler.start()


def parse_cron(value: str) -> dict[str, str]:
    fields = value.split()
    if len(fields) != 5:
        raise ValueError("Cron expression must contain 5 fields: minute hour day month day_of_week")
    minute, hour, day, month, day_of_week = fields
    return {
        "minute": minute,
        "hour": hour,
        "day": day,
        "month": month,
        "day_of_week": day_of_week,
    }


def print_summary(payload: dict[str, object], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    for key, value in payload.items():
        print(f"{key}: {value}")
