from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import asyncpg
import requests

from ragarena.config import settings
from ragarena.runtime import get_redis_cache


EVAL_FILES: dict[str, str] = {
    "qa_gold.json": "[]\n",
    "retrieval.jsonl": "",
    "answer_judge.json": "[]\n",
}


POSTGRES_TABLES = [
    "chunk_embeddings",
    "document_chunks",
    "documents",
    "paper_blocks",
    "paper_files",
    "papers",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Reset RAGArena local data and evaluation state")
    parser.add_argument("--yes", action="store_true", help="Confirm destructive reset")
    parser.add_argument("--keep-eval-readme", action="store_true", default=True)
    args = parser.parse_args()
    if not args.yes:
        raise SystemExit("Refusing to reset data without --yes")

    reset_local_files()
    asyncio.run(reset_postgres())
    reset_elasticsearch()
    reset_redis()
    print("RAGArena data reset complete.")


def reset_local_files() -> None:
    root = Path.cwd().resolve()
    data_dir = (root / "data").resolve()
    papers_dir = (data_dir / "papers").resolve()
    eval_dir = (data_dir / "eval").resolve()
    for path in (papers_dir, eval_dir):
        ensure_within(path, data_dir)
        path.mkdir(parents=True, exist_ok=True)

    for file_path in papers_dir.glob("*"):
        if file_path.is_file():
            file_path.unlink()

    for file_path in eval_dir.glob("*"):
        if file_path.is_file() and file_path.name not in {"README.md", "ablation_plan.json"}:
            file_path.unlink()

    for name, content in EVAL_FILES.items():
        (eval_dir / name).write_text(content, encoding="utf-8")


async def reset_postgres() -> None:
    try:
        conn = await asyncpg.connect(settings.postgres_dsn)
    except Exception as exc:
        print(f"postgres: skipped ({exc})")
        return
    try:
        existing_tables = []
        for table_name in POSTGRES_TABLES:
            exists = await conn.fetchval("SELECT to_regclass($1)", table_name)
            if exists:
                existing_tables.append(table_name)
        if existing_tables:
            table_list = ", ".join(f'"{table_name}"' for table_name in existing_tables)
            await conn.execute(f"TRUNCATE TABLE {table_list} RESTART IDENTITY CASCADE")
        print("postgres: cleared")
    except Exception as exc:
        print(f"postgres: skipped ({exc})")
    finally:
        await conn.close()


def reset_elasticsearch() -> None:
    try:
        response = requests.delete(
            f"{settings.elasticsearch_url.rstrip('/')}/{settings.elasticsearch_index}",
            timeout=10,
        )
        if response.status_code in {200, 404}:
            print("elasticsearch: cleared")
            return
        print(f"elasticsearch: skipped ({response.status_code} {response.text[:200]})")
    except Exception as exc:
        print(f"elasticsearch: skipped ({exc})")


def reset_redis() -> None:
    try:
        get_redis_cache(settings.redis_url).client.flushdb()
        print("redis: cleared")
    except Exception as exc:
        print(f"redis: skipped ({exc})")


def ensure_within(path: Path, parent: Path) -> None:
    if path != parent and parent not in path.parents:
        raise RuntimeError(f"Refusing to reset path outside data directory: {path}")


if __name__ == "__main__":
    main()
