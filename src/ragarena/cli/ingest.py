from __future__ import annotations

import asyncio
import sys

from dotenv import load_dotenv

from ragarena.config import settings
from ragarena.ingestion.loaders import load_documents
from ragarena.ingestion.repository import ensure_documents_table, insert_documents


async def ingest(path: str) -> int:
    documents = load_documents(path)
    await ensure_documents_table(settings.postgres_dsn)
    inserted_count = await insert_documents(settings.postgres_dsn, documents)

    print(f"Loaded documents: {len(documents)}")
    print(f"Inserted documents: {inserted_count}")
    print(f"Skipped duplicates: {len(documents) - inserted_count}")

    return inserted_count


def main() -> None:
    load_dotenv()

    if len(sys.argv) != 2:
        print("Usage: uv run ragarena-ingest data/sample_docs")
        raise SystemExit(2)

    asyncio.run(ingest(sys.argv[1]))
