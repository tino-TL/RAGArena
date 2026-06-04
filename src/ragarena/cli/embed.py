from __future__ import annotations

import asyncio

from dotenv import load_dotenv

from ragarena.pipeline.steps import embed_chunks

__all__ = ["embed_chunks"]


def main() -> None:
    load_dotenv()
    asyncio.run(embed_chunks())
