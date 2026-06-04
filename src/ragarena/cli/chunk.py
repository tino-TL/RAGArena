from __future__ import annotations

import argparse
import asyncio

from dotenv import load_dotenv

from ragarena.pipeline.steps import avg_chunk_tokens, chunk_documents

__all__ = ["avg_chunk_tokens", "chunk_documents"]


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Chunk documents for retrieval")
    parser.add_argument("--chunk-strategy", choices=["fixed", "block", "agentic"], default="agentic")
    parser.add_argument("--planner-provider", choices=["ollama"])
    parser.add_argument("--planner-model")
    parser.add_argument("--debug-planner", action="store_true")
    parser.add_argument("--validate-chunks", action="store_true")
    args = parser.parse_args()
    asyncio.run(
        chunk_documents(
            chunk_strategy=args.chunk_strategy,
            debug_planner=args.debug_planner,
            planner_provider=args.planner_provider,
            planner_model=args.planner_model,
            validate_chunks=args.validate_chunks,
        )
    )
