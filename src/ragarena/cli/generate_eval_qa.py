from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from dotenv import load_dotenv

from ragarena.evaluation.qa_generator import generate_eval_qa

__all__ = ["generate_eval_qa"]


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Generate deterministic retrieval QA from indexed chunks")
    parser.add_argument("--paper-id", type=int, required=True)
    parser.add_argument("--num-questions", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("data/eval/multi_paper_qa.json"))
    args = parser.parse_args()
    asyncio.run(generate_eval_qa(paper_id=args.paper_id, num_questions=args.num_questions, output=args.output))
