from __future__ import annotations

import argparse

from dotenv import load_dotenv

from ragarena.cli.formatters import print_trace_summary
from ragarena.agent.workflow import run_agentic_rag
from ragarena.observability.trace_summary import TraceSummary


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run the LangGraph agentic RAG workflow")
    parser.add_argument("query")
    parser.add_argument("--max-rewrite", type=int, default=3)
    args = parser.parse_args()

    state = run_agentic_rag(args.query, max_rewrite=args.max_rewrite)

    print(f"Original Query: {state['original_query']}")
    print(f"Route: {state['route']}")
    print("Trace")
    print("=====")
    for step in state["trace"]:
        print(f"- {step}")

    print("Final Answer")
    print("============")
    print(state["generation"])
    print()
    print_trace_summary(TraceSummary.from_state(state).to_dict())
