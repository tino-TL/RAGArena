from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from ragarena.agent.nodes import (
    direct_answer_node,
    generate_answer_node,
    give_up_node,
    guardrail_node,
    guardrail_reject_node,
    grade_documents_node,
    hybrid_retrieve_node,
    rerank_node,
    rewrite_query_node,
    router_node,
)
from ragarena.agent.state import AgentState, initial_agent_state
from ragarena.observability import get_langfuse_tracer
from ragarena.observability.trace_summary import TraceSummary

DEFAULT_MAX_REWRITE = 3


def build_agentic_rag_graph():
    graph_builder = StateGraph(AgentState)

    graph_builder.add_node("guardrail", guardrail_node)
    graph_builder.add_node("guardrail_reject", guardrail_reject_node)
    graph_builder.add_node("router", router_node)
    graph_builder.add_node("direct_answer", direct_answer_node)
    graph_builder.add_node("hybrid_retrieve", hybrid_retrieve_node)
    graph_builder.add_node("rerank", rerank_node)
    graph_builder.add_node("grade_documents", grade_documents_node)
    graph_builder.add_node("rewrite_query", rewrite_query_node)
    graph_builder.add_node("generate_answer", generate_answer_node)
    graph_builder.add_node("give_up", give_up_node)

    graph_builder.add_edge(START, "guardrail")
    graph_builder.add_conditional_edges(
        "guardrail",
        route_after_guardrail,
        {
            "continue": "router",
            "reject": "guardrail_reject",
        },
    )
    graph_builder.add_edge("guardrail_reject", END)
    graph_builder.add_conditional_edges(
        "router",
        route_after_router,
        {
            "direct_answer": "direct_answer",
            "local_rag": "hybrid_retrieve",
        },
    )
    graph_builder.add_edge("hybrid_retrieve", "rerank")
    graph_builder.add_edge("rerank", "grade_documents")
    graph_builder.add_conditional_edges(
        "grade_documents",
        route_after_grade,
        {
            "generate": "generate_answer",
            "rewrite": "rewrite_query",
            "give_up": "give_up",
        },
    )
    graph_builder.add_edge("rewrite_query", "hybrid_retrieve")
    graph_builder.add_edge("generate_answer", END)
    graph_builder.add_edge("give_up", END)
    graph_builder.add_edge("direct_answer", END)

    return graph_builder.compile()


def run_agentic_rag(query: str, max_rewrite: int = DEFAULT_MAX_REWRITE) -> AgentState:
    initial_state = initial_agent_state(query, max_rewrite=max_rewrite)
    graph = build_agentic_rag_graph()
    tracer = get_langfuse_tracer()
    with tracer.start_trace(
        "agentic_rag.workflow",
        {"query": query, "max_rewrite": max_rewrite},
        metadata={"workflow": "langgraph"},
    ) as span:
        state = graph.invoke(initial_state)
        state["trace_id"] = tracer.get_trace_id()
        state["trace_url"] = tracer.get_trace_url()
        state["trace_summary"] = TraceSummary.from_state(state).to_dict()
        span.update(
            output={
                "route": state["route"],
                "route_confidence": state["route_confidence"],
                "route_reason": state["route_reason"],
                "grade": state["grade"],
                "grade_reason": state["grade_reason"],
                "rewrite_count": state["rewrite_count"],
                "rewrite_reason": state["rewrite_reason"],
                "generation": state["generation"],
                "trace": state["trace"],
                "trace_summary": state["trace_summary"],
            }
        )
        return state


def route_after_guardrail(state: AgentState) -> str:
    return "continue" if state.get("guardrail_passed", True) else "reject"


def route_after_router(state: AgentState) -> str:
    return state["route"] if state["route"] in {"direct_answer", "local_rag"} else "local_rag"


def route_after_grade(state: AgentState) -> str:
    if state["grade"]:
        return "generate"

    if state["rewrite_count"] < state["max_rewrite"]:
        return "rewrite"

    return "give_up"
