import ragarena.agent.policies.grader as grader_policy
import ragarena.agent.policies.guardrail as guardrail_policy
import ragarena.agent.policies.rewriter as rewriter_policy
import ragarena.agent.policies.router as router_policy
import pytest
from ragarena.agent.policies.grader import GradeDecision, grade_documents
from ragarena.agent.policies.guardrail import evaluate_guardrail
from ragarena.agent.policies.rewriter import clean_rewritten_query, rewrite_query
from ragarena.agent.policies.router import route_query
from ragarena.cli.agent_test import main as agent_test_main


class FakeQwenDecisionModel:
    def __init__(self, answer: str) -> None:
        self.answer = answer

    def generate(self, prompt: str, system_prompt: str, *, json_mode: bool = True):
        return type("Result", (), {"answer": self.answer})()


@pytest.fixture(autouse=True)
def disable_live_qwen(monkeypatch) -> None:
    monkeypatch.setattr(guardrail_policy.settings, "agent_decision_enabled", False)


def test_rewrite_query_returns_non_empty_string() -> None:
    rewritten = rewrite_query("LangGraph")
    assert isinstance(rewritten, str)
    assert rewritten.strip()


def test_clean_rewritten_query_removes_prefixes() -> None:
    assert clean_rewritten_query("Rewritten query: LangGraph workflow") == "LangGraph workflow"


def test_agent_public_imports() -> None:
    assert callable(grade_documents)
    assert callable(evaluate_guardrail)
    assert callable(rewrite_query)
    assert callable(route_query)
    assert callable(agent_test_main)


def test_route_query_returns_legal_route() -> None:
    assert route_query("hello").route in {"direct_answer", "local_rag"}


def test_route_query_local_rag_for_langgraph() -> None:
    decision = route_query("LangGraph and LangChain differences")
    assert decision.route == "local_rag"
    assert decision.confidence > 0
    assert decision.reason


def test_current_events_stay_on_local_rag_without_web_path() -> None:
    assert route_query("latest AI news today").route == "local_rag"


def test_grade_documents_returns_decision() -> None:
    decision = grade_documents("LangGraph", ["LangGraph is a workflow framework."])
    assert isinstance(decision, GradeDecision)
    assert decision.sufficient is True


def test_grade_documents_is_section_aware_for_pe_ratio() -> None:
    decision = grade_documents(
        "What is discussed in Section 4.1 PE Ratio?",
        ["The section explains price to earnings ratios and how they are used in valuation."],
        retrieved_chunks=[
            {
                "chunk_id": 10,
                "document_id": 2,
                "section_name": "4.1 PE Ratio",
                "metadata": {},
            }
        ],
    )

    assert decision.relevant is True
    assert decision.sufficient is True
    assert decision.reason == "section_metadata_match"
    assert decision.useful_chunk_ids == [1]


def test_guardrail_rejects_empty_query() -> None:
    decision = evaluate_guardrail(" ")
    assert decision.passed is False
    assert decision.reason == "empty_query"


def test_guardrail_uses_qwen_for_semantic_decision(monkeypatch) -> None:
    monkeypatch.setattr(guardrail_policy.settings, "agent_decision_enabled", True)
    monkeypatch.setattr(
        guardrail_policy,
        "get_ollama_decision_generator",
        lambda *args: FakeQwenDecisionModel('{"passed":true,"reason":"research_question"}'),
    )

    decision = guardrail_policy.evaluate_guardrail("Explain retrieval augmented generation")

    assert decision.passed is True
    assert decision.reason == "qwen:research_question"


def test_router_uses_qwen_before_fallback(monkeypatch) -> None:
    monkeypatch.setattr(router_policy.settings, "agent_decision_enabled", True)
    monkeypatch.setattr(
        router_policy,
        "get_ollama_decision_generator",
        lambda *args: FakeQwenDecisionModel(
            '{"route":"direct_answer","confidence":0.82,"reason":"small_talk"}'
        ),
    )

    decision = router_policy.route_query("How are you?")

    assert decision.route == "direct_answer"
    assert decision.confidence == 0.82
    assert decision.reason == "qwen:small_talk"


def test_grader_uses_qwen_before_heuristics(monkeypatch) -> None:
    monkeypatch.setattr(grader_policy.settings, "agent_decision_enabled", True)
    monkeypatch.setattr(
        grader_policy,
        "get_ollama_decision_generator",
        lambda *args: FakeQwenDecisionModel(
            '{"relevant":true,"sufficient":false,"score":0.4,'
            '"reason":"needs_more_specific_context","useful_chunk_ids":[1],'
            '"suggested_rewrite":"LangGraph workflow orchestration"}'
        ),
    )

    decision = grader_policy.grade_documents("LangGraph", ["LangGraph is mentioned."])

    assert decision.relevant is True
    assert decision.sufficient is False
    assert decision.reason == "qwen:needs_more_specific_context"
    assert decision.suggested_rewrite == "LangGraph workflow orchestration"


def test_rewriter_uses_qwen(monkeypatch) -> None:
    monkeypatch.setattr(rewriter_policy.settings, "agent_decision_enabled", True)
    monkeypatch.setattr(
        rewriter_policy,
        "get_ollama_decision_generator",
        lambda *args: FakeQwenDecisionModel(
            '{"rewritten_query":"LangGraph workflow orchestration retrieval","reason":"expanded_terms"}'
        ),
    )

    rewritten = rewriter_policy.rewrite_query("LangGraph", current_query="LangGraph")

    assert rewritten == "LangGraph workflow orchestration retrieval"
