from __future__ import annotations

from ragarena.chunking.agentic_chunker import OllamaChunkPlanner
from ragarena.generation.generator import OllamaDecisionGenerator


class FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return self.payload


def test_decision_generator_disables_thinking(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["payload"] = json
        captured["timeout"] = timeout
        return FakeResponse({"model": "qwen3.5:4b", "message": {"content": '{"ok":true}'}})

    generator = OllamaDecisionGenerator("http://localhost:11434", "qwen3.5:4b")
    monkeypatch.setattr(generator.session, "post", fake_post)

    result = generator.generate("prompt", "system")

    assert result.answer == '{"ok":true}'
    assert captured["payload"]["think"] is False  # type: ignore[index]


def test_chunk_planner_disables_thinking(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["payload"] = json
        captured["timeout"] = timeout
        return FakeResponse({"message": {"content": '{"chunks":[]}'}})

    monkeypatch.setattr("ragarena.chunking.agentic_chunker.requests.post", fake_post)
    planner = OllamaChunkPlanner("http://localhost:11434", "qwen3.5:4b")

    result = planner.generate("prompt", "system")

    assert result.answer == '{"chunks":[]}'
    assert captured["payload"]["think"] is False  # type: ignore[index]
