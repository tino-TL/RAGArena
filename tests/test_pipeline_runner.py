from __future__ import annotations

import pytest

from ragarena.pipeline.runner import PipelineConfig, run_knowledge_pipeline, run_with_retry


@pytest.mark.anyio
async def test_pipeline_summary_tracks_steps_and_skips(monkeypatch) -> None:
    captured_chunk_kwargs: dict[str, object] = {}

    async def fake_fetch(**kwargs) -> int:
        return 2

    async def fake_chunk(**kwargs) -> int:
        captured_chunk_kwargs.update(kwargs)
        return 3

    async def fake_embed() -> int:
        return 3

    async def fake_index() -> int:
        return 3

    monkeypatch.setattr("ragarena.pipeline.runner.fetch_and_store_papers", fake_fetch)
    monkeypatch.setattr("ragarena.pipeline.runner.chunk_documents", fake_chunk)
    monkeypatch.setattr("ragarena.pipeline.runner.embed_chunks", fake_embed)
    monkeypatch.setattr("ragarena.pipeline.runner.index_embeddings", fake_index)

    summary = await run_knowledge_pipeline(
        PipelineConfig(download=False, parse=False, retry_attempts=1)
    )

    assert summary.status == "succeeded"
    assert [step.name for step in summary.steps] == [
        "arxiv_fetch",
        "download",
        "parse",
        "chunk",
        "embed",
        "index",
    ]
    assert summary.steps[1].status == "skipped"
    assert summary.steps[-1].count == 3
    assert captured_chunk_kwargs["chunk_strategy"] == "agentic"


def test_pipeline_defaults_use_docling_agentic_chunking() -> None:
    config = PipelineConfig()

    assert config.parser == "docling"
    assert config.chunk_strategy == "agentic"


@pytest.mark.anyio
async def test_pipeline_retry_succeeds_after_transient_error() -> None:
    attempts = {"count": 0}

    async def flaky_operation() -> int:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("temporary")
        return 7

    count = await run_with_retry(
        flaky_operation,
        attempts=2,
        backoff_seconds=0,
        step_name="test",
        run_id="run-1",
    )

    assert count == 7
    assert attempts["count"] == 2
