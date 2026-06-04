from __future__ import annotations

from dataclasses import dataclass

import pytest
import requests

from ragarena.chunking.agentic_chunker import (
    OllamaChunkPlanner,
    build_chunk_quality_report,
    build_planner,
    chunk_agentic_documents,
    cleanup_retrieval_chunks,
    extract_json_object,
    merge_small_agentic_chunks,
    optimize_retrieval_units,
    overlap_ratio,
    print_chunk_quality_report,
)
from ragarena.chunking.fixed_chunker import Chunk
from ragarena.chunking.repository import DocumentRecord
from ragarena.papers.models import PaperBlock


@dataclass(frozen=True)
class FakeGenerationResult:
    answer: str
    total_duration: int | None = None
    load_duration: int | None = None
    eval_duration: int | None = None


class FakePlanner:
    def __init__(self, answer: str | list[str], configured: bool = True) -> None:
        self.answers = answer if isinstance(answer, list) else [answer]
        self.configured = configured
        self.calls = 0

    def is_configured(self) -> bool:
        return self.configured

    def generate(self, prompt: str, system_prompt: str):
        answer = self.answers[min(self.calls, len(self.answers) - 1)]
        self.calls += 1
        return FakeGenerationResult(answer=answer)


class FakePlannerWithDurations(FakePlanner):
    def generate(self, prompt: str, system_prompt: str):
        answer = self.answers[min(self.calls, len(self.answers) - 1)]
        self.calls += 1
        return FakeGenerationResult(
            answer=answer,
            total_duration=4_000_000_000,
            load_duration=2_000_000_000,
            eval_duration=500_000_000,
        )


def document(
    *,
    document_id: int = 10,
    arxiv_id: str = "2401.00001v1",
    title: str = "Paper",
) -> DocumentRecord:
    return DocumentRecord(
        id=document_id,
        title=title,
        source=f"https://arxiv.org/abs/{arxiv_id}",
        content=f"arXiv ID: {arxiv_id}\nPaper content",
        content_hash=f"hash-{arxiv_id}",
    )


def block(
    block_id: int,
    section: str,
    content: str,
    *,
    paper_id: int = 1,
    arxiv_id: str = "2401.00001v1",
    block_type: str = "paragraph",
    should_embed: bool = True,
    markdown_content: str | None = None,
    page_number: int = 1,
) -> PaperBlock:
    return PaperBlock(
        id=block_id,
        paper_id=paper_id,
        arxiv_id=arxiv_id,
        block_type=block_type,
        section_name=section,
        page_number=page_number,
        content=content,
        markdown_content=markdown_content,
        image_path=None,
        order_index=block_id,
        should_embed=should_embed,
        metadata={},
        content_hash=f"b{block_id}",
    )


def blocks() -> list[PaperBlock]:
    return [
        block(1, "Method", "Original method text.", block_type="section_title"),
        block(2, "Method", long_text("Original second text.")),
    ]


def two_section_blocks() -> list[PaperBlock]:
    return blocks() + [block(3, "Results", long_text("Original result text."))]


def long_text(prefix: str, words: int = 90) -> str:
    return f"{prefix} " + " ".join(["retrieval"] * words)


def simple_plan(ids: list[int], chunk_type: str = "method", section: str = "Method") -> str:
    return (
        '{"chunks":[{"chunk_type":"'
        + chunk_type
        + '","section_name":"'
        + section
        + '","source_block_ids":'
        + str(ids)
        + ',"should_embed":true}]}'
    )


def test_ollama_planner_mock_returns_valid_json(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeResponse:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "message": {"content": simple_plan([1, 2])},
                "total_duration": 1_500_000_000,
                "load_duration": 700_000_000,
                "eval_duration": 300_000_000,
            }

    def fake_post(*args, **kwargs):
        captured["payload"] = kwargs["json"]
        return FakeResponse()

    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: FakeResponse())
    monkeypatch.setattr(requests, "post", fake_post)

    planner = OllamaChunkPlanner("http://localhost:11434", "qwen2.5:3b", keep_alive="30m")
    result = planner.generate("prompt")

    assert planner.is_configured() is True
    assert '"chunks"' in result.answer
    assert result.total_duration == 1_500_000_000
    assert result.load_duration == 700_000_000
    assert result.eval_duration == 300_000_000
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["keep_alive"] == "30m"


def test_planner_trace_records_ollama_durations() -> None:
    result = chunk_agentic_documents(
        [document()],
        blocks(),
        model=FakePlannerWithDurations(simple_plan([1, 2])),
        enabled=True,
    )

    assert result.traces[0].total_duration == 4_000_000_000
    assert result.traces[0].load_duration == 2_000_000_000
    assert result.traces[0].eval_duration == 500_000_000
    assert result.traces[0].load_warning is not None


def test_agentic_chunker_uses_source_ids_and_original_content() -> None:
    result = chunk_agentic_documents(
        [document()],
        blocks(),
        model=FakePlanner(simple_plan([1, 2])),
        enabled=True,
    )

    assert len(result.chunks) == 1
    assert result.chunks[0].source_block_ids == [1, 2]
    assert "Original method text." in result.chunks[0].content
    assert "Original second text." in result.chunks[0].content
    assert result.chunks[0].chunk_type == "retrieval_unit"
    assert result.chunks[0].chunking_strategy == "agentic"


def test_planner_output_does_not_fill_removed_metadata_fields() -> None:
    result = chunk_agentic_documents(
        [document()],
        blocks(),
        model=FakePlanner(
            '{"chunks":[{"chunk_type":"method","section_name":"Method","source_block_ids":[1],"should_embed":true,'
            '"query_intents":["bad"],"keywords":["bad"],"reason":"bad"}]}'
        ),
        enabled=True,
    )

    assert result.chunks[0].retrieval_value is None
    assert result.chunks[0].query_intents == []
    assert result.chunks[0].keywords == []
    assert result.chunks[0].planner_reason is None


def test_section_title_merges_to_following_paragraph() -> None:
    result = chunk_agentic_documents(
        [document()],
        blocks(),
        model=FakePlanner(simple_plan([1, 2])),
        enabled=True,
    )

    assert result.chunks[0].content.startswith("## Original method text.")
    assert "Original second text." in result.chunks[0].content


def test_non_adjacent_source_block_ids_are_rejected() -> None:
    source_blocks = [
        block(1, "Method", long_text("one")),
        block(2, "Method", long_text("two")),
        block(3, "Method", long_text("three")),
    ]
    result = chunk_agentic_documents(
        [document()],
        source_blocks,
        model=FakePlanner(["bad-json", "still-bad"]),
        enabled=True,
    )

    assert result.chunks
    assert all(chunk.chunking_strategy == "agentic" for chunk in result.chunks)
    assert all(chunk.chunk_type == "retrieval_unit" for chunk in result.chunks)
    assert result.traces[0].fallback_reason


def test_reference_authors_affiliation_do_not_enter_agentic_chunk() -> None:
    source_blocks = [
        block(1, "unknown", "Author Name", block_type="authors", should_embed=False),
        block(2, "unknown", "Example University", block_type="affiliation", should_embed=False),
        block(3, "References", "[1] Smith", block_type="reference", should_embed=False),
        block(4, "Method", long_text("Real content")),
    ]
    result = chunk_agentic_documents(
        [document()],
        source_blocks,
        model=FakePlanner(simple_plan([4])),
        enabled=True,
    )

    assert "Author Name" not in result.chunks[0].content
    assert "Example University" not in result.chunks[0].content
    assert "[1] Smith" not in result.chunks[0].content
    assert "Real content" in result.chunks[0].content


def test_ollama_unavailable_falls_back_to_block() -> None:
    result = chunk_agentic_documents(
        [document()],
        blocks(),
        model=FakePlanner("{}", configured=False),
        enabled=True,
    )

    assert result.chunks
    assert result.chunks[0].chunking_strategy == "block"
    assert result.traces[-1].fallback_reason == "planning model unavailable"


def test_deepseek_is_not_supported_as_chunk_planner() -> None:
    with pytest.raises(ValueError, match="Unsupported chunk planner provider"):
        build_planner("deepseek", "deepseek-chat")


def test_invalid_json_repair_succeeds_with_mock() -> None:
    planner = FakePlanner(["not-json", simple_plan([1])])

    result = chunk_agentic_documents([document()], blocks(), model=planner, enabled=True)

    assert result.chunks[0].chunking_strategy == "agentic"
    assert planner.calls == 2


def test_one_bad_window_does_not_fallback_entire_run() -> None:
    planner = FakePlanner(["bad-json", "still-bad", simple_plan([3], chunk_type="experiment", section="Results")])

    result = chunk_agentic_documents(
        [document()],
        two_section_blocks(),
        model=planner,
        enabled=True,
        max_tokens=1000,
    )

    assert all(chunk.chunking_strategy == "agentic" for chunk in result.chunks)
    assert all(chunk.chunk_type == "retrieval_unit" for chunk in result.chunks)
    assert any(trace.fallback_reason for trace in result.traces)


def test_fenced_json_extracts() -> None:
    assert extract_json_object('```json\n{"chunks":[]}\n```') == '{"chunks":[]}'


def test_extra_prose_around_json_extracts() -> None:
    assert extract_json_object('Here is the plan:\n{"chunks":[]}\nDone.') == '{"chunks":[]}'


def test_small_adjacent_agentic_chunks_are_merged() -> None:
    chunks = [
        make_chunk(1, "Method", "background", "first short text"),
        make_chunk(2, "Method", "other", "second short text"),
    ]

    merged = merge_small_agentic_chunks(chunks, min_tokens=200, target_tokens=500)

    assert len(merged) == 1
    assert merged[0].source_block_ids == [1, 2]
    assert merged[0].chunk_type == "retrieval_unit"


def test_small_agentic_chunk_merge_preserves_document_order_across_sections() -> None:
    chunks = [
        make_chunk(1, "Abstract", "abstract", long_text("Abstract body", 30)),
        make_chunk(2, "3.2 Short Term Reversion Factor Exploration", "method", long_text("Reversion body", 30)),
    ]

    merged = merge_small_agentic_chunks(chunks, min_tokens=200, target_tokens=500)

    assert [chunk.section_name for chunk in merged] == [
        "Abstract",
        "3.2 Short Term Reversion Factor Exploration",
    ]


def test_micro_merge_merges_small_chunk_with_same_major_neighbor_under_max() -> None:
    chunks = [
        make_chunk(1, "3.1 Momentum Factor Exploration", "retrieval_unit", long_text("Momentum overview", 150)),
        make_chunk(2, "3.1.1 Factor Construction", "retrieval_unit", long_text("Small factor note", 40)),
        make_chunk(3, "3.1.2 Calculate Factor Returns", "retrieval_unit", long_text("Factor returns", 200)),
    ]

    merged = merge_small_agentic_chunks(chunks, min_tokens=180, target_tokens=400, max_tokens=800)

    assert len(merged) == 2
    assert merged[0].source_block_ids == [1, 2]
    assert merged[0].chunk_index == 0
    assert merged[1].chunk_index == 1


def test_micro_merge_does_not_cross_major_section() -> None:
    chunks = [
        make_chunk(1, "3.2.2 Calculate Factor Returns", "retrieval_unit", long_text("Small factor returns", 40)),
        make_chunk(2, "4 Fundamental Analysis", "retrieval_unit", long_text("Fundamental analysis", 120)),
    ]

    merged = merge_small_agentic_chunks(chunks, min_tokens=180, target_tokens=400, max_tokens=800)

    assert len(merged) == 2
    assert [chunk.section_name for chunk in merged] == ["3.2.2 Calculate Factor Returns", "4 Fundamental Analysis"]


def test_micro_merge_does_not_merge_abstract_or_introduction() -> None:
    chunks = [
        make_chunk(1, "Abstract", "retrieval_unit", long_text("Abstract body", 40)),
        make_chunk(2, "1 Introduction", "retrieval_unit", long_text("Introduction body", 40)),
        make_chunk(3, "2 Method", "retrieval_unit", long_text("Method body", 120)),
    ]

    merged = merge_small_agentic_chunks(chunks, min_tokens=180, target_tokens=400, max_tokens=800)

    assert len(merged) == 3


def test_micro_merge_does_not_merge_large_table_result_chunk() -> None:
    table_content = "| A | B |\n|---|---|\n" + "\n".join(f"| row{i} | {i} |" for i in range(310))
    chunks = [
        make_chunk(1, "4.1 Results", "table", table_content),
        make_chunk(2, "4.1 Results", "retrieval_unit", long_text("Short result note", 40)),
    ]

    merged = merge_small_agentic_chunks(chunks, min_tokens=400, target_tokens=500, max_tokens=800)

    assert len(merged) == 2
    assert merged[0].chunk_type == "table"


def test_agentic_stats_reflect_post_merge_average_tokens() -> None:
    source_blocks = [
        block(1, "Method", " ".join(["first"] * 30)),
        block(2, "Method", " ".join(["second"] * 30)),
    ]
    result = chunk_agentic_documents(
        [document()],
        source_blocks,
        model=FakePlanner(
            '{"chunks":['
            '{"chunk_type":"background","section_name":"Method","source_block_ids":[1],"should_embed":true},'
            '{"chunk_type":"background","section_name":"Method","source_block_ids":[2],"should_embed":true}'
            ']}'
        ),
        enabled=True,
    )

    assert result.stats["before_merge_chunks"] == 2
    assert result.stats["after_merge_chunks"] == 1
    assert result.stats["avg_chunk_tokens"] == result.chunks[0].token_count


def test_chunk_strategy_remains_agentic_after_merge() -> None:
    result = chunk_agentic_documents(
        [document()],
        blocks(),
        model=FakePlanner(simple_plan([1])),
        enabled=True,
    )

    assert result.chunks[0].chunking_strategy == "agentic"


def test_agentic_chunk_section_name_prefers_content_heading_over_planner_metadata() -> None:
    source_blocks = [
        block(1, "4.1 PE Ratio", long_text("## 4.2 PB Ratio\n\nPB body")),
    ]

    result = chunk_agentic_documents(
        [document()],
        source_blocks,
        model=FakePlanner(simple_plan([1], chunk_type="paragraph", section="4.1 PE Ratio")),
        enabled=True,
    )

    assert result.chunks[0].section_name == "4.2 PB Ratio"
    assert result.chunks[0].chunk_type == "retrieval_unit"


def test_agentic_chunker_isolates_same_section_name_by_paper() -> None:
    docs = [
        document(document_id=10, arxiv_id="2401.00001v1", title="First Paper"),
        document(document_id=20, arxiv_id="2401.00002v1", title="Second Paper"),
    ]
    source_blocks = [
        block(1, "1 Introduction", long_text("First paper introduction."), paper_id=1, arxiv_id="2401.00001v1"),
        block(2, "1 Introduction", long_text("First paper method context."), paper_id=1, arxiv_id="2401.00001v1"),
        block(101, "1 Introduction", long_text("Second paper introduction."), paper_id=2, arxiv_id="2401.00002v1"),
        block(102, "1 Introduction", long_text("Second paper method context."), paper_id=2, arxiv_id="2401.00002v1"),
    ]
    planner = FakePlanner(
        [
            simple_plan([1, 2], section="1 Introduction"),
            simple_plan([101, 102], section="1 Introduction"),
        ]
    )

    result = chunk_agentic_documents(docs, source_blocks, model=planner, enabled=True)

    assert planner.calls == 2
    assert len(result.chunks) == 2
    assert {chunk.document_id for chunk in result.chunks} == {10, 20}
    assert {tuple(chunk.source_block_ids or []) for chunk in result.chunks} == {(1, 2), (101, 102)}


def test_figure_caption_binds_after_body_as_visual_context() -> None:
    source_blocks = [
        block(1, "Results", "Figure 1: short caption.", block_type="figure_caption"),
        block(2, "Results", long_text("This paragraph explains the figure and reports the main observed trend.")),
    ]

    result = chunk_agentic_documents(
        [document()],
        source_blocks,
        model=FakePlanner(
            '{"chunks":['
            '{"chunk_type":"figure_caption","section_name":"Results","source_block_ids":[1],"should_embed":true},'
            '{"chunk_type":"experiment","section_name":"Results","source_block_ids":[2],"should_embed":true}'
            ']}'
        ),
        enabled=True,
    )

    assert {tuple(chunk.source_block_ids or []) for chunk in result.chunks} == {(1,), (2,), (1, 2)}
    assert any(chunk.chunk_type == "figure_caption" and chunk.source_block_ids == [1] for chunk in result.chunks)
    assert any(chunk.chunk_type == "retrieval_unit" and chunk.source_block_ids == [2] for chunk in result.chunks)
    fused = next(chunk for chunk in result.chunks if chunk.chunk_type == "fused")
    assert "explains the figure" in fused.content
    assert "Related visual/table evidence:" in fused.content
    assert "Figure 1: short caption." in fused.content
    assert fused.metadata["semantic_chunk_type"] == "fused"
    assert fused.metadata["visual_refs"] == ["Fig. 1"]


def test_body_blocks_can_merge_across_separate_figure_caption_lane() -> None:
    source_blocks = [
        block(1, "Method", long_text("Geometry stage supports semantic edits.")),
        block(2, "Method", "Figure 2: Pipeline overview.", block_type="figure_caption"),
        block(3, "Method", long_text("structural edits add missing parts and adjust object placement.")),
    ]
    planner = FakePlanner(
        [
            simple_plan([1, 3], chunk_type="method", section="Method"),
            simple_plan([2], chunk_type="figure_caption", section="Method"),
        ]
    )

    result = chunk_agentic_documents([document()], source_blocks, model=planner, enabled=True)

    assert planner.calls == 2
    assert {tuple(chunk.source_block_ids or []) for chunk in result.chunks} == {(1, 3), (2,), (1, 2, 3)}
    body_chunk = next(chunk for chunk in result.chunks if chunk.source_block_ids == [1, 3])
    assert "Geometry stage" in body_chunk.content
    assert "Related visual/table evidence:" not in body_chunk.content
    assert "Figure 2" not in body_chunk.content
    assert any(chunk.chunk_type == "fused" and chunk.source_block_ids == [1, 2, 3] for chunk in result.chunks)


def test_caption_not_inside_unfinished_sentence() -> None:
    source_blocks = [
        block(1, "3.1 Staged Scene Construction", "Geometry stage performs three edits: (1) local shape edits, (2) geometric transforms, and (3)"),
        block(2, "3.1 Staged Scene Construction", "Fig. 2: Overview of the staged construction pipeline.", block_type="figure_caption"),
        block(3, "3.1 Staged Scene Construction", "structural edits, such as adding missing parts and revising object relations."),
    ]
    planner = FakePlanner(
        [
            simple_plan([1, 3], chunk_type="method", section="3.1 Staged Scene Construction"),
            simple_plan([2], chunk_type="figure_caption", section="3.1 Staged Scene Construction"),
        ]
    )

    result = chunk_agentic_documents([document()], source_blocks, model=planner, enabled=True)

    body = next(chunk for chunk in result.chunks if chunk.chunk_type == "retrieval_unit")
    assert "structural edits" in body.content
    assert "Fig. 2" not in body.content
    assert "and (3)\n\nstructural edits" in body.content
    fused = next(chunk for chunk in result.chunks if chunk.chunk_type == "fused")
    assert fused.content.index("structural edits") < fused.content.index("Related visual/table evidence:")
    assert "and (3)\n\nRelated visual/table evidence" not in fused.content


def test_table_caption_table_and_paragraph_stay_separate() -> None:
    markdown_table = "\n".join(
        [
            "| Metric | Score |",
            "| --- | --- |",
            "| A | 1 |",
            "| B | 2 |",
            "| C | 3 |",
            "| D | 4 |",
            "| E | 5 |",
            "| F | 6 |",
        ]
    )
    source_blocks = [
        block(1, "Experiments", "Table 1: Main benchmark results.", block_type="table_caption"),
        block(2, "Experiments", "full table", block_type="table", markdown_content=markdown_table),
        block(3, "Experiments", long_text("The table shows that the proposed retriever improves recall.")),
    ]

    result = chunk_agentic_documents(
        [document()],
        source_blocks,
        model=FakePlanner(
            '{"chunks":['
            '{"chunk_type":"table","section_name":"Experiments","source_block_ids":[1],"should_embed":true},'
            '{"chunk_type":"table","section_name":"Experiments","source_block_ids":[2],"should_embed":true},'
            '{"chunk_type":"experiment","section_name":"Experiments","source_block_ids":[3],"should_embed":true}'
            ']}'
        ),
        enabled=True,
    )

    assert {tuple(chunk.source_block_ids or []) for chunk in result.chunks} == {(1,), (2,), (3,), (2, 3)}
    assert any("Table 1: Main benchmark results." in chunk.content for chunk in result.chunks)
    table_chunk = next(chunk for chunk in result.chunks if chunk.source_block_ids == [2])
    assert "| E | 5 |" in table_chunk.content
    assert "| F | 6 |" not in table_chunk.content
    assert table_chunk.metadata["semantic_chunk_type"] == "table"
    assert table_chunk.metadata["table_confidence"] == "high"
    assert any("improves recall" in chunk.content for chunk in result.chunks)
    assert any(chunk.chunk_type == "fused" and chunk.source_block_ids == [2, 3] for chunk in result.chunks)


def test_missing_formula_marker_is_preserved_and_flagged() -> None:
    source_blocks = [
        block(1, "Method", "The loss is defined as:\n\n<!-- formula-not-decoded -->"),
    ]

    result = chunk_agentic_documents(
        [document()],
        source_blocks,
        model=FakePlanner(simple_plan([1], chunk_type="method", section="Method")),
        enabled=True,
    )

    assert "<!-- formula-not-decoded -->" in result.chunks[0].content
    assert result.chunks[0].metadata["has_missing_formula"] is True
    assert "missing_formula" in result.chunks[0].metadata["quality_flags"]


def test_quality_report_counts_duplicate_adjacent_sections() -> None:
    text = long_text("Direct Mode and Interactive Mode expose a structured submission interface")
    chunks = [
        Chunk(
            document_id=10,
            chunk_index=1,
            content=text,
            token_count=len(text.split()),
            content_hash="left",
            section_name="3.3 Multi-Source Clinical Environment",
        ),
        Chunk(
            document_id=10,
            chunk_index=2,
            content=f"Interaction Modes\n\n{text}",
            token_count=len(text.split()) + 2,
            content_hash="right",
            section_name="3.4 Interaction Modes",
        ),
    ]

    report = build_chunk_quality_report(chunks)

    assert report["duplicate_neighbor_chunks"] == 1


def test_formula_and_method_paragraph_merge() -> None:
    source_blocks = [
        block(1, "Method", long_text("We optimize the retriever using the following objective.")),
        block(2, "Method", "L = sum_i log p(y_i | x_i)", block_type="formula", should_embed=True),
    ]

    result = chunk_agentic_documents(
        [document()],
        source_blocks,
        model=FakePlanner(
            '{"chunks":['
            '{"chunk_type":"method","section_name":"Method","source_block_ids":[1],"should_embed":true},'
            '{"chunk_type":"other","section_name":"Method","source_block_ids":[2],"should_embed":true}'
            ']}'
        ),
        enabled=True,
    )

    assert len(result.chunks) == 1
    assert result.chunks[0].source_block_ids == [1, 2]
    assert "following objective" in result.chunks[0].content
    assert "L = sum_i" in result.chunks[0].content


def test_isolated_short_caption_generates_standalone_chunk() -> None:
    source_blocks = [
        block(1, "Results", "Figure 2: short caption.", block_type="figure_caption"),
    ]

    result = chunk_agentic_documents(
        [document()],
        source_blocks,
        model=FakePlanner(simple_plan([1], chunk_type="figure_caption", section="Results")),
        enabled=True,
    )

    assert len(result.chunks) == 1
    assert result.chunks[0].source_block_ids == [1]
    assert "Figure 2: short caption." in result.chunks[0].content


def test_section_title_does_not_generate_standalone_chunk() -> None:
    result = chunk_agentic_documents(
        [document()],
        blocks(),
        model=FakePlanner(
            '{"chunks":['
            '{"chunk_type":"other","section_name":"Method","source_block_ids":[1],"should_embed":true},'
            '{"chunk_type":"method","section_name":"Method","source_block_ids":[2],"should_embed":true}'
            ']}'
        ),
        enabled=True,
    )

    assert len(result.chunks) == 1
    assert result.chunks[0].source_block_ids == [1, 2]


def test_abstract_heading_does_not_become_standalone_chunk() -> None:
    source_blocks = [
        block(1, "Abstract", "Abstract", block_type="section_title"),
    ]

    result = chunk_agentic_documents(
        [document()],
        source_blocks,
        model=FakePlanner(simple_plan([1], chunk_type="other", section="Abstract")),
        enabled=True,
    )

    assert result.chunks == []


def test_title_authors_affiliation_do_not_enter_document_chunks() -> None:
    source_blocks = [
        block(1, "unknown", "Paper Title", block_type="title"),
        block(2, "unknown", "Jane Doe and John Smith", block_type="authors", should_embed=False),
        block(3, "unknown", "Example University", block_type="affiliation", should_embed=False),
        block(4, "Introduction", " ".join(["retrieval"] * 60)),
    ]

    result = chunk_agentic_documents(
        [document()],
        source_blocks,
        model=FakePlanner(simple_plan([4], chunk_type="background", section="Introduction")),
        enabled=True,
    )

    assert len(result.chunks) == 1
    assert "Paper Title" not in result.chunks[0].content
    assert "Jane Doe" not in result.chunks[0].content
    assert "University" not in result.chunks[0].content


def test_standalone_page_number_block_does_not_enter_retrieval_unit() -> None:
    source_blocks = [
        block(1, "Abstract", "1"),
        block(2, "Abstract", long_text("Abstract body reports the contribution.")),
    ]

    result = chunk_agentic_documents(
        [document()],
        source_blocks,
        model=FakePlanner(simple_plan([1, 2], chunk_type="abstract", section="Abstract")),
        enabled=True,
    )

    assert len(result.chunks) == 1
    assert result.chunks[0].source_block_ids == [2]
    assert not result.chunks[0].content.startswith("1\n\n")
    assert "Abstract body" in result.chunks[0].content


def test_standalone_date_block_does_not_enter_retrieval_unit() -> None:
    source_blocks = [
        block(1, "Abstract", "Sept 2023"),
        block(2, "Abstract", long_text("Abstract body reports the contribution.")),
    ]

    result = chunk_agentic_documents(
        [document()],
        source_blocks,
        model=FakePlanner(simple_plan([1, 2], chunk_type="abstract", section="Abstract")),
        enabled=True,
    )

    assert len(result.chunks) == 1
    assert "Sept 2023" not in result.chunks[0].content
    assert "Abstract body" in result.chunks[0].content


def test_percent_in_body_is_not_removed_as_metadata_noise() -> None:
    source_blocks = [
        block(1, "Results", long_text("The model improves by 21.19% on the benchmark.")),
    ]

    result = chunk_agentic_documents(
        [document()],
        source_blocks,
        model=FakePlanner(simple_plan([1], chunk_type="experiment", section="Results")),
        enabled=True,
    )

    assert len(result.chunks) == 1
    assert "21.19%" in result.chunks[0].content


def test_table_numbers_are_not_removed_as_metadata_noise() -> None:
    markdown_table = "| Metric | Score |\n| --- | ---: |\n| Recall | 21.19 |\n| NDCG | 15 |"
    source_blocks = [
        block(1, "Results", markdown_table, block_type="table", markdown_content=markdown_table),
    ]

    result = chunk_agentic_documents(
        [document()],
        source_blocks,
        model=FakePlanner(simple_plan([1], chunk_type="table", section="Results")),
        enabled=True,
    )

    assert len(result.chunks) == 1
    assert "| Recall | 21.19 |" in result.chunks[0].content
    assert "| NDCG | 15 |" in result.chunks[0].content


def test_tiny_regular_chunk_merges_to_same_section_neighbor() -> None:
    chunks = [
        make_chunk(1, "Method", "other", "tiny note"),
        make_chunk(2, "Method", "method", " ".join(["method"] * 70)),
    ]

    cleanup = cleanup_retrieval_chunks(chunks)

    assert len(cleanup.chunks) == 1
    assert cleanup.chunks[0].source_block_ids == [1, 2]
    assert cleanup.merged_tiny_chunks == 1


def test_tiny_garbage_not_preserved_by_code_or_table_chunk_type() -> None:
    chunks = [
        make_chunk(1, "Method", "code", "x"),
        make_chunk(2, "Results", "table", "n/a"),
    ]

    cleanup = cleanup_retrieval_chunks(chunks)

    assert cleanup.chunks == []
    assert cleanup.dropped_tiny_chunks == 2


def test_normal_content_with_code_chunk_type_survives_cleanup() -> None:
    chunks = [
        make_chunk(1, "Method", "code", long_text("From this table, we can tell that performance improves")),
    ]

    cleanup = cleanup_retrieval_chunks(chunks)

    assert len(cleanup.chunks) == 1
    assert cleanup.chunks[0].content.startswith("From this table")


def test_docling_code_block_type_normal_prose_enters_retrieval_unit() -> None:
    source_blocks = [
        block(
            1,
            "Analysis",
            long_text("From this table, we can tell that the financial ratio improves"),
            block_type="code",
        ),
    ]

    result = chunk_agentic_documents(
        [document()],
        source_blocks,
        model=FakePlanner(simple_plan([1], chunk_type="code", section="Analysis")),
        enabled=True,
    )

    assert len(result.chunks) == 1
    assert result.chunks[0].chunk_type == "retrieval_unit"
    assert result.chunks[0].source_block_ids == [1]
    assert result.chunks[0].chunking_strategy == "agentic"


def test_docling_table_like_block_type_normal_prose_enters_retrieval_unit() -> None:
    source_blocks = [
        block(
            1,
            "Analysis",
            long_text("The P/B ratio provides insight into relative valuation across sectors"),
            block_type="table_like",
        ),
    ]

    result = chunk_agentic_documents(
        [document()],
        source_blocks,
        model=FakePlanner(simple_plan([1], chunk_type="table", section="Analysis")),
        enabled=True,
    )

    assert len(result.chunks) == 1
    assert result.chunks[0].chunk_type == "retrieval_unit"
    assert result.chunks[0].source_block_ids == [1]


def test_fallback_block_chunks_use_tiny_cleanup() -> None:
    source_blocks = [
        block(1, "Abstract", "Abstract", block_type="section_title"),
        block(2, "Abstract", " ".join(["retrieval"] * 90)),
    ]

    result = chunk_agentic_documents(
        [document()],
        source_blocks,
        model=FakePlanner(["bad-json", "still-bad"]),
        enabled=True,
    )

    assert len(result.chunks) == 1
    assert result.chunks[0].chunking_strategy == "agentic"
    assert result.chunks[0].chunk_type == "retrieval_unit"
    assert result.chunks[0].source_block_ids == [1, 2]
    assert result.stats["min_chunk_tokens"] >= 80


def make_chunk(left_id: int, section: str, chunk_type: str, content: str) -> Chunk:
    return Chunk(
        document_id=10,
        chunk_index=left_id,
        content=content,
        token_count=len(content.split()),
        content_hash=f"hash-{left_id}",
        chunk_type=chunk_type,
        section_name=section,
        source_block_ids=[left_id],
        chunking_strategy="agentic",
    )


def test_overlap_ratio_uses_smaller_chunk_denominator() -> None:
    assert overlap_ratio([196, 197, 198, 199, 200], [200, 201]) == 0.5


def test_optimize_retrieval_units_merges_overlapping_chunks() -> None:
    source_blocks = [
        block(196, "3.1.2 Calculate Factor Returns", "Momentum intro " + " ".join(["a"] * 40)),
        block(197, "3.1.2 Calculate Factor Returns", "Momentum table one " + " ".join(["b"] * 40)),
        block(198, "3.1.2 Calculate Factor Returns", "Momentum table two " + " ".join(["c"] * 40)),
        block(199, "3.1.2 Calculate Factor Returns", "Momentum analysis " + " ".join(["d"] * 40)),
        block(200, "3.1.2 Calculate Factor Returns", "Momentum conclusion return sharpe 1.2"),
        block(201, "3.1.2 Calculate Factor Returns", "Additional conclusion return sharpe 1.3"),
    ]
    chunks = [
        Chunk(
            document_id=10,
            chunk_index=0,
            content="chunk A",
            token_count=100,
            content_hash="a",
            chunk_type="retrieval_unit",
            section_name="3.1.2 Calculate Factor Returns",
            source_block_ids=[196, 197, 198, 199, 200],
            chunking_strategy="agentic",
        ),
        Chunk(
            document_id=10,
            chunk_index=1,
            content="chunk B",
            token_count=40,
            content_hash="b",
            chunk_type="retrieval_unit",
            section_name="3.1.2 Calculate Factor Returns",
            source_block_ids=[200, 201],
            chunking_strategy="agentic",
        ),
    ]

    optimized = optimize_retrieval_units(chunks, source_blocks)

    assert len(optimized) == 1
    assert optimized[0].source_block_ids == [196, 197, 198, 199, 200, 201]
    assert "Momentum conclusion" in optimized[0].content
    assert optimized[0].retrieval_value is None
    assert optimized[0].query_intents == []
    assert optimized[0].keywords == []
    assert optimized[0].planner_reason is None


def test_optimize_retrieval_units_splits_large_chunk_by_source_blocks() -> None:
    source_blocks = [
        block(1, "5.2 Model Construction and Training", long_text("Model principle", 250)),
        block(2, "5.2 Model Construction and Training", long_text("Training process", 250)),
        block(3, "5.2 Model Construction and Training", long_text("Hyperparameter result accuracy 0.92", 250)),
        block(4, "5.2 Model Construction and Training", long_text("Result analysis accuracy 0.94", 250)),
    ]
    chunk = Chunk(
        document_id=10,
        chunk_index=0,
        content="\n\n".join(item.content for item in source_blocks),
        token_count=1000,
        content_hash="large",
        chunk_type="retrieval_unit",
        section_name="5.2 Model Construction and Training",
        source_block_ids=[1, 2, 3, 4],
        chunking_strategy="agentic",
    )

    optimized = optimize_retrieval_units([chunk], source_blocks)

    assert len(optimized) == 2
    assert all(item.token_count <= 800 for item in optimized)
    assert sorted({block_id for item in optimized for block_id in (item.source_block_ids or [])}) == [1, 2, 3, 4]
    assert all(item.retrieval_value is None for item in optimized)
    assert all(item.query_intents == [] for item in optimized)


def test_optimize_retrieval_units_assigns_low_value_to_intro() -> None:
    chunk = make_chunk(
        1,
        "1 Introduction",
        "retrieval_unit",
        "## 1 Introduction\n\nThis background motivation explains the paper context.",
    )

    optimized = optimize_retrieval_units([chunk], [block(1, "1 Introduction", chunk.content)])

    assert optimized[0].retrieval_value is None
    assert optimized[0].query_intents == []
    assert optimized[0].chunk_type == "retrieval_unit"


def test_optimize_retrieval_units_assigns_low_value_to_abstract() -> None:
    chunk = make_chunk(
        1,
        "Abstract",
        "retrieval_unit",
        "## Abstract\n\nThis abstract mentions momentum returns and model performance.",
    )

    optimized = optimize_retrieval_units([chunk], [block(1, "Abstract", chunk.content)])

    assert optimized[0].retrieval_value is None
    assert optimized[0].query_intents == []


def test_optimize_retrieval_units_reindexes_by_document_order_not_section_name() -> None:
    chunks = [
        make_chunk(1, "Abstract", "retrieval_unit", "## Abstract\n\n" + long_text("Abstract body", 30)),
        make_chunk(
            2,
            "3.2 Short Term Reversion Factor Exploration",
            "retrieval_unit",
            "## 3.2 Short Term Reversion Factor Exploration\n\n" + long_text("Reversion body", 30),
        ),
    ]

    optimized = optimize_retrieval_units(chunks)

    assert [(chunk.chunk_index, chunk.section_name) for chunk in optimized] == [
        (0, "Abstract"),
        (1, "3.2 Short Term Reversion Factor Exploration"),
    ]


def test_optimize_retrieval_units_restores_missing_section_chunk() -> None:
    source_blocks = [
        block(1, "3.2 Short Term Reversion Factor Exploration", "3.2 Short Term Reversion Factor Exploration", block_type="section_title"),
        block(2, "3.2 Short Term Reversion Factor Exploration", long_text("Reversion overview", 30)),
        block(3, "3.2.1 Factor Construction", "3.2.1 Factor Construction", block_type="section_title"),
        block(4, "3.2.1 Factor Construction", long_text("Factor construction definition", 30)),
        block(5, "3.2.1 Factor Construction", long_text("Factor construction interval", 30)),
        block(6, "3.2.2 Calculate Factor Returns", "3.2.2 Calculate Factor Returns", block_type="section_title"),
        block(7, "3.2.2 Calculate Factor Returns", long_text("Factor return calculation", 30)),
    ]
    chunks = [
        make_chunk(1, "3.2 Short Term Reversion Factor Exploration", "retrieval_unit", source_blocks[0].content),
        make_chunk(6, "3.2.2 Calculate Factor Returns", "retrieval_unit", source_blocks[5].content),
    ]
    chunks[0] = chunks[0].__class__(**{**chunks[0].__dict__, "source_block_ids": [1, 2]})
    chunks[1] = chunks[1].__class__(**{**chunks[1].__dict__, "source_block_ids": [6, 7]})

    optimized = optimize_retrieval_units(chunks, source_blocks)

    restored = next(chunk for chunk in optimized if "## 3.2.1 Factor Construction" in chunk.content)
    assert {3, 4, 5}.issubset(set(restored.source_block_ids or []))


def test_optimize_retrieval_units_prepends_missing_heading_to_existing_section_chunk() -> None:
    source_blocks = [
        block(1, "4.7 Operating Margin & Profit Margin:", "4.7 Operating Margin & Profit Margin:", block_type="section_title"),
        block(2, "4.7 Operating Margin & Profit Margin:", long_text("Operating margin definition", 30)),
        block(3, "4.7 Operating Margin & Profit Margin:", long_text("For these two ratios", 30)),
    ]
    chunk = make_chunk(3, "4.7 Operating Margin & Profit Margin:", "retrieval_unit", source_blocks[2].content)
    chunk = chunk.__class__(**{**chunk.__dict__, "source_block_ids": [3]})

    optimized = optimize_retrieval_units([chunk], source_blocks)

    assert len(optimized) == 1
    assert optimized[0].content.startswith("## 4.7 Operating Margin & Profit Margin:")
    assert "Operating margin definition" in optimized[0].content
    assert optimized[0].source_block_ids == [1, 2, 3]


def test_body_chunk_not_interrupted_by_figure() -> None:
    source_blocks = [
        block(
            1,
            "3 Geometry Stage",
            "The geometry stage estimates (1) layout, (2) object pose, and (3)",
        ),
        block(2, "3 Geometry Stage", "Fig. 2: Geometry pipeline overview.", block_type="figure_caption"),
        block(3, "3 Geometry Stage", "structural edits before rendering. " + " ".join(["geometry"] * 80)),
    ]

    result = chunk_agentic_documents(
        [document()],
        source_blocks,
        model=FakePlanner([simple_plan([1, 3]), simple_plan([2], chunk_type="figure_caption", section="3 Geometry Stage")]),
        enabled=True,
    )

    body = next(chunk for chunk in result.chunks if chunk.chunk_type == "retrieval_unit")
    assert body.source_block_ids == [1, 3]
    assert "Fig. 2" not in body.content
    assert "and (3)\n\nstructural edits" in body.content


def test_body_chunks_rejoined_across_intervening_figure_caption() -> None:
    source_blocks = [
        block(1, "3.1 Geometry Stage", "Geometry stage performs three edits: (1) local shape edits, (2) geometric transforms, and (3)"),
        block(2, "3.1 Geometry Stage", "Fig. 2: Overview of geometry edits.", block_type="figure_caption"),
        block(3, "3.1 Geometry Stage", "structural edits, such as adding missing parts and revising object relations."),
    ]
    chunks = [
        make_chunk(1, "3.1 Geometry Stage", "retrieval_unit", source_blocks[0].content),
        make_chunk(2, "3.1 Geometry Stage", "figure_caption", source_blocks[1].content),
        make_chunk(3, "3.1 Geometry Stage", "retrieval_unit", source_blocks[2].content),
    ]

    optimized = optimize_retrieval_units(chunks, source_blocks)

    body = next(chunk for chunk in optimized if chunk.chunk_type == "retrieval_unit")
    figure = next(chunk for chunk in optimized if chunk.chunk_type == "figure_caption")
    assert body.source_block_ids == [1, 3]
    assert "Fig. 2" not in body.content
    assert "and (3)\n\nstructural edits" in body.content
    assert figure.source_block_ids == [2]


def test_no_chunk_ends_with_unfinished_enumeration() -> None:
    source_blocks = [
        block(1, "3 Geometry Stage", " ".join(["geometry"] * 790) + " and (3)"),
        block(2, "3 Geometry Stage", "structural edits " + " ".join(["rendering"] * 40)),
    ]
    chunk = Chunk(
        document_id=10,
        chunk_index=0,
        content="\n\n".join(item.content for item in source_blocks),
        token_count=840,
        content_hash="unfinished-enum",
        chunk_type="retrieval_unit",
        section_name="3 Geometry Stage",
        source_block_ids=[1, 2],
        chunking_strategy="agentic",
    )

    optimized = optimize_retrieval_units([chunk], source_blocks)

    assert all(not item.content.rstrip().endswith("and (3)") for item in optimized)


def test_figure_chunk_is_atomic() -> None:
    source_blocks = [
        block(1, "4 Results", long_text("The body explicitly discusses Fig. 2 geometry results", 80)),
        block(2, "4 Results", "Fig. 2: Geometry result examples.", block_type="figure_caption"),
    ]

    result = chunk_agentic_documents(
        [document()],
        source_blocks,
        model=FakePlanner([simple_plan([1], section="4 Results"), simple_plan([2], chunk_type="figure_caption", section="4 Results")]),
        enabled=True,
    )

    figure = next(chunk for chunk in result.chunks if chunk.chunk_type == "figure_caption")
    assert figure.source_block_ids == [2]
    assert figure.content == "Fig. 2: Geometry result examples."


def test_fused_chunk_keeps_source_chunk_ids() -> None:
    source_blocks = [
        block(1, "4 Results", long_text("As shown in Fig. 2, geometry editing improves structural consistency", 80)),
        block(2, "4 Results", "Fig. 2: Geometry editing examples with structural consistency.", block_type="figure_caption"),
    ]

    result = chunk_agentic_documents(
        [document()],
        source_blocks,
        model=FakePlanner([simple_plan([1], section="4 Results"), simple_plan([2], chunk_type="figure_caption", section="4 Results")]),
        enabled=True,
    )

    fused = next(chunk for chunk in result.chunks if chunk.chunk_type == "fused")
    original_hashes = {
        chunk.content_hash
        for chunk in result.chunks
        if chunk.chunk_type in {"retrieval_unit", "figure_caption", "table"}
    }
    assert fused.metadata is not None
    assert len(fused.metadata["source_chunk_ids"]) == 2
    assert set(fused.metadata["source_chunk_ids"]).issubset(original_hashes)
    assert fused.metadata["visual_refs"] == ["Fig. 2"]
    assert fused.metadata["fusion_confidence"] in {"high", "medium", "low"}


def test_fused_chunk_dedupes_same_body_visual_pair() -> None:
    source_blocks = [
        block(1, "4 Results", long_text("As shown in Fig. 2, geometry editing improves structural consistency", 80)),
        block(2, "4 Results", "Fig. 2: Geometry editing examples with structural consistency.", block_type="figure_caption"),
    ]
    body = make_chunk(1, "4 Results", "retrieval_unit", source_blocks[0].content)
    visual = make_chunk(2, "4 Results", "figure_caption", source_blocks[1].content)

    optimized = optimize_retrieval_units([body, visual, visual], source_blocks)

    fused = [chunk for chunk in optimized if chunk.chunk_type == "fused"]
    assert len(fused) == 1


def test_similar_fused_chunks_are_deduped() -> None:
    source_blocks = [
        block(1, "4.4 Applications", long_text("Applications use Fig. 5 and Fig. 6 for editable indoor scene reconstruction", 80), page_number=7),
        block(2, "4.4 Applications", "Fig. 5: Editable indoor scene reconstruction applications.", block_type="figure_caption", page_number=7),
        block(3, "4.4 Applications", "Fig. 6: Editable indoor scene reconstruction applications.", block_type="figure_caption", page_number=7),
    ]

    optimized = optimize_retrieval_units(
        [
            make_chunk(1, "4.4 Applications", "retrieval_unit", source_blocks[0].content),
            make_chunk(2, "4.4 Applications", "figure_caption", source_blocks[1].content),
            make_chunk(3, "4.4 Applications", "figure_caption", source_blocks[2].content),
        ],
        source_blocks,
    )

    assert len([chunk for chunk in optimized if chunk.chunk_type == "fused"]) == 1


def test_figure_bound_to_semantic_section() -> None:
    source_blocks = [
        block(1, "3 Method", long_text("The geometry pipeline estimates layout and camera pose", 80), page_number=4),
        block(2, "4.3 Qualitative Results", long_text("Qualitative examples show structural edits and visual consistency", 80), page_number=5),
        block(3, "4.3 Qualitative Results", "Fig. 7: Qualitative examples of structural edits and visual consistency.", block_type="figure_caption", page_number=5),
        block(4, "5 Conclusion", long_text("Conclusion summarizes structural edits and visual consistency", 80), page_number=6),
        block(5, "5 Conclusion", "Fig. 8: Qualitative examples of structural edits and visual consistency.", block_type="figure_caption", page_number=6),
    ]

    optimized = optimize_retrieval_units(
        [
            make_chunk(1, "3 Method", "retrieval_unit", source_blocks[0].content),
            make_chunk(2, "4.3 Qualitative Results", "retrieval_unit", source_blocks[1].content),
            make_chunk(3, "4.3 Qualitative Results", "figure_caption", source_blocks[2].content),
            make_chunk(4, "5 Conclusion", "retrieval_unit", source_blocks[3].content),
            make_chunk(5, "5 Conclusion", "figure_caption", source_blocks[4].content),
        ],
        source_blocks,
    )

    fused = [chunk for chunk in optimized if chunk.chunk_type == "fused"]
    assert len(fused) == 1
    assert fused[0].section_name == "4.3 Qualitative Results"
    assert fused[0].metadata["visual_refs"] == ["Fig. 7"]


def test_explicit_visual_binding_prefers_same_section_over_later_limitations() -> None:
    source_blocks = [
        block(1, "4.4 Applications", long_text("Applications use Fig. 5 to show editable indoor scene reconstruction", 80), page_number=7),
        block(2, "4.4 Applications", "Fig. 5: Applications for editable indoor scene reconstruction.", block_type="figure_caption", page_number=7),
        block(3, "4.5 Limitations", long_text("Limitations mention Fig. 5 but focus on failure cases and ambiguity", 80), page_number=8),
    ]
    optimized = optimize_retrieval_units(
        [
            make_chunk(1, "4.4 Applications", "retrieval_unit", source_blocks[0].content),
            make_chunk(2, "4.4 Applications", "figure_caption", source_blocks[1].content),
            make_chunk(3, "4.5 Limitations", "retrieval_unit", source_blocks[2].content),
        ],
        source_blocks,
    )

    fused = [chunk for chunk in optimized if chunk.chunk_type == "fused"]

    assert len(fused) == 1
    assert fused[0].section_name == "4.4 Applications"
    assert fused[0].metadata["source_body_block_ids"] == [1]


def test_original_chunks_preserved_after_fusion() -> None:
    source_blocks = [
        block(1, "4 Results", long_text("Table 2 reports accuracy and recall improvements", 80)),
        block(
            2,
            "4 Results",
            "Table 2: Accuracy results.\n| Model | Accuracy |\n| --- | --- |\n| A | 0.91 |",
            block_type="table",
            markdown_content="Table 2: Accuracy results.\n| Model | Accuracy |\n| --- | --- |\n| A | 0.91 |",
        ),
    ]

    result = chunk_agentic_documents(
        [document()],
        source_blocks,
        model=FakePlanner([simple_plan([1], section="4 Results"), simple_plan([2], chunk_type="table", section="4 Results")]),
        enabled=True,
    )

    assert any(chunk.chunk_type == "retrieval_unit" and chunk.source_block_ids == [1] for chunk in result.chunks)
    assert any(chunk.chunk_type == "table" and chunk.source_block_ids == [2] for chunk in result.chunks)
    assert any(chunk.chunk_type == "fused" for chunk in result.chunks)


def test_mixed_body_table_chunk_is_atomized_before_fusion() -> None:
    source_blocks = [
        block(1, "4 CLINENV Benchmark", "4 CLINENV Benchmark", block_type="section_title"),
        block(2, "4 CLINENV Benchmark", long_text("The benchmark samples admissions and reports stage counts in Table 4", 80)),
        block(
            3,
            "4 CLINENV Benchmark",
            "1 CLINENV Huggingface Link. [2 CLINENV GitHub Link.](https://github.com/example) Table 2: Clinical information agents and their readviews.",
        ),
        block(
            4,
            "4 CLINENV Benchmark",
            "| Agent | Readview | Role |\n| --- | --- | --- |\n| Patient | Demographics | Simulates patient reporting |",
            block_type="table",
            markdown_content="| Agent | Readview | Role |\n| --- | --- | --- |\n| Patient | Demographics | Simulates patient reporting |",
        ),
    ]
    mixed = Chunk(
        document_id=10,
        chunk_index=1,
        content="\n\n".join(block.markdown_content or block.content for block in source_blocks),
        token_count=160,
        content_hash="mixed-body-table",
        chunk_type="retrieval_unit",
        section_name="4 CLINENV Benchmark",
        source_block_ids=[1, 2, 3, 4],
        chunking_strategy="agentic",
    )

    optimized = optimize_retrieval_units([mixed], source_blocks)

    body_chunks = [chunk for chunk in optimized if chunk.chunk_type == "retrieval_unit"]
    table_chunks = [chunk for chunk in optimized if chunk.chunk_type == "table"]
    fused_chunks = [chunk for chunk in optimized if chunk.chunk_type == "fused"]
    assert len(body_chunks) == 1
    assert "Table 2: Clinical information agents" not in body_chunks[0].content
    assert "| Agent | Readview | Role |" not in body_chunks[0].content
    assert len(table_chunks) == 1
    assert "Table 2: Clinical information agents" in table_chunks[0].content
    assert "| Agent | Readview | Role |" in table_chunks[0].content
    assert all((chunk.metadata or {}).get("source_visual_block_ids") == [3, 4] for chunk in fused_chunks)


def test_duplicate_table_ref_keeps_single_best_table_chunk_and_marks_low_confidence() -> None:
    source_blocks = [
        block(1, "4 Results", "Table 1: Main results\n| Model | Acc |\n| --- | --- |\n| A | 0.90 |", block_type="table", markdown_content="Table 1: Main results\n| Model | Acc |\n| --- | --- |\n| A | 0.90 |"),
        block(2, "4 Results", "Table 1: Main results\n| Model | Acc | Extra |\n| --- | --- |\n| A | 0.90 | broken |", block_type="table", markdown_content="Table 1: Main results\n| Model | Acc | Extra |\n| --- | --- |\n| A | 0.90 | broken |"),
    ]
    optimized = optimize_retrieval_units(
        [
            make_chunk(1, "4 Results", "table", source_blocks[0].content),
            make_chunk(2, "4 Results", "table", source_blocks[1].content),
        ],
        source_blocks,
    )

    table_chunks = [chunk for chunk in optimized if chunk.chunk_type == "table"]

    assert len(table_chunks) == 1
    assert table_chunks[0].source_block_ids == [1]
    assert table_chunks[0].metadata["table_confidence"] == "high"
    assert table_confidence_for_test(source_blocks[1].content) == "low"


def table_confidence_for_test(content: str) -> str:
    from ragarena.chunking.agentic_chunker import table_confidence

    return table_confidence(content)


def test_debug_report_groups_chunk_types(capsys) -> None:
    chunks = [
        make_chunk(1, "4 Results", "retrieval_unit", "Body text explains Fig. 2."),
        make_chunk(2, "4 Results", "figure_caption", "Fig. 2: Visual evidence."),
        Chunk(
            document_id=10,
            chunk_index=3,
            content="Body text explains Fig. 2.\n\nRelated visual/table evidence:\nFig. 2: Visual evidence.",
            token_count=12,
            content_hash="fused-report",
            chunk_type="fused",
            section_name="4 Results",
            source_block_ids=[1, 2],
            chunking_strategy="agentic_fusion",
            metadata={
                "semantic_chunk_type": "fused",
                "source_chunk_ids": ["body-hash", "visual-hash"],
                "visual_refs": ["Fig. 2"],
                "fusion_confidence": "high",
            },
        ),
    ]

    print_chunk_quality_report(chunks)

    output = capsys.readouterr().out
    assert "BODY CHUNKS" in output
    assert "VISUAL CHUNKS" in output
    assert "FUSED CHUNKS" in output
    assert output.index("BODY CHUNKS") < output.index("VISUAL CHUNKS") < output.index("FUSED CHUNKS")
