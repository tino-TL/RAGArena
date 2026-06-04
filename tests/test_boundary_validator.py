from __future__ import annotations

from dataclasses import dataclass

from ragarena.chunking.boundary_validator import (
    detect_section_leakage,
    derive_section_name_from_content,
    extract_section_name_from_heading,
    find_trailing_section_heading,
    rebuild_chunk_with_content,
    resolve_section_name,
    validate_section_name_consistency,
    validate_chunk_boundaries,
)
from ragarena.chunking.fixed_chunker import Chunk, estimate_token_count
from ragarena.ingestion.hashing import content_hash
from ragarena.papers.models import PaperBlock


def block(block_id: int, section: str, text: str, block_type: str = "paragraph") -> PaperBlock:
    return PaperBlock(
        id=block_id,
        paper_id=1,
        arxiv_id="2401.00001v1",
        block_type=block_type,
        section_name=section,
        page_number=1,
        content=text,
        markdown_content=None,
        image_path=None,
        order_index=block_id,
        should_embed=True,
        metadata={},
        content_hash=f"b{block_id}",
    )


def chunk(chunk_index: int, ids: list[int], text: str, chunk_type: str, section: str) -> Chunk:
    return Chunk(
        document_id=10,
        chunk_index=chunk_index,
        content=text,
        token_count=estimate_token_count(text),
        content_hash=content_hash(text),
        chunk_type=chunk_type,
        section_name=section,
        source_block_ids=ids,
        chunking_strategy="agentic",
    )


def long_text(prefix: str, words: int = 90) -> str:
    return f"{prefix} " + " ".join(["retrieval"] * words)


def test_abstract_chunk_containing_introduction_title_moves_title_forward() -> None:
    blocks = [
        block(1, "Abstract", long_text("Abstract content"), "abstract"),
        block(2, "Introduction", "Introduction", "section_title"),
        block(3, "Introduction", long_text("Introduction content"), "paragraph"),
    ]
    chunks = [
        chunk(1, [1, 2], "bad abstract with intro title", "abstract", "Abstract"),
        chunk(2, [3], "intro paragraph", "paragraph", "Introduction"),
    ]

    result = validate_chunk_boundaries(chunks, blocks)

    assert result.chunks[0].source_block_ids == [1]
    assert result.chunks[1].source_block_ids == [2, 3]
    assert "Introduction" not in result.chunks[0].content
    assert result.stats.rule_fixes >= 1


def test_section_title_does_not_backward_attach() -> None:
    blocks = [
        block(1, "Abstract", long_text("Abstract content"), "abstract"),
        block(2, "Introduction", "Introduction", "section_title"),
        block(3, "Introduction", long_text("Introduction content"), "paragraph"),
    ]
    chunks = [
        chunk(1, [1], "abstract", "abstract", "Abstract"),
        chunk(2, [2, 3], "intro", "paragraph", "Introduction"),
    ]

    result = validate_chunk_boundaries(chunks, blocks)

    assert result.chunks[0].source_block_ids == [1]
    assert result.chunks[1].source_block_ids == [2, 3]


def test_tiny_metadata_chunk_is_dropped() -> None:
    chunks = [
        chunk(1, [], "Jane Doe, Example University", "metadata", "unknown"),
        chunk(2, [], long_text("Introduction content"), "paragraph", "Introduction"),
    ]

    result = validate_chunk_boundaries(chunks, [])

    assert len(result.chunks) == 1
    assert "Jane Doe" not in result.chunks[0].content
    assert result.stats.dropped_chunks == 1


def test_caption_chunk_stays_standalone() -> None:
    blocks = [
        block(1, "Results", "Figure 1: short caption.", "figure_caption"),
        block(2, "Results", long_text("The figure shows the retrieval result"), "paragraph"),
    ]
    chunks = [
        chunk(1, [1], "Figure 1: short caption.", "figure_caption", "Results"),
        chunk(2, [2], "paragraph", "paragraph", "Results"),
    ]

    result = validate_chunk_boundaries(chunks, blocks)

    assert len(result.chunks) == 2
    assert result.chunks[0].source_block_ids == [1]
    assert result.chunks[1].source_block_ids == [2]


def test_running_header_line_is_removed_from_chunk_boundary() -> None:
    content = (
        "## 4 \u00b7 Guangzhao He, Rundong Luo, Wei-Chiu Ma, and Hadar Averbuch-Elor\n\n"
        + long_text("Method body")
    )
    blocks = [block(1, "Method", content, "paragraph")]
    chunks = [
        chunk(
            1,
            [1],
            content,
            "paragraph",
            "Method",
        ),
    ]

    result = validate_chunk_boundaries(chunks, blocks)

    assert len(result.chunks) == 1
    assert "Guangzhao He" not in result.chunks[0].content
    assert result.chunks[0].content.startswith("Method body")
    assert result.stats.rule_fixes >= 1


def test_bad_sentence_boundary_merges_with_next_body_chunk_across_caption() -> None:
    blocks = [
        block(1, "Method", "Geometry stage makes three kinds of edits: (1) semantic edits, (2) relation edits, and (3)", "paragraph"),
        block(2, "Method", "Figure 2: Pipeline overview.", "figure_caption"),
        block(3, "Method", long_text("structural edits add missing parts and adjust object placement"), "paragraph"),
    ]
    chunks = [
        chunk(1, [1], blocks[0].content, "paragraph", "Method"),
        chunk(2, [2], blocks[1].content, "figure_caption", "Method"),
        chunk(3, [3], blocks[2].content, "paragraph", "Method"),
    ]

    result = validate_chunk_boundaries(chunks, blocks)

    assert len(result.chunks) == 2
    body_chunk = next(item for item in result.chunks if item.source_block_ids == [1, 3])
    caption_chunk = next(item for item in result.chunks if item.source_block_ids == [2])
    assert "and (3)" in body_chunk.content
    assert "structural edits" in body_chunk.content
    assert caption_chunk.content.startswith("Figure 2")
    assert result.stats.rule_fixes >= 1


def test_no_chunk_ends_with_and_numbered_item() -> None:
    blocks = [
        block(1, "3.1 Staged Scene Construction", "Geometry stage performs three edits: (1) local shape edits, (2) geometric transforms, and (3)", "paragraph"),
        block(2, "3.1 Staged Scene Construction", "Fig. 2: Overview of the staged construction pipeline.", "figure_caption"),
        block(3, "3.1 Staged Scene Construction", "structural edits, such as adding missing parts and revising object relations.", "paragraph"),
    ]
    chunks = [
        chunk(1, [1], blocks[0].content, "paragraph", "3.1 Staged Scene Construction"),
        chunk(2, [2], blocks[1].content, "figure_caption", "3.1 Staged Scene Construction"),
        chunk(3, [3], blocks[2].content, "paragraph", "3.1 Staged Scene Construction"),
    ]

    result = validate_chunk_boundaries(chunks, blocks)
    body_chunk = next(item for item in result.chunks if item.source_block_ids == [1, 3])

    assert not body_chunk.content.rstrip().endswith("and (3)")
    assert "(1) local shape edits" in body_chunk.content
    assert "(2) geometric transforms" in body_chunk.content
    assert "(3)\n\nstructural edits" in body_chunk.content


def test_quantitative_section_not_include_qualitative_results() -> None:
    quantitative = "## 4.2 Quantitative Results\n\nTable 2 reports numeric scores across baselines."
    qualitative = "## 4.3 Qualitative Results\n\nFig. 8: Qualitative examples from Blender scenes."
    chunks = [
        chunk(1, [1], f"{quantitative}\n\n{qualitative}", "paragraph", "4.2 Quantitative Results"),
    ]

    result = validate_chunk_boundaries(chunks, [])

    assert len(result.chunks) == 2
    assert result.chunks[0].section_name == "4.2 Quantitative Results"
    assert "Fig. 8" not in result.chunks[0].content
    assert result.chunks[1].section_name == "4.3 Qualitative Results"
    assert "Fig. 8" in result.chunks[1].content


def test_adjacent_duplicate_body_chunk_is_removed() -> None:
    text = long_text("The evaluation interface exposes direct and interactive modes")
    chunks = [
        chunk(1, [1], text, "paragraph", "3.3 Multi-Source Clinical Environment"),
        chunk(2, [2], f"{text} extra", "paragraph", "3.3 Multi-Source Clinical Environment"),
    ]

    result = validate_chunk_boundaries(chunks, [])

    assert len(result.chunks) == 1
    assert result.stats.dropped_chunks == 1


def test_adjacent_section_duplicate_keeps_better_matching_section() -> None:
    duplicated = long_text("Direct Mode and Interactive Mode expose a structured submission interface")
    chunks = [
        chunk(1, [1], duplicated, "paragraph", "3.3 Multi-Source Clinical Environment"),
        chunk(2, [2], f"Interaction Modes\n\n{duplicated}", "paragraph", "3.4 Interaction Modes"),
    ]

    result = validate_chunk_boundaries(chunks, [])

    assert len(result.chunks) == 1
    assert result.chunks[0].section_name == "3.4 Interaction Modes"
    assert result.stats.dropped_chunks == 1


def test_no_duplicate_adjacent_sections() -> None:
    duplicated = (
        "The evaluation interface exposes Direct Mode and Interactive Mode through a structured submission interface. "
        "Direct Mode evaluates final answers, while Interactive Mode supports iterative clinical environment queries."
    )
    chunks = [
        chunk(1, [1], duplicated, "paragraph", "3.3 Multi-Source Clinical Environment"),
        chunk(2, [2], f"## 3.4 Interaction Modes\n\n{duplicated}", "paragraph", "3.4 Interaction Modes"),
    ]

    result = validate_chunk_boundaries(chunks, [])

    assert len(result.chunks) == 1
    assert result.chunks[0].section_name == "3.4 Interaction Modes"


@dataclass(frozen=True)
class InvalidModelResponse:
    answer: str


class InvalidBoundaryModel:
    def generate(self, prompt: str, system_prompt: str) -> InvalidModelResponse:
        return InvalidModelResponse(answer="not-json")


def test_model_invalid_output_does_not_affect_pipeline() -> None:
    blocks = [
        block(1, "Results", long_text("A valid result chunk"), "paragraph"),
    ]
    chunks = [
        chunk(1, [1], "valid result chunk", "paragraph", "Results"),
    ]

    result = validate_chunk_boundaries(chunks, blocks, model=InvalidBoundaryModel(), use_model=True)

    assert len(result.chunks) == 1
    assert result.stats.model_fixes == 0


def test_trailing_introduction_heading_removed_from_abstract_and_prepended_to_next_chunk() -> None:
    abstract_content = f"{long_text('Abstract content')}\n\nKeywords: retrieval\n## 1 Introduction"
    intro_content = long_text("Introduction body")
    chunks = [
        chunk(1, [1], abstract_content, "abstract", "Abstract"),
        chunk(2, [2], intro_content, "paragraph", "1 Introduction"),
    ]

    result = validate_chunk_boundaries(chunks, [])

    assert "## 1 Introduction" not in result.chunks[0].content
    assert result.chunks[1].content.startswith("## 1 Introduction\n\n")
    assert result.stats.boundary_issues_found >= 1
    assert result.stats.rule_fixes >= 1


def test_trailing_method_heading_removed_from_introduction() -> None:
    intro_content = f"{long_text('Introduction body')}\n\n## 2 Method"
    method_content = long_text("Method body")
    chunks = [
        chunk(1, [], intro_content, "paragraph", "Introduction"),
        chunk(2, [], method_content, "paragraph", "2 Method"),
    ]

    result = validate_chunk_boundaries(chunks, [])

    assert not result.chunks[0].content.rstrip().endswith("## 2 Method")
    assert result.chunks[1].content.startswith("## 2 Method")


def test_trailing_heading_is_deleted_when_no_matching_next_chunk() -> None:
    content = f"{long_text('Current section body')}\n\n## 4 Missing Section"
    chunks = [
        chunk(1, [], content, "paragraph", "Current"),
        chunk(2, [], long_text("Unrelated body"), "paragraph", "Appendix"),
    ]

    result = validate_chunk_boundaries(chunks, [])

    assert "## 4 Missing Section" not in result.chunks[0].content
    assert not result.chunks[1].content.startswith("## 4 Missing Section")


def test_heading_at_chunk_start_is_preserved() -> None:
    content = f"## 1 Introduction\n\n{long_text('Introduction body')}"
    chunks = [chunk(1, [], content, "paragraph", "1 Introduction")]

    result = validate_chunk_boundaries(chunks, [])

    assert result.chunks[0].content.startswith("## 1 Introduction")


def test_heading_in_chunk_middle_is_preserved() -> None:
    content = f"{long_text('Before heading')}\n\n## 1 Introduction\n\n{long_text('After heading')}"
    chunks = [chunk(1, [], content, "paragraph", "Mixed")]

    result = validate_chunk_boundaries(chunks, [])

    assert "## 1 Introduction" in result.chunks[0].content


def test_small_markdown_heading_is_preserved() -> None:
    content = f"{long_text('Main body')}\n\n### 1.1 Small Heading"
    chunks = [chunk(1, [], content, "paragraph", "Main")]

    result = validate_chunk_boundaries(chunks, [])

    assert result.chunks[0].content.rstrip().endswith("### 1.1 Small Heading")


def test_unnumbered_markdown_heading_is_preserved() -> None:
    content = f"{long_text('Main body')}\n\n## Important Observation"
    chunks = [chunk(1, [], content, "paragraph", "Main")]

    result = validate_chunk_boundaries(chunks, [])

    assert result.chunks[0].content.rstrip().endswith("## Important Observation")


def test_trailing_heading_fix_updates_token_count_hash_and_preserves_source_ids() -> None:
    content = f"{long_text('Main body')}\n\n## 3 Results"
    original = chunk(1, [42], content, "paragraph", "Main")

    result = validate_chunk_boundaries([original], [])

    assert result.chunks[0].source_block_ids == [42]
    assert result.chunks[0].token_count != original.token_count
    assert result.chunks[0].content_hash != original.content_hash


def test_boundary_validator_removes_leading_page_number_line() -> None:
    original = chunk(1, [42], f"1\n\n## Abstract\n\n{long_text('Abstract body')}", "abstract", "Abstract")

    result = validate_chunk_boundaries([original], [])

    assert result.chunks[0].content.startswith("## Abstract")
    assert not result.chunks[0].content.startswith("1\n\n")
    assert result.chunks[0].source_block_ids == [42]
    assert result.chunks[0].token_count != original.token_count
    assert result.chunks[0].content_hash != original.content_hash


def test_detect_section_leakage_only_detects_trailing_numbered_h2() -> None:
    assert find_trailing_section_heading(f"{long_text('Body')}\n## 3.1.1 Factor Construction")
    assert detect_section_leakage(chunk(1, [], f"{long_text('Body')}\n## 2 Method", "paragraph", "Body"))
    assert not detect_section_leakage(chunk(1, [], f"## 2 Method\n\n{long_text('Body')}", "paragraph", "Body"))
    assert not detect_section_leakage(chunk(1, [], f"{long_text('Body')}\n### 2.1 Detail", "paragraph", "Body"))
    assert not detect_section_leakage(chunk(1, [], f"{long_text('Body')}\n## Important", "paragraph", "Body"))


def test_trailing_heading_updates_next_chunk_section_name_even_when_stale() -> None:
    chunks = [
        chunk(1, [], f"{long_text('Abstract body')}\n\n## 1 Introduction", "abstract", "Abstract"),
        chunk(2, [], long_text("Introduction body"), "paragraph", "Abstract"),
    ]

    result = validate_chunk_boundaries(chunks, [])

    assert result.chunks[0].section_name == "Abstract"
    assert "## 1 Introduction" not in result.chunks[0].content
    assert result.chunks[1].section_name == "1 Introduction"
    assert result.chunks[1].content.startswith("## 1 Introduction\n\n")


def test_pb_ratio_heading_updates_next_chunk_section_name() -> None:
    chunks = [
        chunk(1, [], f"{long_text('PE body')}\n\n## 4.2 PB Ratio", "paragraph", "4.1 PE Ratio"),
        chunk(2, [], long_text("PB body"), "paragraph", "4.1 PE Ratio"),
    ]

    result = validate_chunk_boundaries(chunks, [])

    assert result.chunks[0].section_name == "4.1 PE Ratio"
    assert result.chunks[1].section_name == "4.2 PB Ratio"
    assert result.chunks[1].content.startswith("## 4.2 PB Ratio\n\n")


def test_missing_next_chunk_does_not_change_section_name() -> None:
    original = chunk(1, [], f"{long_text('Current body')}\n\n## 7 Future Work", "paragraph", "Current")

    result = validate_chunk_boundaries([original], [])

    assert result.chunks[0].section_name == "Current"
    assert "## 7 Future Work" not in result.chunks[0].content


def test_prepend_updates_next_chunk_hash_token_count_and_preserves_ids() -> None:
    next_chunk = chunk(2, [10, 11], long_text("PB body"), "paragraph", "4.1 PE Ratio")
    chunks = [
        chunk(1, [9], f"{long_text('PE body')}\n\n## 4.2 PB Ratio", "paragraph", "4.1 PE Ratio"),
        next_chunk,
    ]

    result = validate_chunk_boundaries(chunks, [])

    assert result.chunks[1].section_name == "4.2 PB Ratio"
    assert result.chunks[1].source_block_ids == [10, 11]
    assert result.chunks[1].chunk_index == next_chunk.chunk_index
    assert result.chunks[1].document_id == next_chunk.document_id
    assert result.chunks[1].content_hash != next_chunk.content_hash
    assert result.chunks[1].token_count != next_chunk.token_count


def test_boundary_validator_rebuild_keeps_agentic_chunk_type_retrieval_unit() -> None:
    blocks = [
        block(1, "Abstract", long_text("Abstract content"), "abstract"),
        block(2, "Introduction", "Introduction", "section_title"),
        block(3, "Introduction", long_text("Introduction content"), "paragraph"),
    ]
    chunks = [
        chunk(1, [1, 2], "bad abstract with intro title", "code", "Abstract"),
        chunk(2, [3], long_text("intro paragraph"), "table", "Introduction"),
    ]

    result = validate_chunk_boundaries(chunks, blocks)

    assert all(chunk.chunk_type == "retrieval_unit" for chunk in result.chunks)
    assert result.chunks[0].source_block_ids == [1]
    assert result.chunks[1].source_block_ids == [2, 3]


def test_extract_section_name_from_heading() -> None:
    assert extract_section_name_from_heading("## 4.2 PB Ratio") == "4.2 PB Ratio"


def test_validate_section_name_consistency() -> None:
    valid = chunk(1, [], f"## 4.2 PB Ratio\n\n{long_text('PB body')}", "paragraph", "4.2 PB Ratio")
    invalid = chunk(1, [], f"## 4.2 PB Ratio\n\n{long_text('PB body')}", "paragraph", "4.1 PE Ratio")
    no_heading = chunk(1, [], long_text("PB body"), "paragraph", "4.1 PE Ratio")

    assert validate_section_name_consistency(valid) is True
    assert validate_section_name_consistency(invalid) is False
    assert validate_section_name_consistency(no_heading) is True


def test_resolve_section_name_prefers_content_heading_over_planner_metadata() -> None:
    resolved = resolve_section_name(
        f"## 4.2 PB Ratio\n\n{long_text('PB body')}",
        "4.1 PE Ratio",
        None,
    )

    assert resolved == "4.2 PB Ratio"


def test_resolve_section_name_derives_abstract_from_content() -> None:
    resolved = resolve_section_name(
        f"## Abstract\n\n{long_text('Abstract body')}",
        "unknown",
        None,
    )

    assert resolved == "Abstract"


def test_resolve_section_name_uses_planner_when_content_has_no_heading() -> None:
    assert resolve_section_name(long_text("Normal paragraph"), "Method", "Fallback") == "Method"


def test_rebuild_chunk_with_content_syncs_section_name_from_new_content() -> None:
    original = chunk(2, [10], long_text("PB body"), "paragraph", "4.1 PE Ratio")

    rebuilt = rebuild_chunk_with_content(
        original,
        f"## 4.2 PB Ratio\n\n{long_text('PB body')}",
    )

    assert rebuilt.section_name == "4.2 PB Ratio"
    assert rebuilt.source_block_ids == [10]
    assert rebuilt.chunk_index == original.chunk_index
    assert rebuilt.document_id == original.document_id
    assert rebuilt.content_hash != original.content_hash
    assert rebuilt.token_count != original.token_count


def test_derive_section_name_from_content() -> None:
    assert derive_section_name_from_content("## 4.4 EV/EBIT &amp; EV/EBITDA\n\nbody") == "4.4 EV/EBIT & EV/EBITDA"
    assert derive_section_name_from_content("Normal paragraph") is None
