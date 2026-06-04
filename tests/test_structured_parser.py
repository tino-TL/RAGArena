from __future__ import annotations

from pathlib import Path

from ragarena.papers.models import PaperFile
from ragarena.papers.structured_parser import (
    TextLine,
    build_paper_block,
    classify_block,
    coalesce_text_lines,
    sort_page_blocks_for_reading,
)


def paper_file() -> PaperFile:
    return PaperFile(
        paper_id=1,
        arxiv_id="2401.00001v1",
        pdf_url="https://arxiv.org/pdf/2401.00001v1",
        file_path=Path("paper.pdf"),
        file_sha256="sha",
        file_size=123,
    )


def test_structured_parser_identifies_section_title() -> None:
    assert classify_block("Introduction", order_index=1) == "section_title"


def test_structured_parser_identifies_figure_caption() -> None:
    assert classify_block("Figure 1: System architecture", order_index=1) == "figure_caption"
    assert classify_block("Fig. 1: Retrieval flow", order_index=1) == "figure_caption"


def test_structured_parser_identifies_table_caption() -> None:
    assert classify_block("Table 1: Evaluation results", order_index=1) == "table_caption"


def test_structured_parser_identifies_code_block() -> None:
    assert classify_block("def search(query):\n    return query", order_index=1) == "code"


def test_structured_parser_does_not_identify_bracket_fragment_as_code() -> None:
    assert classify_block("[...]", order_index=1) == "paragraph"


def test_structured_parser_identifies_algorithm_block() -> None:
    assert classify_block("Algorithm 1: Query Rewrite", order_index=1) == "algorithm"


def test_structured_parser_identifies_formula_block() -> None:
    assert classify_block("y = α + βx + λ", order_index=1) == "formula"


def test_reference_block_should_not_embed() -> None:
    block = build_paper_block(
        paper_file=paper_file(),
        block_type="reference",
        section_name="References",
        page_number=10,
        content="[1] Smith et al. Retrieval systems.",
        markdown_content=None,
        image_path=None,
        order_index=1,
    )

    assert block.should_embed is False


def test_unknown_block_type_does_not_crash() -> None:
    block = build_paper_block(
        paper_file=paper_file(),
        block_type="unexpected_type",
        section_name=None,
        page_number=1,
        content="unclassified content",
        markdown_content=None,
        image_path=None,
        order_index=1,
    )

    assert block.block_type == "unknown"
    assert block.should_embed is False


def test_sort_page_blocks_for_reading_prefers_columns_before_y_order() -> None:
    class Rect:
        width = 600

    class Page:
        rect = Rect()

    blocks = [
        (320, 100, 560, 120, "right top"),
        (30, 180, 270, 200, "left bottom"),
        (30, 100, 270, 120, "left top"),
        (320, 180, 560, 200, "right bottom"),
    ]

    ordered = sort_page_blocks_for_reading(Page(), blocks)

    assert [block[4] for block in ordered] == ["left top", "left bottom", "right top", "right bottom"]


def test_related_work_no_split_write_render_compare_revise_loop() -> None:
    class Rect:
        width = 600

    class Page:
        rect = Rect()

    blocks = [
        (30, 100, 270, 120, "Closest to our work, VIGA follows a write-render-compare-revise"),
        (30, 150, 270, 170, "Recent works explore using vision-language models for scene editing."),
        (30, 210, 270, 230, "loop, enabling executable 3D scene reconstruction from feedback."),
        (320, 100, 560, 120, "Right column content starts here."),
    ]

    ordered = sort_page_blocks_for_reading(Page(), blocks)
    texts = [block[4] for block in ordered]

    assert texts.index("Closest to our work, VIGA follows a write-render-compare-revise") + 1 == texts.index(
        "loop, enabling executable 3D scene reconstruction from feedback."
    )


def test_structured_parser_filters_author_affiliation_email_blocks() -> None:
    blocks = coalesce_text_lines(
        paper_file(),
        [
            TextLine(text="Example Paper Title", page_number=1),
            TextLine(text="Alice Smith, Bob Jones", page_number=1),
            TextLine(text="Department of Computer Science, Example University", page_number=1),
            TextLine(text="alice@example.edu", page_number=1),
            TextLine(text="Abstract", page_number=1),
            TextLine(text="This paper studies retrieval augmented generation with structured chunks.", page_number=1),
        ],
    )

    contents = "\n".join(block.content for block in blocks)

    assert "Alice Smith" not in contents
    assert "Example University" not in contents
    assert "alice@example.edu" not in contents
    assert any(block.block_type == "abstract" for block in blocks)


def test_structured_parser_splits_metadata_prefixed_table_caption() -> None:
    blocks = coalesce_text_lines(
        paper_file(),
        [
            TextLine(text="Example Paper Title", page_number=1),
            TextLine(text="4 Benchmark", page_number=2),
            TextLine(
                text="1 CLINENV Huggingface Link. [2 CLINENV GitHub Link.](https://github.com/example) Table 2: Clinical information agents and their readviews.",
                page_number=2,
            ),
            TextLine(text="| Agent | Readview | Role |", page_number=2),
            TextLine(text="| --- | --- | --- |", page_number=2),
            TextLine(text="| Patient | Demographics | Simulates patient reporting |", page_number=2),
        ],
    )

    assert any(block.block_type == "table_caption" and block.content.startswith("Table 2:") for block in blocks)
    assert not any("CLINENV Huggingface Link" in block.content for block in blocks if block.block_type == "paragraph")
