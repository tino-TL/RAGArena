from __future__ import annotations

from pathlib import Path

from ragarena.papers.docling_parser import blocks_from_docling_markdown, normalize_block_type
from ragarena.papers.models import PaperFile


def paper_file() -> PaperFile:
    return PaperFile(
        paper_id=1,
        arxiv_id="2401.00001v1",
        pdf_url="https://arxiv.org/pdf/2401.00001v1",
        file_path=Path("paper.pdf"),
        file_sha256="sha",
        file_size=123,
    )


def test_docling_backend_mock_outputs_paper_blocks() -> None:
    blocks = blocks_from_docling_markdown(
        paper_file(),
        "# Paper Title\n\n## Abstract\n\nThis is an abstract.\n\n## Introduction\n\nThis is a paragraph.",
    )

    assert [block.block_type for block in blocks] == [
        "title",
        "section_title",
        "paragraph",
        "section_title",
        "paragraph",
    ]


def test_docling_parser_does_not_treat_bracket_fragment_as_code() -> None:
    blocks = blocks_from_docling_markdown(paper_file(), "# Paper Title\n\n[...]")

    assert blocks[-1].block_type == "paragraph"


def test_docling_table_converts_to_markdown_content() -> None:
    blocks = blocks_from_docling_markdown(
        paper_file(),
        "| Method | Score |\n|---|---:|\n| RAG | 0.9 |",
    )

    assert blocks[0].block_type == "table"
    assert "| Method | Score |" in (blocks[0].markdown_content or "")
    assert blocks[0].should_embed is True


def test_docling_reference_does_not_embed() -> None:
    blocks = blocks_from_docling_markdown(
        paper_file(),
        "## References\n\n[1] Smith et al. Retrieval systems.",
    )

    reference_blocks = [block for block in blocks if block.block_type == "reference"]
    assert reference_blocks
    assert all(block.should_embed is False for block in reference_blocks)


def test_author_line_after_title_becomes_metadata_not_paragraph() -> None:
    blocks = blocks_from_docling_markdown(
        paper_file(),
        "# Sector Rotation by Factor Model and Fundamental Analysis\n\n"
        "Runjia Yang1 and Beining Shi2\n\n"
        "1 Department of Finance, Example University\n\n"
        "## Introduction\n\n"
        "This paper studies sector rotation.",
    )

    assert blocks[0].block_type == "title"
    assert blocks[1].block_type == "authors"
    assert blocks[1].should_embed is False
    assert blocks[2].block_type == "affiliation"
    assert blocks[2].should_embed is False
    assert "paragraph" not in [block.block_type for block in blocks[:3]]


def test_prose_with_numbers_is_not_code() -> None:
    blocks = blocks_from_docling_markdown(
        paper_file(),
        "There are many different ways to divide sectors, including 10 sectors and 25 industries.",
    )

    assert blocks[0].block_type == "title"


def test_specific_prose_sentence_is_paragraph_after_title() -> None:
    blocks = blocks_from_docling_markdown(
        paper_file(),
        "# Paper\n\nThere are many different ways to divide sectors in the stock market.",
    )

    assert blocks[-1].block_type == "paragraph"


def test_numeric_heavy_repeated_lines_do_not_become_table_like() -> None:
    blocks = blocks_from_docling_markdown(
        paper_file(),
        "# Paper\n\n"
        "2019 2020 2021 0.12 0.14\n"
        "Energy 10 12 15 0.33\n"
        "Finance 20 22 28 0.41\n",
    )

    assert not any(block.block_type == "table_like" for block in blocks)
    paragraph = next(block for block in blocks if "Energy 10" in block.content)
    assert paragraph.block_type == "paragraph"
    assert paragraph.should_embed is True


def test_title_remains_title() -> None:
    blocks = blocks_from_docling_markdown(paper_file(), "# Strong Paper Title\n\nAuthors Name")

    assert blocks[0].block_type == "title"


def test_docling_section_title_section_names_follow_heading_content() -> None:
    blocks = blocks_from_docling_markdown(
        paper_file(),
        "# Paper Title\n\n"
        "## Abstract\n\n"
        "Abstract text.\n\n"
        "## 1 Introduction\n\n"
        "Introduction paragraph.\n\n"
        "## 2 Sector Classification and Return Analysis\n\n"
        "Sector paragraph.\n\n"
        "## 4 Fundamental Analysis\n\n"
        "Fundamental paragraph.\n\n"
        "## 4.1 PE Ratio\n\n"
        "PE paragraph.",
    )

    section_titles = [block for block in blocks if block.block_type == "section_title"]
    assert [(block.content, block.section_name) for block in section_titles] == [
        ("Abstract", "Abstract"),
        ("1 Introduction", "1 Introduction"),
        ("2 Sector Classification and Return Analysis", "2 Sector Classification and Return Analysis"),
        ("4 Fundamental Analysis", "4 Fundamental Analysis"),
        ("4.1 PE Ratio", "4.1 PE Ratio"),
    ]

    introduction_paragraph = next(
        block for block in blocks if block.block_type == "paragraph" and "Introduction paragraph" in block.content
    )
    sector_paragraph = next(
        block for block in blocks if block.block_type == "paragraph" and "Sector paragraph" in block.content
    )
    fundamental_paragraph = next(
        block for block in blocks if block.block_type == "paragraph" and "Fundamental paragraph" in block.content
    )
    pe_paragraph = next(
        block for block in blocks if block.block_type == "paragraph" and "PE paragraph" in block.content
    )

    assert introduction_paragraph.section_name == "1 Introduction"
    assert sector_paragraph.section_name == "2 Sector Classification and Return Analysis"
    assert fundamental_paragraph.section_name == "4 Fundamental Analysis"
    assert pe_paragraph.section_name == "4.1 PE Ratio"


def test_docling_metadata_blocks_do_not_enter_body_section() -> None:
    blocks = blocks_from_docling_markdown(
        paper_file(),
        "# Paper Title\n\n"
        "Jane Doe1\n\n"
        "1 Department of Finance, Example University\n\n"
        "## 1 Introduction\n\n"
        "Introduction paragraph.",
    )

    metadata_blocks = [block for block in blocks if block.block_type in {"title", "authors", "affiliation"}]
    assert metadata_blocks
    assert all(block.section_name is None for block in metadata_blocks)
    assert all(block.should_embed is False for block in metadata_blocks)


def test_docling_standalone_page_number_noise_is_removed() -> None:
    blocks = blocks_from_docling_markdown(
        paper_file(),
        "# Paper Title\n\n"
        "1\n\n"
        "## Abstract\n\n"
        "This abstract reports a 21.19% improvement.",
    )

    assert not any(block.content == "1" for block in blocks)
    assert any("21.19%" in block.content for block in blocks)


def test_docling_standalone_date_noise_is_removed() -> None:
    blocks = blocks_from_docling_markdown(
        paper_file(),
        "# Paper Title\n\n"
        "Sept 2023\n\n"
        "## Abstract\n\n"
        "This abstract remains.",
    )

    assert not any(block.content == "Sept 2023" for block in blocks)
    assert any(block.section_name == "Abstract" and "abstract remains" in block.content for block in blocks)


def test_docling_normalize_block_type_does_not_trust_code_for_prose() -> None:
    assert (
        normalize_block_type(
            "code",
            "From this table, we can tell that valuation ratios vary across sectors.",
        )
        == "paragraph"
    )


def test_docling_normalize_block_type_does_not_trust_table_like_for_prose() -> None:
    assert (
        normalize_block_type(
            "table_like",
            "The P/B ratio provides a useful valuation signal for financial analysis.",
        )
        == "paragraph"
    )


def test_docling_normalize_block_type_keeps_real_markdown_table() -> None:
    assert normalize_block_type("table_like", "| A | B |\n|---|---|\n| 1 | 2 |") == "table"


def test_docling_normalize_block_type_image_reference_not_embedded() -> None:
    blocks = blocks_from_docling_markdown(paper_file(), "<!-- image -->")

    assert blocks[0].block_type == "image_reference"
    assert blocks[0].should_embed is False
