from __future__ import annotations

import re
from dataclasses import replace

from ragarena.papers.models import PaperBlock, PaperFile
from ragarena.papers.metadata_noise import is_metadata_noise
from ragarena.papers.structured_parser import build_paper_block, classify_block
from ragarena.utils.text import sanitize_text


def parse_pdf_with_docling(paper_file: PaperFile) -> list[PaperBlock]:
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
            )
        }
    )

    result = converter.convert(str(paper_file.file_path))
    document = result.document
    markdown = sanitize_text(document.export_to_markdown())
    return blocks_from_docling_markdown(paper_file, markdown)


def blocks_from_docling_markdown(paper_file: PaperFile, markdown: str) -> list[PaperBlock]:
    blocks: list[PaperBlock] = []
    current_section: str | None = None
    paragraph_lines: list[str] = []
    table_lines: list[str] = []
    in_code = False
    code_lines: list[str] = []
    order_index = 0

    def add_block(
        block_type: str,
        content: str,
        *,
        markdown_content: str | None = None,
        section_name: str | None = None,
        image_path: str | None = None,
    ) -> None:
        nonlocal order_index, current_section
        clean_content = sanitize_text(content).strip()
        if block_type != "figure" and not clean_content:
            return
        block = build_paper_block(
            paper_file=paper_file,
            block_type=block_type,
            section_name=section_name if section_name is not None else current_section,
            page_number=None,
            content=clean_content,
            markdown_content=markdown_content,
            image_path=image_path,
            order_index=order_index,
            metadata={"source": "docling"},
        )
        blocks.append(block)
        order_index += 1
        if block_type in {"abstract", "section_title"}:
            current_section = clean_content[:160]

    def flush_paragraph() -> None:
        nonlocal paragraph_lines
        if paragraph_lines:
            content = " ".join(line.strip() for line in paragraph_lines).strip()
            block_type = classify_docling_text_block(content, order_index=order_index)
            add_block(block_type, content)
            paragraph_lines = []

    def flush_table() -> None:
        nonlocal table_lines
        if table_lines:
            markdown_table = "\n".join(table_lines)
            add_block("table", markdown_table, markdown_content=markdown_table)
            table_lines = []

    for raw_line in markdown.splitlines():
        line = sanitize_text(raw_line).rstrip()
        stripped = line.strip()

        if stripped.startswith("```"):
            flush_paragraph()
            flush_table()
            if in_code:
                add_block("code", "\n".join(code_lines), markdown_content="\n".join(code_lines))
                code_lines = []
                in_code = False
            else:
                in_code = True
            continue

        if in_code:
            code_lines.append(line)
            continue

        if is_markdown_table_line(stripped):
            flush_paragraph()
            table_lines.append(line)
            continue
        flush_table()

        image_match = re.match(r"!\[(?P<caption>[^\]]*)\]\((?P<path>[^)]+)\)", stripped)
        if image_match:
            flush_paragraph()
            caption = image_match.group("caption").strip()
            image_path = image_match.group("path").strip()
            add_block("figure", "", image_path=image_path)
            if caption:
                add_block("figure_caption", caption)
            continue

        if not stripped:
            flush_paragraph()
            continue

        heading_match = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if heading_match:
            flush_paragraph()
            heading_text = heading_match.group(2).strip()
            if order_index == 0:
                add_block("title", heading_text, markdown_content=stripped)
            else:
                block_type = "reference" if heading_text.lower() == "references" else "section_title"
                add_block(block_type, heading_text, markdown_content=stripped)
            continue

        block_type = classify_docling_text_block(stripped, order_index=order_index)
        if block_type in {"figure_caption", "table_caption", "algorithm", "formula", "reference"}:
            flush_paragraph()
            add_block(block_type, stripped)
            continue

        paragraph_lines.append(stripped)

    flush_paragraph()
    flush_table()
    if code_lines:
        add_block("code", "\n".join(code_lines), markdown_content="\n".join(code_lines))
    return normalize_docling_blocks(paper_file, blocks)


def normalize_docling_blocks(paper_file: PaperFile, blocks: list[PaperBlock]) -> list[PaperBlock]:
    normalized: list[PaperBlock] = []
    pending_paragraphs: list[PaperBlock] = []
    numeric_lines: list[str] = []
    title_seen = False
    title_index: int | None = None

    def flush_paragraphs() -> None:
        nonlocal pending_paragraphs
        if not pending_paragraphs:
            return
        content = " ".join(block.content.strip() for block in pending_paragraphs).strip()
        if content:
            normalized.append(rebuild_block(paper_file, pending_paragraphs[0], "paragraph", content, len(normalized)))
        pending_paragraphs = []

    def flush_numeric_lines() -> None:
        nonlocal numeric_lines
        if not numeric_lines:
            return
        content = "\n".join(numeric_lines)
        block_type = normalize_block_type("table_like", content)
        normalized.append(
            build_paper_block(
                paper_file=paper_file,
                block_type=block_type,
                section_name=normalized[-1].section_name if normalized else None,
                page_number=None,
                content=content,
                markdown_content=content if block_type == "table" else None,
                image_path=None,
                order_index=len(normalized),
                metadata={"source": "docling", "normalized": True},
            )
        )
        numeric_lines = []

    for index, block in enumerate(blocks):
        block_type = normalize_block_type(block.block_type, block.markdown_content or block.content)
        content = block.content.strip()
        if not content and block_type != "image_reference":
            continue
        if block_type not in {"table", "title", "authors", "affiliation"} and is_metadata_noise(
            content,
            block.section_name,
            block.order_index,
        ):
            flush_paragraphs()
            flush_numeric_lines()
            continue

        if not title_seen and (
            block_type == "title"
            or (block_type in {"section_title", "paragraph"} and looks_like_title(content))
        ):
            flush_paragraphs()
            flush_numeric_lines()
            normalized.append(rebuild_block(paper_file, block, "title", content, len(normalized)))
            title_seen = True
            title_index = index
            continue

        if title_index is not None and index <= title_index + 4 and is_author_or_affiliation_line(content):
            flush_paragraphs()
            flush_numeric_lines()
            metadata_type = "affiliation" if is_affiliation_line(content) else "authors"
            normalized.append(rebuild_block(paper_file, block, metadata_type, content, len(normalized)))
            continue

        if block_type == "table":
            flush_paragraphs()
            flush_numeric_lines()
            normalized.append(rebuild_block(paper_file, block, "table", content, len(normalized)))
            continue

        if is_numeric_heavy_line(content):
            flush_paragraphs()
            numeric_lines.append(content)
            continue

        if block_type in {"paragraph", "content_block"}:
            flush_numeric_lines()
            pending_paragraphs.append(block)
            if sum(len(item.content) for item in pending_paragraphs) >= 900:
                flush_paragraphs()
            continue

        flush_paragraphs()
        flush_numeric_lines()
        normalized.append(rebuild_block(paper_file, block, block_type, content, len(normalized)))

    flush_paragraphs()
    flush_numeric_lines()
    return assign_section_names(paper_file, normalized)


def assign_section_names(paper_file: PaperFile, blocks: list[PaperBlock]) -> list[PaperBlock]:
    assigned: list[PaperBlock] = []
    current_section: str | None = None
    for index, block in enumerate(blocks):
        if block.block_type in {"title", "authors", "affiliation", "date", "metadata"}:
            assigned.append(
                rebuild_block(
                    paper_file,
                    block,
                    block.block_type,
                    block.content,
                    index,
                    section_name=None,
                    use_block_section_name=False,
                )
            )
            continue
        if block.block_type == "section_title":
            current_section = normalize_heading(block.content)
            assigned.append(
                rebuild_block(
                    paper_file,
                    block,
                    block.block_type,
                    block.content,
                    index,
                    section_name=current_section,
                )
            )
            continue
        assigned.append(
            rebuild_block(
                paper_file,
                block,
                block.block_type,
                block.content,
                index,
                section_name=current_section,
            )
        )
    return assigned


def normalize_heading(content: str) -> str:
    text = sanitize_text(content).strip()
    text = re.sub(r"^#{1,6}\s+", "", text).strip()
    return re.sub(r"\s+", " ", text)


def normalize_block_type(raw_type: str, content: str) -> str:
    text = sanitize_text(content).strip()
    if not text:
        return "image_reference" if raw_type in {"figure", "image_reference"} else "unknown"
    if is_image_reference_text(text):
        return "image_reference"
    if is_markdown_table(text):
        return "table"
    if raw_type == "title":
        return "title"
    if is_clear_section_heading(raw_type, text):
        return "section_title"
    if raw_type in {"title", "authors", "affiliation", "date", "page_number", "reference", "metadata"}:
        return raw_type
    return "paragraph"


def rebuild_block(
    paper_file: PaperFile,
    block: PaperBlock,
    block_type: str,
    content: str,
    order_index: int,
    section_name: str | None = None,
    use_block_section_name: bool = True,
) -> PaperBlock:
    markdown_content = content if block_type == "table" else block.markdown_content
    resolved_section_name = block.section_name if use_block_section_name and section_name is None else section_name
    rebuilt = build_paper_block(
        paper_file=paper_file,
        block_type=block_type,
        section_name=resolved_section_name,
        page_number=block.page_number,
        content=content,
        markdown_content=markdown_content,
        image_path=block.image_path,
        order_index=order_index,
        metadata={**block.metadata, "normalized": True},
    )
    if block_type in {"title", "authors", "affiliation", "date", "page_number", "metadata"}:
        return replace(rebuilt, should_embed=False)
    return rebuilt


def classify_docling_text_block(text: str, *, order_index: int) -> str:
    if is_ambiguous_bracket_fragment(text):
        return "paragraph"
    return classify_block(text, order_index=order_index)


def is_markdown_table_line(line: str) -> bool:
    return line.startswith("|") and line.endswith("|") and line.count("|") >= 2


def is_markdown_table(text: str) -> bool:
    lines = [line.strip() for line in sanitize_text(text).splitlines() if line.strip()]
    return any("|" in line for line in lines) and any("---" in line for line in lines)


def is_image_reference_text(text: str) -> bool:
    stripped = sanitize_text(text).strip().lower()
    return stripped.startswith("<!-- image") and stripped.endswith("-->")


def is_clear_section_heading(raw_type: str, text: str) -> bool:
    first_line = next((line.strip() for line in sanitize_text(text).splitlines() if line.strip()), "")
    if re.match(r"^#{1,6}\s+\S+", first_line):
        return True
    if raw_type == "section_title":
        return True
    return bool(re.match(r"^\d+(?:\.\d+)*\s+[A-Z][^\n.]{1,160}$", first_line))


def is_ambiguous_bracket_fragment(text: str) -> bool:
    stripped = text.strip()
    return stripped in {"[]", "[...]", "...", "[ ]"} or bool(re.fullmatch(r"\[[.\s]*\]", stripped))


def looks_like_title(text: str) -> bool:
    stripped = text.strip()
    if is_ambiguous_bracket_fragment(stripped) or is_numeric_heavy_line(stripped):
        return False
    if len(stripped) < 12 or len(stripped) > 220:
        return False
    return bool(re.search(r"[A-Za-z\u4e00-\u9fff]", stripped))


def is_author_or_affiliation_line(text: str) -> bool:
    return is_author_line(text) or is_affiliation_line(text)


def is_author_line(text: str) -> bool:
    if "@" in text:
        return True
    if re.search(r"\b(and|,)\b", text, re.IGNORECASE) and re.search(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b", text):
        return True
    return bool(re.search(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+[0-9¹²³⁴⁵]*\b", text)) and len(text.split()) <= 18


def is_affiliation_line(text: str) -> bool:
    return bool(
        re.search(
            r"\b(Department of|School of|University|Institute|College|Laboratory|Lab|Faculty|Email|@)\b",
            text,
            re.IGNORECASE,
        )
    )


def is_numeric_heavy_line(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < 8:
        return False
    digit_count = sum(char.isdigit() for char in stripped)
    token_count = max(1, len(stripped.split()))
    numeric_tokens = len(re.findall(r"\b\d+(?:\.\d+)?%?\b", stripped))
    separators = stripped.count("|") + stripped.count("\t") + len(re.findall(r"\s{2,}", stripped))
    return numeric_tokens >= 3 or (digit_count / max(len(stripped), 1) > 0.35 and token_count >= 3) or separators >= 2
