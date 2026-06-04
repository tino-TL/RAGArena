from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ragarena.ingestion.hashing import content_hash
from ragarena.papers.metadata_noise import is_metadata_noise
from ragarena.papers.models import (
    PaperBlock,
    PaperFile,
    normalize_block_type,
    should_embed_block,
)
from ragarena.utils.text import sanitize_text

SECTION_TITLES = {
    "abstract",
    "introduction",
    "related work",
    "background",
    "method",
    "methods",
    "methodology",
    "approach",
    "experiments",
    "experimental setup",
    "results",
    "discussion",
    "conclusion",
    "conclusions",
    "references",
    "appendix",
}

FIGURE_CAPTION_RE = re.compile(r"^(figure|fig\.)\s*\d+[:.\s]", re.IGNORECASE)
TABLE_CAPTION_RE = re.compile(r"^table\s*\d+[:.\s]", re.IGNORECASE)
ALGORITHM_RE = re.compile(r"^(algorithm\s*\d+|procedure|pseudo\s*code|pseudocode)\b", re.IGNORECASE)
REFERENCE_RE = re.compile(r"^\[\d+\]|\bet al\.\b|^references$", re.IGNORECASE)
TABLE_RE = re.compile(r"(^\s*\|.+\|\s*$|^\s*[-+]{3,}\s*$)", re.MULTILINE)
FORMULA_RE = re.compile(
    r"(\\[a-zA-Z]+|[\u03b1\u03b2\u03b3\u03bb\u03bc\u03c3\u2211\u222b\u221a"
    r"\u2248\u2264\u2265\u2260\u221e]|\b[a-zA-Z]\s*=\s*|[\u2211\u222b]\s*)"
)
CODE_RE = re.compile(
    r"(^\s*(def|class|import|from|for|while|if|elif|else|try|except|return)\b"
    r"|^\s*\{.+:.+\}\s*$"
    r"|^\s*\[[^\]]+:.+\]\s*$"
    r"|^\s*[-\w]+:\s*[^,]+$"
    r"|^\s*(SELECT|INSERT|UPDATE|DELETE|CREATE|WITH)\b"
    r"|^\s*(curl|uv|python|pip|docker|git|npm)\b)",
    re.IGNORECASE | re.MULTILINE,
)


class TextLine:
    def __init__(self, *, text: str, page_number: int) -> None:
        self.text = text
        self.page_number = page_number


def parse_pdf_to_blocks(
    paper_file: PaperFile,
    *,
    image_dir: Path | None = None,
) -> list[PaperBlock]:
    import fitz

    image_root = image_dir or paper_file.file_path.parent / "images" / paper_file.arxiv_id
    lines: list[TextLine] = []
    image_paths_by_page: dict[int, list[Path]] = {}

    with fitz.open(str(paper_file.file_path)) as doc:
        for page_index, page in enumerate(doc, start=1):
            for raw_block in sort_page_blocks_for_reading(page, page.get_text("blocks")):
                text = sanitize_text(str(raw_block[4])).strip("\n")
                raw_type = classify_block(text, order_index=len(lines))
                if raw_type in {"code", "algorithm", "table"} and not is_noise_line(text):
                    lines.append(TextLine(text=text, page_number=page_index))
                    continue
                for line in text.splitlines():
                    clean_line = clean_pdf_line(line)
                    if clean_line:
                        lines.append(TextLine(text=clean_line, page_number=page_index))
            image_paths_by_page[page_index] = extract_page_images(doc, page, image_root, page_index)

    blocks = coalesce_text_lines(paper_file, lines)
    order_index = len(blocks)
    current_section = blocks[-1].section_name if blocks else None
    image_block_count = 0
    for page_number, image_paths in image_paths_by_page.items():
        for image_path in image_paths:
            if image_block_count >= 30:
                break
            blocks.append(
                build_paper_block(
                    paper_file=paper_file,
                    block_type="figure",
                    section_name=current_section,
                    page_number=page_number,
                    content="",
                    markdown_content=None,
                    image_path=str(image_path),
                    order_index=order_index,
                    metadata={"source": "pymupdf"},
                )
            )
            order_index += 1
            image_block_count += 1
    return blocks


def sort_page_blocks_for_reading(page: Any, raw_blocks: list[tuple[Any, ...]]) -> list[tuple[Any, ...]]:
    text_blocks = [block for block in raw_blocks if sanitize_text(str(block[4])).strip()]
    if not text_blocks:
        return []

    page_width = float(page.rect.width)
    mid_x = page_width / 2
    left_blocks = [block for block in text_blocks if float(block[0]) < mid_x and float(block[2]) <= mid_x + 48]
    right_blocks = [block for block in text_blocks if float(block[0]) >= mid_x - 48]
    two_column = len(left_blocks) >= 2 and len(right_blocks) >= 1
    if not two_column:
        return repair_sentence_continuations(sorted(text_blocks, key=lambda item: (float(item[1]), float(item[0]))))

    def column_key(block: tuple[Any, ...]) -> tuple[int, float, float]:
        column = 0 if float(block[0]) < mid_x else 1
        return (column, float(block[1]), float(block[0]))

    return repair_sentence_continuations(sorted(text_blocks, key=column_key))


def repair_sentence_continuations(blocks: list[tuple[Any, ...]]) -> list[tuple[Any, ...]]:
    items = list(blocks)
    index = 0
    while index < len(items) - 1:
        current_text = sanitize_text(str(items[index][4])).strip()
        if not looks_like_unfinished_text(current_text):
            index += 1
            continue
        continuation_index = find_continuation_block(items, index)
        if continuation_index is None:
            index += 1
            continue
        continuation = items.pop(continuation_index)
        items.insert(index + 1, continuation)
        index += 2
    return items


def find_continuation_block(blocks: list[tuple[Any, ...]], index: int) -> int | None:
    current = blocks[index]
    current_x = float(current[0])
    for candidate_index in range(index + 1, min(index + 5, len(blocks))):
        candidate = blocks[candidate_index]
        if abs(float(candidate[0]) - current_x) > 80:
            continue
        candidate_text = sanitize_text(str(candidate[4])).strip()
        if looks_like_continuation_start(candidate_text):
            return candidate_index
    return None


def looks_like_unfinished_text(text: str) -> bool:
    stripped = sanitize_text(text).strip()
    if not stripped:
        return False
    return not stripped.endswith((".", "!", "?", ":", ";", ")"))


def looks_like_continuation_start(text: str) -> bool:
    stripped = sanitize_text(text).strip()
    if not stripped:
        return False
    return bool(re.match(r"^(?:[a-z][a-z-]*|loop|where|because|including|such as)\b", stripped))


def coalesce_text_lines(paper_file: PaperFile, lines: list[TextLine]) -> list[PaperBlock]:
    blocks: list[PaperBlock] = []
    paragraph_lines: list[TextLine] = []
    reference_lines: list[TextLine] = []
    current_section: str | None = None
    seen_title: str | None = None

    def next_order_index() -> int:
        return len(blocks)

    def flush_paragraph() -> None:
        nonlocal paragraph_lines
        if not paragraph_lines:
            return
        content = merge_lines(paragraph_lines)
        if has_semantic_content(content):
            blocks.append(
                build_paper_block(
                    paper_file=paper_file,
                    block_type="paragraph",
                    section_name=current_section,
                    page_number=paragraph_lines[0].page_number,
                    content=content,
                    markdown_content=content,
                    image_path=None,
                    order_index=next_order_index(),
                    metadata={"source": "pymupdf", "coalesced_lines": len(paragraph_lines)},
                )
            )
        paragraph_lines = []

    def flush_references() -> None:
        nonlocal reference_lines
        if not reference_lines:
            return
        content = merge_lines(reference_lines)
        if has_semantic_content(content):
            blocks.append(
                build_paper_block(
                    paper_file=paper_file,
                    block_type="reference",
                    section_name="References",
                    page_number=reference_lines[0].page_number,
                    content=content,
                    markdown_content=content,
                    image_path=None,
                    order_index=next_order_index(),
                    metadata={"source": "pymupdf", "coalesced_lines": len(reference_lines)},
                )
            )
        reference_lines = []

    for line in lines:
        text = line.text
        if is_noise_line(text):
            continue
        embedded_visual = split_metadata_prefixed_visual_caption(text)
        if embedded_visual is not None:
            prefix, caption, visual_type = embedded_visual
            if prefix and has_semantic_content(prefix) and not is_metadata_noise(prefix) and not looks_like_metadata_prefix(prefix):
                paragraph_lines.append(TextLine(text=prefix, page_number=line.page_number))
                flush_paragraph()
            else:
                flush_paragraph()
            flush_references()
            blocks.append(
                build_paper_block(
                    paper_file=paper_file,
                    block_type=visual_type,
                    section_name=current_section,
                    page_number=line.page_number,
                    content=caption,
                    markdown_content=to_markdown(visual_type, caption),
                    image_path=None,
                    order_index=next_order_index(),
                    metadata={"source": "pymupdf", "split_embedded_caption": True},
                )
            )
            continue
        line_type = classify_line(text, order_index=next_order_index())
        if line_type in {"authors", "affiliation", "metadata"}:
            flush_paragraph()
            continue
        if seen_title and normalize_text_for_dedupe(text) == seen_title:
            continue

        if current_section == "References" and line_type not in {"section_title", "appendix"}:
            reference_lines.append(line)
            if len(reference_lines) >= 80 or sum(len(item.text) for item in reference_lines) >= 10000:
                flush_references()
            continue

        if line_type in {
            "title",
            "abstract",
            "section_title",
            "figure_caption",
            "table_caption",
            "code",
            "algorithm",
            "table",
        }:
            flush_paragraph()
            flush_references()
            if line_type == "title":
                seen_title = normalize_text_for_dedupe(text)
            if line_type in {"abstract", "section_title"}:
                current_section = text[:160]
            blocks.append(
                build_paper_block(
                    paper_file=paper_file,
                    block_type=line_type,
                    section_name=current_section,
                    page_number=line.page_number,
                    content=text,
                    markdown_content=to_markdown(line_type, text),
                    image_path=None,
                    order_index=next_order_index(),
                    metadata={"source": "pymupdf"},
                )
            )
            continue

        if line_type == "reference":
            flush_paragraph()
            current_section = "References"
            reference_lines.append(line)
            continue

        paragraph_lines.append(line)
        if line_type != "formula" and should_flush_paragraph(paragraph_lines):
            flush_paragraph()

    flush_paragraph()
    flush_references()
    return blocks


def build_paper_block(
    *,
    paper_file: PaperFile,
    block_type: str,
    section_name: str | None,
    page_number: int | None,
    content: str,
    markdown_content: str | None,
    image_path: str | None,
    order_index: int,
    metadata: dict[str, Any] | None = None,
) -> PaperBlock:
    normalized_type = normalize_block_type(block_type)
    clean_content = sanitize_text(content)
    clean_section = sanitize_text(section_name) if section_name else None
    clean_markdown = sanitize_text(markdown_content) if markdown_content else None
    clean_image_path = sanitize_text(image_path) if image_path else None
    hash_input = "\n".join(
        [
            paper_file.arxiv_id,
            normalized_type,
            str(page_number or ""),
            str(order_index),
            clean_section or "",
            clean_content,
            clean_image_path or "",
        ]
    )
    return PaperBlock(
        id=None,
        paper_id=paper_file.paper_id,
        arxiv_id=paper_file.arxiv_id,
        block_type=normalized_type,
        section_name=clean_section,
        page_number=page_number,
        content=clean_content,
        markdown_content=clean_markdown,
        image_path=clean_image_path,
        order_index=order_index,
        should_embed=should_embed_block(normalized_type),
        metadata=metadata or {},
        content_hash=content_hash(hash_input),
    )


def classify_block(text: str, *, order_index: int = 0) -> str:
    clean = sanitize_text(text).strip()
    first_line = clean.splitlines()[0].strip() if clean else ""
    normalized = re.sub(r"^\d+(\.\d+)*\s+", "", first_line).strip().lower()

    if not clean:
        return "unknown"
    if order_index == 0 and len(first_line) <= 220:
        return "title"
    if order_index <= 6 and is_author_or_affiliation_line(first_line):
        return "affiliation" if is_affiliation_line(first_line) else "authors"
    if normalized == "abstract" or first_line.lower().startswith("abstract"):
        return "abstract"
    if normalized in SECTION_TITLES:
        if normalized == "references":
            return "reference"
        if normalized == "appendix":
            return "appendix"
        return "section_title"
    if FIGURE_CAPTION_RE.search(first_line):
        return "figure_caption"
    if TABLE_CAPTION_RE.search(first_line):
        return "table_caption"
    if is_table_block(clean):
        return "table"
    if ALGORITHM_RE.search(first_line):
        return "algorithm"
    if is_code_block(clean):
        return "code"
    if is_formula_block(clean):
        return "formula"
    if REFERENCE_RE.search(first_line):
        return "reference"
    if first_line.startswith(("-", "*", "•")):
        return "list"
    return "paragraph"


def classify_line(text: str, *, order_index: int = 0) -> str:
    return classify_block(text, order_index=order_index)


def is_code_block(text: str) -> bool:
    if is_ambiguous_bracket_fragment(text):
        return False
    lines = text.splitlines()
    if len(lines) >= 2 and any(line.startswith(("    ", "\t")) for line in lines[1:]):
        return True
    return bool(CODE_RE.search(text))


def is_ambiguous_bracket_fragment(text: str) -> bool:
    stripped = text.strip()
    return stripped in {"[]", "[...]", "...", "[ ]"} or bool(re.fullmatch(r"\[[.\s]*\]", stripped))


def is_formula_block(text: str) -> bool:
    if len(text) > 500:
        return False
    return bool(FORMULA_RE.search(text)) and (
        "=" in text or "\\" in text or any(symbol in text for symbol in "αβγλ∑∫")
    )


def is_table_block(text: str) -> bool:
    return bool(TABLE_RE.search(text)) or text.count("\t") >= 4


def to_markdown(block_type: str, text: str) -> str:
    if block_type == "section_title":
        return f"## {text.strip()}"
    if block_type == "title":
        return f"# {text.strip()}"
    if block_type in {"code", "algorithm"}:
        return f"```\n{text.rstrip()}\n```"
    return text


def clean_pdf_line(value: str) -> str:
    return re.sub(r"[ \t]+", " ", sanitize_text(value)).strip()


def merge_lines(lines: list[TextLine]) -> str:
    parts: list[str] = []
    for line in lines:
        if parts and should_join_without_space(parts[-1], line.text):
            parts[-1] = parts[-1].rstrip("-") + line.text
        else:
            parts.append(line.text)
    return sanitize_text(" ".join(parts)).strip()


def should_join_without_space(previous: str, current: str) -> bool:
    return previous.endswith("-") and current[:1].islower()


def should_flush_paragraph(lines: list[TextLine]) -> bool:
    content_length = sum(len(line.text) for line in lines)
    return content_length >= 1200 or len(lines) >= 12


def has_semantic_content(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < 20:
        return False
    return bool(re.search(r"[A-Za-z\u4e00-\u9fff]", stripped))


def is_noise_line(text: str) -> bool:
    stripped = text.strip()
    lowered = stripped.lower()
    if is_metadata_noise(stripped):
        return True
    if not stripped:
        return True
    if re.fullmatch(r"\d+", stripped):
        return True
    if len(stripped) <= 2:
        return True
    if lowered.startswith(("arxiv:", "preprint", "submitted to", "under review")):
        return True
    if "all rights reserved" in lowered or "creative commons" in lowered:
        return True
    if "copyright" in lowered or "license" in lowered:
        return True
    if lowered in {"doi", "www", "http", "https"}:
        return True
    return False


def split_metadata_prefixed_visual_caption(text: str) -> tuple[str, str, str] | None:
    clean = sanitize_text(text).strip()
    if not clean or is_caption_start(clean):
        return None
    match = re.search(r"\b(?P<label>Table|Figure|Fig\.)\s*\d+[A-Za-z]?\s*[:.]\s*", clean, flags=re.IGNORECASE)
    if not match:
        return None
    prefix = clean[: match.start()].strip()
    caption = clean[match.start() :].strip()
    if not prefix or not caption:
        return None
    if not looks_like_metadata_prefix(prefix):
        return None
    visual_type = "table_caption" if match.group("label").lower().startswith("table") else "figure_caption"
    return prefix, caption, visual_type


def is_caption_start(text: str) -> bool:
    return bool(re.match(r"^(Table|Figure|Fig\.)\s*\d+[A-Za-z]?\s*[:.]", sanitize_text(text).strip(), flags=re.IGNORECASE))


def looks_like_metadata_prefix(text: str) -> bool:
    lowered = sanitize_text(text).lower()
    return (
        "http" in lowered
        or "github" in lowered
        or "huggingface" in lowered
        or "physionet" in lowered
        or "link" in lowered
        or bool(re.fullmatch(r"(?:\d+\s+[^.]+\.?\s*){1,4}", sanitize_text(text).strip()))
    )


def is_author_or_affiliation_line(text: str) -> bool:
    return is_author_line(text) or is_affiliation_line(text)


def is_author_line(text: str) -> bool:
    stripped = sanitize_text(text).strip()
    if not stripped or len(stripped) > 220:
        return False
    if re.match(r"^(Algorithm|Figure|Fig\.|Table)\s*\d+\b", stripped, re.IGNORECASE):
        return False
    if "@" in stripped:
        return True
    if re.search(r"\b(and|,)\b", stripped, re.IGNORECASE) and re.search(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b", stripped):
        return True
    return bool(re.search(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+[0-9*†‡§¶]*\b", stripped)) and len(stripped.split()) <= 18


def is_affiliation_line(text: str) -> bool:
    return bool(
        re.search(
            r"\b(Department of|School of|University|Institute|College|Laboratory|Lab|Faculty|Email|E-mail|@)\b",
            sanitize_text(text),
            re.IGNORECASE,
        )
    )


def normalize_text_for_dedupe(text: str) -> str:
    return re.sub(r"\W+", "", text).lower()


def extract_page_images(
    doc: Any,
    page: Any,
    image_root: Path,
    page_number: int,
) -> list[Path]:
    paths: list[Path] = []
    image_root.mkdir(parents=True, exist_ok=True)
    for image_index, image_info in enumerate(page.get_images(full=True), start=1):
        xref = image_info[0]
        image = doc.extract_image(xref)
        width = int(image.get("width", 0))
        height = int(image.get("height", 0))
        if width < 180 or height < 120 or (width * height) < 40000:
            continue
        extension = image.get("ext", "png")
        image_path = image_root / f"page-{page_number}-image-{image_index}.{extension}"
        image_path.write_bytes(image["image"])
        paths.append(image_path)
    return paths
