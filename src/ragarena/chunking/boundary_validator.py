from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Protocol

from ragarena.chunking.agentic_chunker import cleanup_retrieval_chunks
from ragarena.chunking.agentic_chunker import AGENTIC_CHUNK_TYPE
from ragarena.chunking.fixed_chunker import Chunk, estimate_token_count
from ragarena.chunking.section_names import (
    derive_section_name_from_content,
    extract_section_name_from_heading,
    first_non_empty_line,
    resolve_section_name,
)
from ragarena.ingestion.hashing import content_hash
from ragarena.papers.metadata_noise import (
    is_metadata_noise as is_shared_metadata_noise,
    is_standalone_boundary_noise_line,
)
from ragarena.papers.models import PaperBlock
from ragarena.utils.text import sanitize_text

TRAILING_SECTION_HEADING_RE = re.compile(r"^##\s+\d+(?:\.\d+)*\s+.+$")


@dataclass(frozen=True)
class BoundaryValidationStats:
    boundary_issues_found: int = 0
    rule_fixes: int = 0
    model_fixes: int = 0
    dropped_chunks: int = 0


@dataclass(frozen=True)
class BoundaryValidationResult:
    chunks: list[Chunk]
    stats: BoundaryValidationStats


class BoundaryPlanningModel(Protocol):
    def generate(self, prompt: str, system_prompt: str) -> object: ...


def validate_chunk_boundaries(
    chunks: list[Chunk],
    blocks: list[PaperBlock],
    *,
    model: BoundaryPlanningModel | None = None,
    use_model: bool = False,
) -> BoundaryValidationResult:
    block_by_id = {block.id: block for block in blocks if block.id is not None}
    rule_result = apply_rule_boundary_fixes(chunks, block_by_id)
    model_result = apply_model_boundary_fixes(
        rule_result.chunks,
        block_by_id,
        model=model,
        use_model=use_model,
    )
    stats = BoundaryValidationStats(
        boundary_issues_found=rule_result.stats.boundary_issues_found + model_result.stats.boundary_issues_found,
        rule_fixes=rule_result.stats.rule_fixes,
        model_fixes=model_result.stats.model_fixes,
        dropped_chunks=rule_result.stats.dropped_chunks + model_result.stats.dropped_chunks,
    )
    return BoundaryValidationResult(chunks=model_result.chunks, stats=stats)


def apply_rule_boundary_fixes(
    chunks: list[Chunk],
    block_by_id: dict[int, PaperBlock],
) -> BoundaryValidationResult:
    ordered = sorted(chunks, key=lambda chunk: (chunk.document_id, chunk.chunk_index))
    rebuilt: list[Chunk] = []
    pending_titles: dict[tuple[int, str], list[int]] = {}
    issues = 0
    fixes = 0
    dropped = 0

    for chunk in ordered:
        source_ids = [block_id for block_id in chunk.source_block_ids or [] if block_id in block_by_id]
        if not source_ids:
            if is_metadata_like_chunk(chunk):
                issues += 1
                fixes += 1
                dropped += 1
                continue
            rebuilt.append(chunk)
            continue

        original_paper_ids = {block_by_id[block_id].paper_id for block_id in source_ids}
        source_ids = same_paper_source_ids(source_ids, block_by_id)
        if len(original_paper_ids) > 1:
            issues += 1
            fixes += 1
        source_ids = sorted(source_ids, key=lambda block_id: block_by_id[block_id].order_index)
        original_ids = list(source_ids)
        source_ids = [
            block_id
            for block_id in source_ids
            if not is_metadata_like_content(
                block_by_id[block_id].markdown_content or block_by_id[block_id].content,
                block_by_id[block_id].section_name,
                block_by_id[block_id].order_index,
            )
        ]
        if len(source_ids) != len(original_ids):
            issues += 1
            fixes += 1

        source_ids, pending_updates = detach_backward_section_titles(source_ids, chunk, block_by_id)
        for key, title_ids in pending_updates.items():
            pending_titles.setdefault(key, []).extend(title_ids)
            issues += len(title_ids)
            fixes += len(title_ids)

        section_key = (chunk.document_id, normalize_section(chunk.section_name))
        if pending_titles.get(section_key):
            source_ids = pending_titles.pop(section_key) + source_ids
            fixes += 1

        if not source_ids:
            issues += 1
            fixes += 1
            dropped += 1
            continue

        rebuilt.append(rebuild_chunk_from_blocks(chunk, source_ids, block_by_id))

    for title_ids in pending_titles.values():
        if title_ids:
            issues += len(title_ids)
            fixes += len(title_ids)
            dropped += 1

    split_result = split_internal_section_heading_leakage(rebuilt)
    cleanup = cleanup_retrieval_chunks(split_result.chunks)
    heading_result = fix_trailing_section_heading_leakage(cleanup.chunks)
    trim_result = trim_boundary_noise_lines(heading_result.chunks)
    sentence_result = fix_bad_sentence_boundaries(trim_result.chunks)
    dedupe_result = dedupe_adjacent_chunks(sentence_result.chunks)
    cleaned = dedupe_result.chunks
    cleanup_issues = cleanup.dropped_tiny_chunks + cleanup.merged_tiny_chunks
    stats = BoundaryValidationStats(
        boundary_issues_found=issues
        + split_result.stats.boundary_issues_found
        + cleanup_issues
        + heading_result.stats.boundary_issues_found
        + trim_result.stats.boundary_issues_found
        + sentence_result.stats.boundary_issues_found
        + dedupe_result.stats.boundary_issues_found,
        rule_fixes=fixes
        + split_result.stats.rule_fixes
        + cleanup_issues
        + heading_result.stats.rule_fixes
        + trim_result.stats.rule_fixes
        + sentence_result.stats.rule_fixes
        + dedupe_result.stats.rule_fixes,
        dropped_chunks=dropped
        + split_result.stats.dropped_chunks
        + cleanup.dropped_tiny_chunks
        + heading_result.stats.dropped_chunks
        + trim_result.stats.dropped_chunks
        + sentence_result.stats.dropped_chunks
        + dedupe_result.stats.dropped_chunks,
    )
    return BoundaryValidationResult(chunks=cleaned, stats=stats)


def split_internal_section_heading_leakage(chunks: list[Chunk]) -> BoundaryValidationResult:
    output: list[Chunk] = []
    issues = 0
    fixes = 0
    for chunk in sorted(chunks, key=lambda item: (item.document_id, item.chunk_index)):
        pieces = split_chunk_on_internal_h2_headings(chunk)
        if len(pieces) > 1:
            issues += len(pieces) - 1
            fixes += len(pieces) - 1
        output.extend(pieces)
    return BoundaryValidationResult(
        chunks=output,
        stats=BoundaryValidationStats(boundary_issues_found=issues, rule_fixes=fixes),
    )


def split_chunk_on_internal_h2_headings(chunk: Chunk) -> list[Chunk]:
    lines = sanitize_text(chunk.content).splitlines()
    heading_indexes = [
        index
        for index, line in enumerate(lines)
        if index > 0
        and index < len(lines) - 1
        and TRAILING_SECTION_HEADING_RE.match(line.strip())
        and should_split_internal_heading(chunk.section_name, line.strip())
    ]
    if not heading_indexes:
        return [chunk]

    starts = [0, *heading_indexes]
    ends = [*heading_indexes, len(lines)]
    pieces: list[Chunk] = []
    for offset, (start, end) in enumerate(zip(starts, ends, strict=True)):
        content = sanitize_text("\n".join(lines[start:end])).strip()
        if not content:
            continue
        first_line = first_non_empty_line(content)
        section_name = (
            extract_section_name_from_heading(first_line)
            if first_line and TRAILING_SECTION_HEADING_RE.match(first_line)
            else chunk.section_name
        )
        pieces.append(
            rebuild_chunk_with_content(
                chunk,
                content,
                section_name=section_name,
                chunk_index=chunk.chunk_index + offset,
            )
        )
    return pieces or [chunk]


def should_split_internal_heading(current_section: str | None, heading: str) -> bool:
    current_number = section_number(current_section)
    heading_number = section_number(extract_section_name_from_heading(heading))
    if not current_number or not heading_number:
        return False
    return current_number != heading_number


def trim_boundary_noise_lines(chunks: list[Chunk]) -> BoundaryValidationResult:
    cleaned: list[Chunk] = []
    issues = 0
    fixes = 0
    dropped = 0
    for chunk in chunks:
        trimmed_content = remove_boundary_noise_lines(chunk.content)
        if trimmed_content == chunk.content.strip():
            cleaned.append(chunk)
            continue
        issues += 1
        fixes += 1
        if not trimmed_content:
            dropped += 1
            continue
        cleaned.append(rebuild_chunk_with_content(chunk, trimmed_content))
    return BoundaryValidationResult(
        chunks=cleaned,
        stats=BoundaryValidationStats(
            boundary_issues_found=issues,
            rule_fixes=fixes,
            dropped_chunks=dropped,
        ),
    )


def remove_boundary_noise_lines(content: str) -> str:
    lines = sanitize_text(content).strip().splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    while lines and is_standalone_boundary_noise_line(lines[0]):
        lines.pop(0)
        while lines and not lines[0].strip():
            lines.pop(0)
    while lines and is_standalone_boundary_noise_line(lines[-1]):
        lines.pop()
        while lines and not lines[-1].strip():
            lines.pop()
    return sanitize_text("\n".join(lines)).strip()


def fix_bad_sentence_boundaries(chunks: list[Chunk]) -> BoundaryValidationResult:
    ordered = sorted(chunks, key=lambda chunk: (chunk.document_id, chunk.chunk_index))
    result = list(ordered)
    consumed: set[int] = set()
    issues = 0
    fixes = 0
    for index, chunk in enumerate(ordered):
        if index in consumed or not has_bad_sentence_end(chunk.content) or is_visual_chunk(chunk):
            continue
        target_index = find_sentence_continuation_chunk(result, index, consumed)
        if target_index is None:
            continue
        result[index] = merge_boundary_chunks(result[index], result[target_index])
        consumed.add(target_index)
        issues += 1
        fixes += 1
    return BoundaryValidationResult(
        chunks=[chunk for index, chunk in enumerate(result) if index not in consumed],
        stats=BoundaryValidationStats(boundary_issues_found=issues, rule_fixes=fixes),
    )


def has_bad_sentence_end(content: str) -> bool:
    text = sanitize_text(content).rstrip()
    if not text:
        return False
    tail = last_meaningful_text(text)
    if re.search(r"(?i)\b(?:and|or|where|because|therefore|including|such as|with|by|for|to|in)\s*$", tail):
        return True
    if re.search(r"(?:,\s*|;\s*|:\s*)$", tail):
        return True
    if re.search(r"\b(?:and|or)\s*\(\d+\)\s*$", tail, flags=re.IGNORECASE):
        return True
    if unmatched_parentheses(tail):
        return True
    return False


def last_meaningful_text(content: str) -> str:
    lines = [line.strip() for line in sanitize_text(content).splitlines() if line.strip()]
    return lines[-1] if lines else ""


def unmatched_parentheses(text: str) -> bool:
    return text.count("(") > text.count(")")


def find_sentence_continuation_chunk(chunks: list[Chunk], index: int, consumed: set[int]) -> int | None:
    current = chunks[index]
    for candidate_index in range(index + 1, min(index + 4, len(chunks))):
        if candidate_index in consumed:
            continue
        candidate = chunks[candidate_index]
        if candidate.document_id != current.document_id:
            return None
        if is_visual_chunk(candidate):
            continue
        if normalize_section(candidate.section_name) != normalize_section(current.section_name):
            return None
        return candidate_index
    return None


def dedupe_adjacent_chunks(chunks: list[Chunk]) -> BoundaryValidationResult:
    ordered = sorted(chunks, key=lambda chunk: (chunk.document_id, chunk.chunk_index))
    output: list[Chunk] = []
    issues = 0
    dropped = 0
    for chunk in ordered:
        if output and should_drop_as_adjacent_duplicate(output[-1], chunk):
            issues += 1
            dropped += 1
            if should_prefer_duplicate_candidate(chunk, output[-1]):
                output[-1] = chunk
            continue
        output.append(chunk)
    return BoundaryValidationResult(
        chunks=output,
        stats=BoundaryValidationStats(boundary_issues_found=issues, rule_fixes=issues, dropped_chunks=dropped),
    )


def should_drop_as_adjacent_duplicate(left: Chunk, right: Chunk) -> bool:
    if left.document_id != right.document_id:
        return False
    if is_visual_chunk(left) or is_visual_chunk(right):
        return False
    if normalize_section(left.section_name) == normalize_section(right.section_name):
        return text_duplicate_score(left.content, right.content) >= 0.82 and has_long_text_containment(left.content, right.content)
    if not are_adjacent_sections(left.section_name, right.section_name):
        return False
    return text_duplicate_score(left.content, right.content) >= 0.82 and has_long_text_containment(left.content, right.content)


def should_prefer_duplicate_candidate(candidate: Chunk, current: Chunk) -> bool:
    candidate_score = section_title_match_score(candidate.section_name, candidate.content)
    current_score = section_title_match_score(current.section_name, current.content)
    if candidate_score != current_score:
        return candidate_score > current_score
    return candidate.token_count > current.token_count


def section_title_match_score(section: str | None, content: str) -> int:
    title = re.sub(r"^\s*\d+(?:\.\d+)*\s*", "", sanitize_text(section or "").lower())
    words = {word for word in re.findall(r"[a-z][a-z0-9_-]{2,}", title) if word not in {"the", "and", "for"}}
    if not words:
        return 0
    content_text = sanitize_text(content).lower()
    score = sum(1 for word in words if word in content_text)
    first_line = first_non_empty_line(content)
    if first_line and TRAILING_SECTION_HEADING_RE.match(first_line):
        heading = normalize_heading_text(first_line)
        if strip_leading_section_number(heading) == strip_leading_section_number(normalize_heading_text(section or "")):
            score += 10
    return score


def same_or_adjacent_section(left: str | None, right: str | None) -> bool:
    if normalize_section(left) == normalize_section(right):
        return True
    return are_adjacent_sections(left, right)


def are_adjacent_sections(left: str | None, right: str | None) -> bool:
    left_number = section_number(left)
    right_number = section_number(right)
    if not left_number or not right_number:
        return False
    if left_number[:-1] != right_number[:-1]:
        return False
    return abs(left_number[-1] - right_number[-1]) == 1


def section_number(section: str | None) -> tuple[int, ...] | None:
    match = re.match(r"^\s*(\d+(?:\.\d+)*)\b", sanitize_text(section or ""))
    if not match:
        return None
    return tuple(int(part) for part in match.group(1).split("."))


def text_jaccard(left: str, right: str) -> float:
    left_tokens = content_tokens(left)
    right_tokens = content_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def text_duplicate_score(left: str, right: str) -> float:
    left_tokens = content_tokens(left)
    right_tokens = content_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(left_tokens & right_tokens)
    return max(overlap / len(left_tokens | right_tokens), overlap / min(len(left_tokens), len(right_tokens)))


def has_long_text_containment(left: str, right: str) -> bool:
    left_text = normalize_for_containment(left)
    right_text = normalize_for_containment(right)
    if len(left_text) < 120 or len(right_text) < 120:
        return False
    shorter, longer = sorted([left_text, right_text], key=len)
    if has_enough_unique_tokens(shorter) and shorter in longer:
        return True
    window = min(len(shorter), 240)
    for start in range(0, max(1, len(shorter) - window + 1), 80):
        candidate = shorter[start : start + window]
        if has_enough_unique_tokens(candidate) and candidate in longer:
            return True
    return False


def normalize_for_containment(content: str) -> str:
    text = re.sub(r"^##\s*\d+(?:\.\d+)*\s+.*$", " ", sanitize_text(content), flags=re.MULTILINE)
    return re.sub(r"\s+", " ", text).strip().lower()


def has_enough_unique_tokens(content: str) -> bool:
    tokens = {token for token in re.findall(r"[a-z][a-z0-9_-]{2,}", content.lower())}
    return len(tokens) >= 8


def content_tokens(content: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", sanitize_text(content))
    }


def is_visual_chunk(chunk: Chunk) -> bool:
    text = sanitize_text(chunk.content).strip()
    lowered = text.lower()
    return (
        chunk.chunk_type in {"figure_caption", "table"}
        or bool(re.match(r"^(?:figure|fig\.)\s*\d+[:.\s]", text, flags=re.IGNORECASE))
        or lowered.startswith("table ")
        or ("|" in text and "---" in text)
    )


def merge_boundary_chunks(left: Chunk, right: Chunk) -> Chunk:
    content = sanitize_text(f"{left.content}\n\n{right.content}").strip()
    source_ids = sorted(set(left.source_block_ids or []) | set(right.source_block_ids or []))
    return Chunk(
        document_id=left.document_id,
        chunk_index=min(left.chunk_index, right.chunk_index),
        content=content,
        token_count=estimate_token_count(content),
        content_hash=content_hash(
            f"sentence-boundary:{left.document_id}:{left.section_name}:{','.join(map(str, source_ids))}:{content}"
        ),
        chunk_type=left.chunk_type,
        section_name=left.section_name,
        source_block_ids=source_ids,
        chunking_strategy=left.chunking_strategy,
        retrieval_value=left.retrieval_value,
        query_intents=left.query_intents,
        keywords=left.keywords,
        planner_reason=left.planner_reason,
    )


def detect_section_leakage(chunk: Chunk) -> bool:
    return find_trailing_section_heading(chunk.content) is not None


def find_trailing_section_heading(content: str) -> str | None:
    text = sanitize_text(content).rstrip()
    if not text:
        return None
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    heading = lines[-1].strip()
    return heading if TRAILING_SECTION_HEADING_RE.match(heading) else None


def fix_trailing_section_heading_leakage(chunks: list[Chunk]) -> BoundaryValidationResult:
    ordered = sorted(chunks, key=lambda chunk: (chunk.document_id, chunk.chunk_index))
    result = list(ordered)
    issues = 0
    fixes = 0
    dropped = 0
    drop_indexes: set[int] = set()

    for index, chunk in enumerate(ordered):
        if index in drop_indexes:
            continue
        heading = find_trailing_section_heading(chunk.content)
        if heading is None:
            continue

        issues += 1
        stripped_content = remove_trailing_heading(chunk.content, heading)
        next_index = find_forward_attach_chunk_index(result, index, heading)
        if next_index is not None:
            result[next_index] = rebuild_chunk_with_content(
                result[next_index],
                sanitize_text(f"{heading}\n\n{result[next_index].content}").strip(),
                section_name=extract_section_name_from_heading(heading),
            )
        if stripped_content:
            result[index] = rebuild_chunk_with_content(chunk, stripped_content)
        else:
            drop_indexes.add(index)
            dropped += 1
        fixes += 1

    return BoundaryValidationResult(
        chunks=[chunk for index, chunk in enumerate(result) if index not in drop_indexes],
        stats=BoundaryValidationStats(
            boundary_issues_found=issues,
            rule_fixes=fixes,
            dropped_chunks=dropped,
        ),
    )


def remove_trailing_heading(content: str, heading: str) -> str:
    text = sanitize_text(content).rstrip()
    lines = text.splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    if lines and lines[-1].strip() == heading:
        lines.pop()
    return sanitize_text("\n".join(lines)).strip()


def find_forward_attach_chunk_index(chunks: list[Chunk], index: int, heading: str) -> int | None:
    current = chunks[index]
    for candidate_index in range(index + 1, len(chunks)):
        candidate = chunks[candidate_index]
        if candidate.document_id != current.document_id:
            continue
        if section_heading_matches_chunk(heading, candidate):
            return candidate_index
        if normalize_heading_text(candidate.section_name or "") == normalize_heading_text(current.section_name or ""):
            return candidate_index
        return None
    return None


def section_heading_matches_chunk(heading: str, chunk: Chunk) -> bool:
    heading_text = normalize_heading_text(heading)
    section_text = normalize_heading_text(chunk.section_name or "")
    if not heading_text or not section_text:
        return False
    return heading_text == section_text or strip_leading_section_number(heading_text) == strip_leading_section_number(section_text)


def validate_section_name_consistency(chunk: Chunk) -> bool:
    first_line = first_non_empty_line(chunk.content)
    if first_line is None or not TRAILING_SECTION_HEADING_RE.match(first_line):
        return True
    heading_section = normalize_heading_text(derive_section_name_from_content(chunk.content) or "")
    chunk_section = normalize_heading_text(chunk.section_name or "")
    return bool(chunk_section) and heading_section == chunk_section


def normalize_heading_text(value: str) -> str:
    text = extract_section_name_from_heading(value).lower()
    text = re.sub(r"^##\s+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def strip_leading_section_number(value: str) -> str:
    return re.sub(r"^\d+(?:\.\d+)*\s+", "", value).strip()


def rebuild_chunk_with_content(
    chunk: Chunk,
    content: str,
    section_name: str | None = None,
    chunk_index: int | None = None,
) -> Chunk:
    clean_content = sanitize_text(content).strip()
    new_section_name = resolve_section_name(clean_content, section_name, chunk.section_name)
    new_chunk_index = chunk.chunk_index if chunk_index is None else chunk_index
    return Chunk(
        document_id=chunk.document_id,
        chunk_index=new_chunk_index,
        content=clean_content,
        token_count=estimate_token_count(clean_content),
        content_hash=content_hash(
            f"boundary-content:{chunk.document_id}:{new_chunk_index}:{chunk.chunk_type}:{new_section_name}:{clean_content}"
        ),
        chunk_type=AGENTIC_CHUNK_TYPE if chunk.chunking_strategy == "agentic" else chunk.chunk_type,
        section_name=new_section_name,
        source_block_ids=chunk.source_block_ids,
        chunking_strategy=chunk.chunking_strategy,
        retrieval_value=chunk.retrieval_value,
        query_intents=chunk.query_intents,
        keywords=chunk.keywords,
        planner_reason=chunk.planner_reason,
    )


def detach_backward_section_titles(
    source_ids: list[int],
    chunk: Chunk,
    block_by_id: dict[int, PaperBlock],
) -> tuple[list[int], dict[tuple[int, str], list[int]]]:
    if not source_ids:
        return source_ids, {}

    kept: list[int] = []
    pending: dict[tuple[int, str], list[int]] = {}
    for offset, block_id in enumerate(source_ids):
        block = block_by_id[block_id]
        if not is_heading_block(block):
            kept.append(block_id)
            continue

        following = source_ids[offset + 1 :]
        if offset == 0 and following and all(same_block_section(block, block_by_id[next_id]) for next_id in following):
            kept.append(block_id)
            continue

        key = (chunk.document_id, normalize_section(block.section_name))
        pending.setdefault(key, []).append(block_id)

    return kept, pending


def rebuild_chunk_from_blocks(
    template: Chunk,
    source_ids: list[int],
    block_by_id: dict[int, PaperBlock],
) -> Chunk:
    source_ids = same_paper_source_ids(source_ids, block_by_id)
    if not source_ids:
        return template
    blocks = [block_by_id[block_id] for block_id in source_ids]
    content = sanitize_text("\n\n".join(block.markdown_content or block.content for block in blocks)).strip()
    chunk_type = AGENTIC_CHUNK_TYPE if template.chunking_strategy == "agentic" else template.chunk_type
    section_name = resolve_section_name(
        content,
        choose_section_name(blocks, template.section_name),
        template.section_name,
    )
    return Chunk(
        document_id=template.document_id,
        chunk_index=min(block.order_index for block in blocks),
        content=content,
        token_count=estimate_token_count(content),
        content_hash=content_hash(
            f"boundary:{template.document_id}:{chunk_type}:{section_name}:{','.join(map(str, source_ids))}:{content}"
        ),
        chunk_type=chunk_type,
        section_name=section_name,
        source_block_ids=source_ids,
        chunking_strategy=template.chunking_strategy,
    )


def choose_section_name(blocks: list[PaperBlock], fallback: str | None) -> str | None:
    for block in blocks:
        if not is_heading_block(block) and block.section_name:
            return block.section_name
    return blocks[0].section_name or fallback


def same_paper_source_ids(source_ids: list[int], block_by_id: dict[int, PaperBlock]) -> list[int]:
    known_ids = [block_id for block_id in source_ids if block_id in block_by_id]
    if not known_ids:
        return []
    first_paper_id = block_by_id[known_ids[0]].paper_id
    return [block_id for block_id in known_ids if block_by_id[block_id].paper_id == first_paper_id]


def apply_model_boundary_fixes(
    chunks: list[Chunk],
    block_by_id: dict[int, PaperBlock],
    *,
    model: BoundaryPlanningModel | None,
    use_model: bool,
) -> BoundaryValidationResult:
    if not use_model or model is None:
        return BoundaryValidationResult(chunks=chunks, stats=BoundaryValidationStats())

    model_fixes = 0
    issues = 0
    dropped = 0
    current_chunks = list(chunks)
    for index, chunk in enumerate(list(current_chunks)):
        if not needs_model_boundary_check(chunk):
            continue
        try:
            plan = load_model_boundary_plan(model, current_chunks, index, block_by_id)
        except Exception:
            continue
        if not plan.get("has_boundary_issue"):
            continue
        issues += 1
        action = plan.get("action")
        if action == "drop_chunk":
            current_chunks = [candidate for candidate in current_chunks if candidate is not chunk]
            model_fixes += 1
            dropped += 1

    return BoundaryValidationResult(
        chunks=current_chunks,
        stats=BoundaryValidationStats(
            boundary_issues_found=issues,
            model_fixes=model_fixes,
            dropped_chunks=dropped,
        ),
    )


def needs_model_boundary_check(chunk: Chunk) -> bool:
    if chunk.token_count >= 80:
        return False
    content = sanitize_text(chunk.content).strip()
    return content.startswith(("Figure", "Fig.", "Table")) or looks_like_formula_fragment(content)


def looks_like_formula_fragment(content: str) -> bool:
    if not content:
        return False
    formula_markers = ("=", "∑", "∫", "\\frac", "\\sum", "\\int")
    return any(marker in content for marker in formula_markers) and len(content.split()) < 40


def load_model_boundary_plan(
    model: BoundaryPlanningModel,
    chunks: list[Chunk],
    index: int,
    block_by_id: dict[int, PaperBlock],
) -> dict[str, object]:
    response = model.generate(
        build_model_prompt(chunks, index, block_by_id),
        system_prompt=(
            "You are a chunk boundary validator. Return JSON only. "
            "Do not generate or rewrite paper content."
        ),
    )
    answer = getattr(response, "answer", response)
    payload = json.loads(extract_json_object(str(answer)))
    if not isinstance(payload, dict):
        raise ValueError("boundary validator model output must be an object")
    return payload


def build_model_prompt(chunks: list[Chunk], index: int, block_by_id: dict[int, PaperBlock]) -> str:
    window = chunks[max(0, index - 1) : min(len(chunks), index + 2)]
    return json.dumps(
        {
            "task": "Return a boundary correction plan only.",
            "schema": {
                "has_boundary_issue": True,
                "action": "move_block|merge_chunks|split_chunk|drop_chunk|keep",
                "source_block_ids": [1],
                "target_chunk_index": 1,
            },
            "chunks": [
                {
                    "chunk_index": chunk.chunk_index,
                    "chunk_type": chunk.chunk_type,
                    "section_name": chunk.section_name,
                    "source_block_ids": chunk.source_block_ids or [],
                    "content_preview": chunk.content[:300],
                }
                for chunk in window
            ],
            "source_blocks": [
                {
                    "id": block.id,
                    "section_name": block.section_name,
                    "content_preview": block.content[:300],
                }
                for chunk in window
                for block_id in (chunk.source_block_ids or [])
                if block_id in block_by_id
                for block in [block_by_id[block_id]]
            ],
        },
        ensure_ascii=False,
    )


def extract_json_object(value: str) -> str:
    start = value.find("{")
    end = value.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("model output does not contain JSON object")
    return value[start : end + 1]


def is_metadata_like_chunk(chunk: Chunk) -> bool:
    return normalize_section(chunk.section_name) == "unknown" and (
        is_metadata_like_content(chunk.content, chunk.section_name, chunk.chunk_index) or is_short_title_like(chunk.content)
    )


def is_metadata_like_content(
    content: str,
    section_name: str | None = None,
    order_index: int | None = None,
) -> bool:
    return is_shared_metadata_noise(content, section_name, order_index) or looks_like_metadata(content) or looks_like_arxiv_noise(content)


def looks_like_metadata(content: str) -> bool:
    lowered = sanitize_text(content).strip().lower()
    if not lowered:
        return True
    return (
        "@" in lowered
        or "university" in lowered
        or "department of" in lowered
        or "school of" in lowered
        or "affiliation" in lowered
        or lowered.startswith("authors")
    )


def looks_like_arxiv_noise(content: str) -> bool:
    lowered = sanitize_text(content).strip().lower()
    return lowered.startswith("arxiv:") or "arxiv.org" in lowered or "copyright" in lowered or "license" in lowered


def is_short_title_like(content: str) -> bool:
    text = sanitize_text(content).strip()
    if not text or text.startswith("## "):
        return False
    words = text.split()
    if len(words) > 14:
        return False
    if any(mark in text for mark in (".", "?", "!", ":", ";")):
        return False
    return any(char.isalpha() for char in text)


def is_heading_block(block: PaperBlock) -> bool:
    content = sanitize_text(block.markdown_content or block.content).strip()
    first_line = first_non_empty_line(content)
    if not first_line:
        return False
    if first_line.startswith("## "):
        return True
    return normalize_section(block.section_name) == normalize_section(first_line) and is_short_title_like(first_line)


def same_block_section(left: PaperBlock, right: PaperBlock) -> bool:
    return normalize_section(left.section_name) == normalize_section(right.section_name)


def normalize_section(section_name: str | None) -> str:
    return (section_name or "unknown").strip().lower()
