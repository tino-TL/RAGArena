from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from dataclasses import replace
from typing import Protocol

import requests

from ragarena.chunking.block_chunker import chunk_block_documents, map_documents_by_arxiv_id
from ragarena.chunking.fixed_chunker import Chunk, estimate_token_count
from ragarena.chunking.repository import DocumentRecord
from ragarena.config import settings
from ragarena.ingestion.hashing import content_hash
from ragarena.papers.metadata_noise import is_metadata_noise as is_shared_metadata_noise
from ragarena.papers.models import PaperBlock
from ragarena.chunking.section_names import resolve_section_name
from ragarena.utils.text import sanitize_text

AGENTIC_CHUNK_SYSTEM_PROMPT = """You are a chunk boundary planner for scientific papers.
Return JSON only. No markdown, no prose, no explanation.
You must not summarize, rewrite, explain, create queries, or generate keywords.
Only choose adjacent source_block_ids from the provided Docling PaperBlocks.
Final chunk text will be reconstructed from original PaperBlocks, not from your output.
"""

SUPPORTED_CHUNK_TYPES = {
    "abstract",
    "background",
    "method",
    "experiment",
    "table",
    "figure_caption",
    "code",
    "conclusion",
    "other",
}
AGENTIC_CHUNK_TYPE = "retrieval_unit"
FUSED_CHUNK_TYPE = "fused"

SHORT_CONTEXT_TOKEN_LIMIT = 150
SHORT_CAPTION_STANDALONE_TOKENS = 80
TABLE_EMBED_TOP_ROWS = 5
MIN_RETRIEVAL_UNIT_TOKENS = 180
TARGET_RETRIEVAL_UNIT_TOKENS = 400
MAX_RETRIEVAL_UNIT_TOKENS = 800
OVERLAP_MERGE_THRESHOLD = 0.5
ENABLE_CHUNK_METADATA = False


@dataclass(frozen=True)
class ChunkPlanItem:
    chunk_type: str
    section_name: str
    source_block_ids: list[int]
    should_embed: bool


@dataclass(frozen=True)
class AgenticChunkTrace:
    section_name: str
    input_block_count: int
    generated_chunk_count: int
    fallback_reason: str | None = None
    total_duration: int | None = None
    load_duration: int | None = None
    eval_duration: int | None = None
    load_warning: str | None = None


@dataclass(frozen=True)
class AgenticChunkResult:
    chunks: list[Chunk]
    traces: list[AgenticChunkTrace]
    stats: dict[str, object]


@dataclass(frozen=True)
class ChunkCleanupResult:
    chunks: list[Chunk]
    dropped_tiny_chunks: int = 0
    merged_tiny_chunks: int = 0


class ChunkPlanningModel(Protocol):
    def is_configured(self) -> bool: ...

    def generate(self, prompt: str, system_prompt: str = AGENTIC_CHUNK_SYSTEM_PROMPT) -> "GenerationLike": ...


class GenerationLike(Protocol):
    @property
    def answer(self) -> str: ...


@dataclass(frozen=True)
class PlannerGenerationResult:
    answer: str
    total_duration: int | None = None
    load_duration: int | None = None
    eval_duration: int | None = None


@dataclass(frozen=True)
class ChunkPlanResult:
    plan: list[ChunkPlanItem]
    total_duration: int | None = None
    load_duration: int | None = None
    eval_duration: int | None = None


class OllamaChunkPlanner:
    def __init__(self, url: str, model: str, timeout: int = 120, keep_alive: str | None = None) -> None:
        self.url = url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.keep_alive = keep_alive or settings.ollama_keep_alive

    def is_configured(self) -> bool:
        try:
            response = requests.get(f"{self.url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def generate(self, prompt: str, system_prompt: str = AGENTIC_CHUNK_SYSTEM_PROMPT) -> PlannerGenerationResult:
        response = requests.post(
            f"{self.url}/api/chat",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "format": "json",
                "keep_alive": self.keep_alive,
                "think": False,
                "options": {"temperature": 0},
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        message = payload.get("message") or {}
        return PlannerGenerationResult(
            answer=str(message.get("content", "")).strip(),
            total_duration=duration_value(payload.get("total_duration")),
            load_duration=duration_value(payload.get("load_duration")),
            eval_duration=duration_value(payload.get("eval_duration")),
        )


def chunk_agentic_documents(
    documents: list[DocumentRecord],
    blocks: list[PaperBlock],
    *,
    model: ChunkPlanningModel | None = None,
    max_tokens: int | None = None,
    enabled: bool | None = None,
    planner_provider: str | None = None,
    planner_model: str | None = None,
    debug_planner: bool = False,
) -> AgenticChunkResult:
    traces: list[AgenticChunkTrace] = []
    if enabled is None:
        enabled = settings.agentic_chunk_enabled
    if max_tokens is None:
        max_tokens = settings.agentic_chunk_max_tokens
    if not enabled:
        return fallback_to_block(documents, blocks, traces, "agentic chunking disabled")

    planner = model or build_planner(planner_provider, planner_model)
    if not planner.is_configured():
        return fallback_to_block(documents, blocks, traces, "planning model unavailable")

    document_by_arxiv_id = map_documents_by_arxiv_id(documents)
    block_by_id = {block.id: block for block in blocks if block.id is not None}
    chunks: list[Chunk] = []
    before_merge_count = 0

    for planning_key, section_blocks in group_blocks_for_planning(blocks).items():
        _, section_name, _ = planning_key
        for window in window_blocks(section_blocks, max_tokens=max_tokens):
            try:
                planned = request_chunk_plan(planner, section_name, window, debug_planner=debug_planner)
                plan = validate_and_repair_plan_adjacency(planned.plan, window)
                plan = add_missing_embeddable_blocks(plan, window)
                plan = normalize_retrieval_unit_plan(plan, window)
            except Exception as exc:
                traces.append(
                    AgenticChunkTrace(
                        section_name=section_name,
                        input_block_count=len(window),
                        generated_chunk_count=0,
                        fallback_reason=str(exc),
                    )
                )
                fallback_cleanup = cleanup_retrieval_chunks(
                    coerce_agentic_retrieval_units(chunk_block_documents(documents, window))
                )
                chunks.extend(fallback_cleanup.chunks)
                continue

            traces.append(
                AgenticChunkTrace(
                    section_name=section_name,
                    input_block_count=len(window),
                    generated_chunk_count=len(plan),
                    total_duration=planned.total_duration,
                    load_duration=planned.load_duration,
                    eval_duration=planned.eval_duration,
                    load_warning=build_load_warning(planned.load_duration, planned.total_duration),
                )
            )
            for item in plan:
                if not item.should_embed:
                    continue
                selected_blocks = [
                    block_by_id[block_id]
                    for block_id in item.source_block_ids
                    if block_id in block_by_id and should_include_source_block(block_by_id[block_id])
                ]
                if not selected_blocks:
                    continue
                if not same_paper_blocks(selected_blocks):
                    raise ValueError(f"chunk plan crossed paper_id boundary: {item.source_block_ids}")
                document = document_by_arxiv_id.get(selected_blocks[0].arxiv_id)
                if document is None:
                    continue
                chunks.append(build_agentic_chunk(document, selected_blocks, item))

    before_merge_count = len(chunks)
    chunks = merge_small_agentic_chunks(
        chunks,
        min_tokens=settings.agentic_chunk_min_tokens,
        target_tokens=settings.agentic_chunk_target_tokens,
        max_tokens=MAX_RETRIEVAL_UNIT_TOKENS,
    )
    cleanup = cleanup_retrieval_chunks(chunks)
    chunks = optimize_retrieval_units(cleanup.chunks, blocks)
    if not chunks:
        return AgenticChunkResult(
            chunks=[],
            traces=traces,
            stats=planner_stats(chunks, before_merge_count=before_merge_count, cleanup=cleanup),
        )

    return AgenticChunkResult(
        chunks=chunks,
        traces=traces,
        stats=planner_stats(chunks, before_merge_count=before_merge_count, cleanup=cleanup),
    )


def build_planner(provider: str | None, model: str | None) -> ChunkPlanningModel:
    selected_provider = provider or settings.agentic_chunk_provider
    selected_model = model or settings.agentic_chunk_model
    if selected_provider == "ollama":
        return OllamaChunkPlanner(settings.ollama_url, selected_model, keep_alive=settings.ollama_keep_alive)
    raise ValueError(f"Unsupported chunk planner provider: {selected_provider}")


def request_chunk_plan(
    planner: ChunkPlanningModel,
    section_name: str,
    blocks: list[PaperBlock],
    *,
    debug_planner: bool = False,
) -> ChunkPlanResult:
    prompt = build_planning_prompt(section_name, blocks)
    result = planner.generate(prompt, system_prompt=AGENTIC_CHUNK_SYSTEM_PROMPT)
    if debug_planner:
        print(f"Planner raw output for section={section_name}: {result.answer}")
    try:
        return chunk_plan_result(validate_chunk_plan_payload(load_planner_json(result.answer)), result)
    except Exception as exc:
        repaired_result = planner.generate(build_repair_prompt(result.answer), system_prompt="Fix JSON. Return JSON only.")
        repaired = repaired_result.answer
        if debug_planner:
            print(f"Planner repaired output for section={section_name}: {repaired}")
        try:
            return chunk_plan_result(validate_chunk_plan_payload(load_planner_json(repaired)), repaired_result)
        except Exception as repaired_exc:
            raise ValueError(f"invalid chunk plan after repair: {repaired_exc}") from exc


def chunk_plan_result(plan: list[ChunkPlanItem], result: GenerationLike) -> ChunkPlanResult:
    return ChunkPlanResult(
        plan=plan,
        total_duration=duration_value(getattr(result, "total_duration", None)),
        load_duration=duration_value(getattr(result, "load_duration", None)),
        eval_duration=duration_value(getattr(result, "eval_duration", None)),
    )


def duration_value(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def build_load_warning(load_duration: int | None, total_duration: int | None) -> str | None:
    if load_duration is None or total_duration is None or total_duration <= 0:
        return None
    if load_duration >= 1_000_000_000 and load_duration / total_duration >= 0.25:
        return "high load_duration; Ollama model may be reloading between planner requests"
    return None


def build_planning_prompt(section_name: str, blocks: list[PaperBlock]) -> str:
    block_lines = []
    for block in blocks:
        block_lines.append(
            {
                "id": block.id,
                "section_name": block.section_name,
                "order_index": block.order_index,
                "token_count": estimate_token_count(block.content),
                "content_preview": sanitize_text(block.content)[:500],
            }
        )
    return json.dumps(
        {
            "section_name": section_name,
            "task": "Plan chunk boundaries only. Select source_block_ids; do not write chunk content.",
            "rules": [
                "Abstract usually becomes one chunk.",
                "Section title should not be standalone; merge it into following related content.",
                "Paragraphs in the same section should be merged by semantic continuity.",
                "Figure image blocks are not chunked.",
                "Figure captions should stay as figure_caption chunks unless the caption is only a broken fragment.",
                "Tables and table captions should stay as table chunks. Do not merge tables into body paragraphs.",
                "Formula blocks should merge into nearby method paragraphs; isolated formulas are not embedded.",
                "Short code snippets should merge with context; long code and algorithms can be standalone.",
                "Reference, authors, affiliation, footnote are not chunked.",
                "Target 300-800 tokens. Avoid chunks under 150 tokens except abstract or genuinely standalone long code/table/caption.",
                "Do not merge across different major sections.",
                "Use adjacent source_block_ids only.",
            ],
            "output_schema": {
                "chunks": [
                    {
                        "chunk_type": "abstract|background|method|experiment|table|figure_caption|code|conclusion|other",
                        "section_name": "string",
                        "source_block_ids": [1, 2, 3],
                        "should_embed": True,
                    }
                ]
            },
            "blocks": block_lines,
        },
        ensure_ascii=False,
    )


def load_planner_json(value: str) -> object:
    return json.loads(extract_json_object(value))


def extract_json_object(value: str) -> str:
    text = sanitize_text(value).strip()
    fenced = re_search_json_fence(text)
    if fenced:
        return fenced
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("planner output does not contain a JSON object")
    return text[start : end + 1]


def re_search_json_fence(text: str) -> str | None:
    import re

    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else None


def validate_chunk_plan_payload(payload: object) -> list[ChunkPlanItem]:
    if not isinstance(payload, dict):
        raise ValueError("chunk plan JSON must be an object")
    chunks = payload.get("chunks")
    if not isinstance(chunks, list):
        raise ValueError("chunk plan JSON must contain a chunks list")
    return [parse_plan_item(item) for item in chunks]


def parse_plan_item(value: object) -> ChunkPlanItem:
    if not isinstance(value, dict):
        raise ValueError("chunk plan item must be an object")
    source_block_ids = [int(item) for item in value.get("source_block_ids", [])]
    if not source_block_ids:
        raise ValueError("chunk plan item must contain source_block_ids")
    chunk_type = str(value.get("chunk_type", "other"))
    if chunk_type not in SUPPORTED_CHUNK_TYPES:
        chunk_type = "other"
    return ChunkPlanItem(
        chunk_type=chunk_type,
        section_name=str(value.get("section_name", "unknown")),
        source_block_ids=source_block_ids,
        should_embed=bool(value.get("should_embed", True)),
    )


def validate_and_repair_plan_adjacency(
    plan: list[ChunkPlanItem],
    blocks: list[PaperBlock],
) -> list[ChunkPlanItem]:
    position_by_id = {block.id: index for index, block in enumerate(blocks) if block.id is not None}
    repaired: list[ChunkPlanItem] = []
    for item in plan:
        known_ids = [block_id for block_id in item.source_block_ids if block_id in position_by_id]
        if not known_ids:
            continue
        ordered_ids = sorted(known_ids, key=lambda block_id: position_by_id[block_id])
        if not are_adjacent(ordered_ids, position_by_id):
            raise ValueError(f"non-adjacent source_block_ids rejected: {item.source_block_ids}")
        repaired.append(
            ChunkPlanItem(
                chunk_type=item.chunk_type,
                section_name=item.section_name,
                source_block_ids=ordered_ids,
                should_embed=item.should_embed,
            )
        )
    return repaired


def add_missing_embeddable_blocks(
    plan: list[ChunkPlanItem],
    blocks: list[PaperBlock],
) -> list[ChunkPlanItem]:
    planned_ids = {block_id for item in plan for block_id in item.source_block_ids}
    repaired = list(plan)
    for block in blocks:
        if block.id is None or block.id in planned_ids:
            continue
        if should_plan_block(block) and block.id is not None:
            repaired.append(
                ChunkPlanItem(
                    chunk_type="other",
                    section_name=block.section_name or "unknown",
                    source_block_ids=[block.id],
                    should_embed=True,
                )
            )
    return repaired


def normalize_retrieval_unit_plan(
    plan: list[ChunkPlanItem],
    blocks: list[PaperBlock],
) -> list[ChunkPlanItem]:
    block_by_id = {block.id: block for block in blocks if block.id is not None}
    ordered = sorted(plan, key=lambda item: min(block_by_id[block_id].order_index for block_id in item.source_block_ids if block_id in block_by_id))
    normalized: list[ChunkPlanItem] = []

    for item in ordered:
        item_blocks = [block_by_id[block_id] for block_id in item.source_block_ids if block_id in block_by_id]
        if not item_blocks:
            continue
        token_count = sum(estimate_token_count(block.markdown_content or block.content) for block in item_blocks)
        if not item.should_embed:
            normalized.append(item)
            continue
        content = "\n\n".join(block.markdown_content or block.content for block in item_blocks)
        if should_drop_standalone_plan_content(content, token_count):
            normalized.append(replace_plan_item(item, should_embed=False))
            continue
        if should_attach_contextual_content(content, token_count):
            normalized.append(replace_plan_item(item, should_embed=False))
            continue
        normalized.append(item)

    return merge_contextual_plan_items(normalized, block_by_id)


def merge_contextual_plan_items(
    plan: list[ChunkPlanItem],
    block_by_id: dict[int, PaperBlock],
) -> list[ChunkPlanItem]:
    items = list(plan)
    index = 0
    while index < len(items):
        item = items[index]
        item_blocks = [block_by_id[block_id] for block_id in item.source_block_ids if block_id in block_by_id]
        if not item_blocks or not should_attach_contextual_item(item, item_blocks):
            index += 1
            continue

        neighbor_index = find_neighbor_for_contextual_item(items, index, block_by_id)
        if neighbor_index is None:
            index += 1
            continue

        merged = merge_plan_items(items[neighbor_index], item, block_by_id)
        keep = [candidate for offset, candidate in enumerate(items) if offset not in {index, neighbor_index}]
        keep.append(merged)
        items = sorted_plan_items(keep, block_by_id)
        index = 0
    return items


def should_attach_contextual_item(item: ChunkPlanItem, blocks: list[PaperBlock]) -> bool:
    if not item.should_embed:
        return True
    content = "\n\n".join(block.markdown_content or block.content for block in blocks)
    return should_attach_contextual_content(content, estimate_token_count(content))


def find_neighbor_for_contextual_item(
    items: list[ChunkPlanItem],
    index: int,
    block_by_id: dict[int, PaperBlock],
) -> int | None:
    item = items[index]
    for neighbor_index in (index + 1, index - 1):
        if neighbor_index < 0 or neighbor_index >= len(items):
            continue
        neighbor = items[neighbor_index]
        if not neighbor.should_embed:
            continue
        if not same_section(item, neighbor, block_by_id):
            continue
        if plan_items_touch(item, neighbor, block_by_id):
            return neighbor_index
    return None


def same_section(left: ChunkPlanItem, right: ChunkPlanItem, block_by_id: dict[int, PaperBlock]) -> bool:
    left_sections = {block_by_id[block_id].section_name for block_id in left.source_block_ids if block_id in block_by_id}
    right_sections = {block_by_id[block_id].section_name for block_id in right.source_block_ids if block_id in block_by_id}
    return bool(left_sections & right_sections)


def plan_items_touch(left: ChunkPlanItem, right: ChunkPlanItem, block_by_id: dict[int, PaperBlock]) -> bool:
    left_orders = [block_by_id[block_id].order_index for block_id in left.source_block_ids if block_id in block_by_id]
    right_orders = [block_by_id[block_id].order_index for block_id in right.source_block_ids if block_id in block_by_id]
    if not left_orders or not right_orders:
        return False
    return abs(min(right_orders) - max(left_orders)) == 1 or abs(min(left_orders) - max(right_orders)) == 1


def merge_plan_items(left: ChunkPlanItem, right: ChunkPlanItem, block_by_id: dict[int, PaperBlock]) -> ChunkPlanItem:
    source_block_ids = sorted(set(left.source_block_ids) | set(right.source_block_ids), key=lambda block_id: block_by_id[block_id].order_index)
    return ChunkPlanItem(
        chunk_type="other",
        section_name=left.section_name if left.section_name != "unknown" else right.section_name,
        source_block_ids=source_block_ids,
        should_embed=left.should_embed or right.should_embed,
    )


def sorted_plan_items(items: list[ChunkPlanItem], block_by_id: dict[int, PaperBlock]) -> list[ChunkPlanItem]:
    return sorted(items, key=lambda item: min(block_by_id[block_id].order_index for block_id in item.source_block_ids if block_id in block_by_id))


def replace_plan_item(item: ChunkPlanItem, *, should_embed: bool) -> ChunkPlanItem:
    return ChunkPlanItem(
        chunk_type=item.chunk_type,
        section_name=item.section_name,
        source_block_ids=item.source_block_ids,
        should_embed=should_embed,
    )


def are_adjacent(ids: list[int], order_by_id: dict[int, int]) -> bool:
    orders = [order_by_id[block_id] for block_id in ids]
    return orders == list(range(min(orders), max(orders) + 1))


def build_agentic_chunk(
    document: DocumentRecord,
    blocks: list[PaperBlock],
    item: ChunkPlanItem,
) -> Chunk:
    content = "\n\n".join(chunk_content_for_block(block) for block in blocks)
    clean_content = clean_retrieval_content(content)
    source_block_ids = [block.id for block in blocks if block.id is not None]
    section_name = resolve_section_name(clean_content, item.section_name, None)
    return Chunk(
        document_id=document.id,
        chunk_index=min(block.order_index for block in blocks),
        content=clean_content,
        token_count=estimate_token_count(clean_content),
        content_hash=content_hash(
            f"agentic:{document.id}:{AGENTIC_CHUNK_TYPE}:{section_name}:{','.join(map(str, source_block_ids))}:{clean_content}"
        ),
        chunk_type=semantic_output_chunk_type(item, blocks),
        section_name=section_name,
        source_block_ids=source_block_ids,
        chunking_strategy="agentic",
    )


def semantic_output_chunk_type(item: ChunkPlanItem, blocks: list[PaperBlock]) -> str:
    block_types = {block.block_type for block in blocks}
    content = "\n\n".join(block.markdown_content or block.content for block in blocks)
    if item.chunk_type == "figure_caption" or block_types == {"figure_caption"}:
        return "figure_caption"
    if bool(block_types & {"table", "table_caption"}) or (
        item.chunk_type == "table" and (looks_like_markdown_table(content) or is_caption_like(content))
    ):
        return "table"
    return AGENTIC_CHUNK_TYPE


def coerce_agentic_retrieval_units(chunks: list[Chunk]) -> list[Chunk]:
    coerced: list[Chunk] = []
    for chunk in chunks:
        clean_content = clean_retrieval_content(chunk.content)
        coerced.append(
            Chunk(
                document_id=chunk.document_id,
                chunk_index=chunk.chunk_index,
                content=clean_content,
                token_count=estimate_token_count(clean_content),
                content_hash=content_hash(
                    f"agentic-fallback:{chunk.document_id}:{chunk.section_name}:{chunk.chunk_index}:{clean_content}"
                ),
                chunk_type=AGENTIC_CHUNK_TYPE,
                section_name=chunk.section_name,
                source_block_ids=chunk.source_block_ids,
                chunking_strategy="agentic",
                retrieval_value=chunk.retrieval_value,
                query_intents=chunk.query_intents,
                keywords=chunk.keywords,
                planner_reason=chunk.planner_reason,
            )
        )
    return coerced


def chunk_content_for_block(block: PaperBlock) -> str:
    if block.block_type == "section_title":
        return block.markdown_content or f"## {block.content.strip()}"
    if block.markdown_content and looks_like_markdown_table(block.markdown_content):
        return truncate_markdown_table(block.markdown_content, top_rows=TABLE_EMBED_TOP_ROWS)
    return block.markdown_content or block.content


def clean_retrieval_content(content: str) -> str:
    text = sanitize_text(content)
    text = re_sub(r"(?im)^\s*<!--\s*image[^>]*-->\s*$", "", text)
    text = normalize_known_joined_terms(text)
    text = re_sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_known_joined_terms(text: str) -> str:
    replacements = {
        r"\bWeuse\b": "We use",
        r"\bfreetext\b": "free-text",
        r"\bgroundtruth\b": "ground-truth",
        r"\bontologygrounded\b": "ontology-grounded",
        r"\bobjectdecomposed\b": "object-decomposed",
        r"\bgeneralpurpose\b": "general-purpose",
        r"\bin-thewild\b": "in-the-wild",
        r"\bvisionas-inverse-graphics\b": "vision-as-inverse-graphics",
    }
    output = text
    for pattern, replacement in replacements.items():
        output = re_sub(pattern, replacement, output)
    return output


def truncate_markdown_table(markdown: str, *, top_rows: int) -> str:
    lines = [line for line in markdown.splitlines() if line.strip()]
    if len(lines) <= top_rows + 2:
        return markdown
    return "\n".join(lines[: top_rows + 2])


def should_plan_block(block: PaperBlock) -> bool:
    content = sanitize_text(block.markdown_content or block.content).strip()
    if not content:
        return False
    if not block.should_embed:
        return False
    if normalize_section_name(block.section_name) == "unknown" and is_short_title_like(content):
        return False
    if is_metadata_noise(content, block.section_name, block.order_index):
        return False
    return True


def should_drop_standalone_plan_content(content: str, token_count: int) -> bool:
    text = sanitize_text(content).strip()
    if not text:
        return True
    if is_metadata_noise(text):
        return True
    if is_section_heading_text(text):
        return True
    if "<!-- formula-not-decoded" in text:
        return False
    if is_formula_like(text) and token_count < SHORT_CONTEXT_TOKEN_LIMIT:
        return True
    if is_image_reference(text):
        return True
    return False


def should_attach_contextual_content(content: str, token_count: int) -> bool:
    text = sanitize_text(content).strip()
    if "<!-- formula-not-decoded" in text:
        return False
    if token_count >= SHORT_CONTEXT_TOKEN_LIMIT:
        return False
    return (
        is_section_heading_text(text)
        or is_formula_like(text)
        or is_image_reference(text)
    )


def is_section_heading_text(content: str) -> bool:
    lines = [line.strip() for line in sanitize_text(content).splitlines() if line.strip()]
    return len(lines) == 1 and (lines[0].startswith("## ") or is_plain_section_heading(lines[0]))


def is_plain_section_heading(text: str) -> bool:
    if len(text.split()) > 12:
        return False
    if text.endswith("."):
        return False
    return bool(re_match(r"^(\d+(?:\.\d+)*\s+)?[A-Z][\w/&: -]+$", text))


def is_caption_like(content: str) -> bool:
    return bool(re_match(r"^(Figure|Fig\.|Table)\s*\d+[:.]", sanitize_text(content).strip(), ignore_case=True))


def looks_like_markdown_table(content: str) -> bool:
    lines = [line.strip() for line in sanitize_text(content).splitlines() if line.strip()]
    return any("|" in line for line in lines) and any("---" in line for line in lines)


def is_formula_like(content: str) -> bool:
    text = sanitize_text(content)
    formula_markers = ("<!-- formula", "\\frac", "\\sum", "\\int", "∑", "∫")
    return any(marker in text for marker in formula_markers) or bool(re_match(r"^[A-Za-z]\s*=", text.strip()))


def is_image_reference(content: str) -> bool:
    return "<!-- image" in sanitize_text(content).lower()


def is_metadata_noise(content: str, section_name: str | None = None, order_index: int | None = None) -> bool:
    return is_shared_metadata_noise(content, section_name, order_index) or looks_like_metadata(content) or is_arxiv_noise(content)


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


def is_arxiv_noise(content: str) -> bool:
    lowered = sanitize_text(content).strip().lower()
    return lowered.startswith("arxiv:") or "arxiv.org" in lowered or "copyright" in lowered or "license" in lowered


def re_match(pattern: str, text: str, *, ignore_case: bool = False):
    import re

    flags = re.IGNORECASE if ignore_case else 0
    return re.match(pattern, text, flags=flags)


def re_sub(pattern: str, replacement: str, text: str) -> str:
    import re

    return re.sub(pattern, replacement, text)


def group_blocks_for_planning(blocks: list[PaperBlock]) -> dict[tuple[int, str, str], list[PaperBlock]]:
    grouped: dict[tuple[int, str, str], list[PaperBlock]] = defaultdict(list)
    for block in blocks:
        if block.id is None:
            continue
        if not should_plan_block(block):
            continue
        grouped[(block.paper_id, block.section_name or "unknown", block_planning_lane(block))].append(block)
    return dict(grouped)


def block_planning_lane(block: PaperBlock) -> str:
    if block.block_type == "figure_caption":
        return "figure"
    if block.block_type in {"table", "table_caption"}:
        return "table"
    return "body"


def window_blocks(blocks: list[PaperBlock], *, max_tokens: int) -> list[list[PaperBlock]]:
    windows: list[list[PaperBlock]] = []
    current: list[PaperBlock] = []
    current_tokens = 0
    current_paper_id: int | None = None
    for block in blocks:
        block_tokens = estimate_token_count(block.content)
        crosses_paper = current_paper_id is not None and block.paper_id != current_paper_id
        if current and (crosses_paper or current_tokens + block_tokens > max_tokens):
            windows.append(current)
            current = []
            current_tokens = 0
            current_paper_id = None
        current.append(block)
        current_tokens += block_tokens
        current_paper_id = block.paper_id
    if current:
        windows.append(current)
    return windows


def fallback_to_block(
    documents: list[DocumentRecord],
    blocks: list[PaperBlock],
    traces: list[AgenticChunkTrace],
    reason: str,
) -> AgenticChunkResult:
    traces.append(
        AgenticChunkTrace(
            section_name="fallback",
            input_block_count=len(blocks),
            generated_chunk_count=0,
            fallback_reason=reason,
        )
    )
    raw_chunks = chunk_block_documents(documents, blocks)
    cleanup = cleanup_retrieval_chunks(raw_chunks)
    chunks = optimize_retrieval_units(cleanup.chunks, blocks)
    return AgenticChunkResult(
        chunks=chunks,
        traces=traces,
        stats=planner_stats(chunks, before_merge_count=len(raw_chunks), cleanup=cleanup),
    )


def merge_small_agentic_chunks(
    chunks: list[Chunk],
    *,
    min_tokens: int = MIN_RETRIEVAL_UNIT_TOKENS,
    target_tokens: int = TARGET_RETRIEVAL_UNIT_TOKENS,
    max_tokens: int = MAX_RETRIEVAL_UNIT_TOKENS,
) -> list[Chunk]:
    if not chunks:
        return []

    ordered = sorted(chunks, key=lambda chunk: (chunk.document_id, chunk.chunk_index))
    items = list(ordered)
    index = 0
    while index < len(items):
        chunk = items[index]
        if not should_micro_merge_chunk(chunk, min_tokens=min_tokens):
            index += 1
            continue
        neighbor_index = find_micro_merge_neighbor(items, index, target_tokens=target_tokens, max_tokens=max_tokens)
        if neighbor_index is None:
            index += 1
            continue
        left_index, right_index = sorted((index, neighbor_index))
        merged = merge_two_chunks(items[left_index], items[right_index])
        items = [candidate for offset, candidate in enumerate(items) if offset not in {left_index, right_index}]
        items.insert(left_index, merged)
        index = max(left_index - 1, 0)
    return reindex_retrieval_units(items)


def should_micro_merge_chunk(chunk: Chunk, *, min_tokens: int) -> bool:
    if is_figure_or_table_chunk(chunk):
        return False
    if chunk.token_count >= min_tokens:
        return False
    if is_abstract_or_introduction(chunk):
        return False
    if is_table_result_chunk(chunk) and chunk.token_count > 300:
        return False
    return True


def find_micro_merge_neighbor(
    chunks: list[Chunk],
    index: int,
    *,
    target_tokens: int,
    max_tokens: int,
) -> int | None:
    fallback_index: int | None = None
    for neighbor_index in (index - 1, index + 1):
        if neighbor_index < 0 or neighbor_index >= len(chunks):
            continue
        if can_micro_merge_chunks(chunks[index], chunks[neighbor_index], max_tokens=max_tokens):
            if chunks[index].token_count + chunks[neighbor_index].token_count <= target_tokens:
                return neighbor_index
            if fallback_index is None:
                fallback_index = neighbor_index
    return fallback_index


def can_micro_merge_chunks(left: Chunk, right: Chunk, *, max_tokens: int) -> bool:
    if left.document_id != right.document_id:
        return False
    if is_figure_or_table_chunk(left) or is_figure_or_table_chunk(right):
        return False
    if left.chunking_strategy != right.chunking_strategy:
        return False
    if is_abstract_or_introduction(left) or is_abstract_or_introduction(right):
        return False
    if left.token_count + right.token_count > max_tokens:
        return False
    if is_table_result_chunk(left) and left.token_count > 300:
        return False
    if is_table_result_chunk(right) and right.token_count > 300:
        return False
    return same_section_or_parent_child_section(left.section_name, right.section_name)


def same_section_or_parent_child_section(left: str | None, right: str | None) -> bool:
    left_normalized = normalize_section_name(left)
    right_normalized = normalize_section_name(right)
    if left_normalized == right_normalized:
        return True
    left_number = section_number_key(left)
    right_number = section_number_key(right)
    if not left_number or not right_number:
        return False
    return left_number == right_number[:-1] or right_number == left_number[:-1]


def is_abstract_or_introduction(chunk: Chunk) -> bool:
    section = normalize_section_name(chunk.section_name)
    return section == "abstract" or "introduction" in section


def is_table_result_chunk(chunk: Chunk) -> bool:
    text = sanitize_text(f"{chunk.section_name or ''}\n{chunk.content}").lower()
    return chunk.chunk_type == "table" or "|" in chunk.content or "table" in text or "result" in text


def is_figure_or_table_chunk(chunk: Chunk) -> bool:
    text = sanitize_text(chunk.content).strip()
    lowered = text.lower()
    return is_caption_like(text) or looks_like_markdown_table(text) or lowered.startswith("table:")


def same_parent_or_major_section(left: str | None, right: str | None) -> bool:
    left_major = major_section_key(left)
    right_major = major_section_key(right)
    if left_major is None or right_major is None:
        return normalize_section_name(left) == normalize_section_name(right)
    if left_major != right_major:
        return False
    left_parent = parent_section_key(left)
    right_parent = parent_section_key(right)
    return left_parent == right_parent or left_parent == major_section_key(right) or right_parent == major_section_key(left)


def major_section_key(section_name: str | None) -> str | None:
    match = re_match(r"^\s*(\d+)(?:\.\d+)*\b", sanitize_text(section_name or ""))
    return str(match.group(1)) if match else None


def parent_section_key(section_name: str | None) -> str | None:
    match = re_match(r"^\s*(\d+(?:\.\d+)*)\b", sanitize_text(section_name or ""))
    if not match:
        return normalize_section_name(section_name)
    parts = match.group(1).split(".")
    if len(parts) == 1:
        return parts[0]
    return ".".join(parts[:-1])


def cleanup_retrieval_chunks(chunks: list[Chunk]) -> ChunkCleanupResult:
    ordered = sorted(chunks, key=lambda chunk: (chunk.document_id, chunk.chunk_index))
    title_result = merge_or_drop_section_titles(ordered)
    tiny_result = merge_or_drop_tiny_chunks(title_result.chunks)
    return ChunkCleanupResult(
        chunks=tiny_result.chunks,
        dropped_tiny_chunks=title_result.dropped_tiny_chunks + tiny_result.dropped_tiny_chunks,
        merged_tiny_chunks=title_result.merged_tiny_chunks + tiny_result.merged_tiny_chunks,
    )


def optimize_retrieval_units(chunks: list[Chunk], blocks: list[PaperBlock] | None = None) -> list[Chunk]:
    """Improve retrieval-unit boundaries without relying on parser block_type metadata."""
    block_by_id = {block.id: block for block in blocks or [] if block.id is not None}
    covered = ensure_section_coverage(chunks, block_by_id)
    overlapped = merge_overlapping_retrieval_units(covered, block_by_id)
    micro_merged = merge_small_agentic_chunks(
        overlapped,
        min_tokens=settings.agentic_chunk_min_tokens,
        target_tokens=settings.agentic_chunk_target_tokens,
        max_tokens=MAX_RETRIEVAL_UNIT_TOKENS,
    )
    rebuilt = rebuild_chunks_from_source_blocks(micro_merged, block_by_id)
    split = split_large_retrieval_units(rebuilt, block_by_id, max_tokens=MAX_RETRIEVAL_UNIT_TOKENS)
    atomized = atomize_body_and_visual_chunks(split, block_by_id)
    body_repaired = merge_body_chunks_across_visuals(atomized, block_by_id)
    visual_deduped = dedupe_visual_chunks(body_repaired, block_by_id)
    originals = reindex_retrieval_units(annotate_retrieval_units(visual_deduped))
    fused = append_visual_fusion_chunks(originals, block_by_id)
    reindexed = reindex_fused_chunks_after_originals(fused)
    return enrich_chunks_with_metadata(reindexed, block_by_id)


def atomize_body_and_visual_chunks(chunks: list[Chunk], block_by_id: dict[int, PaperBlock]) -> list[Chunk]:
    if not block_by_id:
        return chunks
    output: list[Chunk] = []
    for chunk in chunks:
        output.extend(atomize_chunk(chunk, block_by_id))
    return sorted(output, key=lambda item: (item.document_id, item.chunk_index, item.chunk_type))


def atomize_chunk(chunk: Chunk, block_by_id: dict[int, PaperBlock]) -> list[Chunk]:
    source_ids = [block_id for block_id in chunk.source_block_ids or [] if block_id in block_by_id]
    if not source_ids:
        return [chunk]

    body_parts: list[tuple[int, str]] = []
    visual_parts: list[tuple[list[int], str, str]] = []
    pending_table_captions: list[tuple[int, str]] = []

    for block_id in sorted(source_ids, key=lambda item: block_order(item, block_by_id)):
        block = block_by_id[block_id]
        content = chunk_content_for_block(block)
        if block.block_type == "section_title":
            body_parts.append((block_id, content))
            continue
        if block.block_type == "figure_caption":
            visual_parts.append(([block_id], content, "figure_caption"))
            continue
        if block.block_type in {"table", "table_caption"}:
            visual_content = content
            visual_source_ids = [block_id]
            if block.block_type == "table" and pending_table_captions:
                visual_source_ids = [caption_id for caption_id, _ in pending_table_captions] + [block_id]
                visual_content = clean_retrieval_content(
                    "\n\n".join(caption for _, caption in pending_table_captions) + "\n\n" + content
                )
                pending_table_captions = []
            visual_parts.append((visual_source_ids, visual_content, "table"))
            continue

        body_text, embedded_visuals = split_embedded_visual_references(content)
        if body_text and has_semantic_content_for_chunk(body_text):
            body_parts.append((block_id, body_text))
        for visual_text, visual_type in embedded_visuals:
            if visual_type == "table":
                pending_table_captions.append((block_id, visual_text))
            else:
                visual_parts.append(([block_id], visual_text, visual_type))

    for block_id, caption in pending_table_captions:
        visual_parts.append(([block_id], caption, "table"))

    if not visual_parts:
        if chunk.chunk_type == "table" and not looks_like_markdown_table(chunk.content) and not is_caption_like(chunk.content):
            return []
        return [chunk]

    atomized: list[Chunk] = []
    body_chunk = build_atomized_body_chunk(chunk, body_parts, block_by_id)
    if body_chunk is not None:
        atomized.append(body_chunk)
    atomized.extend(build_atomized_visual_chunks(chunk, visual_parts, block_by_id))
    return atomized or [chunk]


def split_embedded_visual_references(content: str) -> tuple[str, list[tuple[str, str]]]:
    text = clean_retrieval_content(content)
    if not text:
        return "", []
    embedded: list[tuple[str, str]] = []

    def collect(pattern: str, visual_type: str, value: str) -> str:
        matches = list(re_finditer(pattern, value, ignore_case=True))
        if not matches:
            return value
        keep = value
        for match in reversed(matches):
            start = match.start()
            caption = value[start:].strip()
            prefix = value[:start].strip()
            if not caption:
                continue
            embedded.append((caption, visual_type))
            keep = prefix
        return keep

    if not is_caption_like(text):
        text = collect(r"\bTable\s*\d+[A-Za-z]?\s*[:.]\s*", "table", text)
        text = collect(r"\b(?:Figure|Fig\.)\s*\d+[A-Za-z]?\s*[:.]\s*", "figure_caption", text)
    else:
        visual_type = "table" if text.lower().startswith("table") else "figure_caption"
        return "", [(text, visual_type)]

    return clean_retrieval_content(text), list(reversed(embedded))


def re_finditer(pattern: str, text: str, *, ignore_case: bool = False):
    import re

    flags = re.IGNORECASE if ignore_case else 0
    return re.finditer(pattern, text, flags=flags)


def has_semantic_content_for_chunk(content: str) -> bool:
    text = sanitize_text(content).strip()
    if not text:
        return False
    if is_metadata_noise(text):
        return False
    if estimate_token_count(text) <= 14 and re_match(r"^\d+\s+.+(?:link|github|huggingface|physionet|http)", text, ignore_case=True):
        return False
    return True


def build_atomized_body_chunk(
    template: Chunk,
    body_parts: list[tuple[int, str]],
    block_by_id: dict[int, PaperBlock],
) -> Chunk | None:
    if not body_parts:
        return None
    source_ids = [block_id for block_id, _ in body_parts]
    content = clean_retrieval_content("\n\n".join(part for _, part in body_parts))
    if not content:
        return None
    section_name = resolve_section_name(content, template.section_name, None)
    return replace(
        template,
        chunk_index=min(block_order(block_id, block_by_id) for block_id in source_ids),
        content=content,
        token_count=estimate_token_count(content),
        content_hash=content_hash(
            f"atomized-body:{template.document_id}:{section_name}:{','.join(map(str, source_ids))}:{content}"
        ),
        chunk_type=AGENTIC_CHUNK_TYPE if template.chunking_strategy == "agentic" else template.chunk_type,
        section_name=section_name,
        source_block_ids=source_ids,
        metadata=template.metadata,
    )


def build_atomized_visual_chunks(
    template: Chunk,
    visual_parts: list[tuple[list[int], str, str]],
    block_by_id: dict[int, PaperBlock],
) -> list[Chunk]:
    merged_tables: dict[str, tuple[list[int], list[str]]] = {}
    output: list[Chunk] = []
    for source_ids, content, visual_type in visual_parts:
        clean_content = clean_retrieval_content(content)
        if not clean_content:
            continue
        if visual_type == "table":
            refs = extract_visual_refs(clean_content)
            key = refs[0] if refs else f"table-block-{source_ids[-1]}"
            merged_source_ids, parts = merged_tables.setdefault(key, ([], []))
            merged_source_ids.extend(source_ids)
            parts.append(clean_content)
            continue
        output.append(build_atomized_visual_chunk(template, source_ids, clean_content, visual_type, block_by_id))
    for source_ids, parts in merged_tables.values():
        source_ids = sorted(set(source_ids), key=lambda block_id: block_order(block_id, block_by_id))
        content = clean_retrieval_content("\n\n".join(parts))
        output.append(build_atomized_visual_chunk(template, source_ids, content, "table", block_by_id))
    return output


def build_atomized_visual_chunk(
    template: Chunk,
    source_ids: list[int],
    content: str,
    chunk_type: str,
    block_by_id: dict[int, PaperBlock],
) -> Chunk:
    section_name = block_by_id[source_ids[0]].section_name or template.section_name
    return Chunk(
        document_id=template.document_id,
        chunk_index=min(block_order(block_id, block_by_id) for block_id in source_ids),
        content=content,
        token_count=estimate_token_count(content),
        content_hash=content_hash(
            f"atomized-visual:{template.document_id}:{chunk_type}:{section_name}:{','.join(map(str, source_ids))}:{content}"
        ),
        chunk_type=chunk_type,
        section_name=section_name,
        source_block_ids=source_ids,
        chunking_strategy=template.chunking_strategy,
        metadata=template.metadata,
    )


def rebuild_chunks_from_source_blocks(chunks: list[Chunk], block_by_id: dict[int, PaperBlock]) -> list[Chunk]:
    if not block_by_id:
        return chunks
    rebuilt: list[Chunk] = []
    for chunk in chunks:
        source_ids = [block_id for block_id in chunk.source_block_ids or [] if block_id in block_by_id]
        if not source_ids:
            rebuilt.append(chunk)
            continue
        rebuilt.append(rebuild_chunk_with_source_blocks(chunk, [], block_by_id))
    return rebuilt


def ensure_section_coverage(chunks: list[Chunk], block_by_id: dict[int, PaperBlock]) -> list[Chunk]:
    if not chunks or not block_by_id:
        return chunks
    result = list(chunks)
    for heading in section_heading_blocks(block_by_id):
        if heading.id is None:
            continue
        section_ids = section_source_block_ids(heading, block_by_id)
        missing_ids = [block_id for block_id in section_ids if not block_is_chunked(result, block_id)]
        if not missing_ids:
            continue
        section = heading.section_name or heading.content
        target_index = find_chunk_for_section(result, section, heading.order_index, heading.paper_id, block_by_id)
        if target_index is not None:
            result[target_index] = rebuild_chunk_with_source_blocks(result[target_index], missing_ids, block_by_id)
            continue
        if len(section_ids) <= 1:
            continue
        template = find_template_chunk_for_paper(result, heading.paper_id, block_by_id)
        if template is None:
            continue
        result.append(build_coverage_chunk(template, section_ids, block_by_id))
    return sorted(result, key=lambda chunk: (chunk.document_id, chunk.chunk_index))


def section_heading_blocks(block_by_id: dict[int, PaperBlock]) -> list[PaperBlock]:
    return sorted(
        (
            block
            for block in block_by_id.values()
            if block.id is not None and block.block_type == "section_title" and should_include_source_block(block)
        ),
        key=lambda block: (block.paper_id, block.order_index),
    )


def block_is_chunked(chunks: list[Chunk], block_id: int) -> bool:
    return any(block_id in (chunk.source_block_ids or []) for chunk in chunks)


def find_chunk_for_section(
    chunks: list[Chunk],
    section_name: str,
    heading_order: int,
    paper_id: int,
    block_by_id: dict[int, PaperBlock],
) -> int | None:
    normalized = normalize_section_name(section_name)
    for index, chunk in enumerate(chunks):
        if not chunk_belongs_to_paper(chunk, paper_id, block_by_id):
            continue
        if normalize_section_name(chunk.section_name) == normalized and chunk.chunk_index >= heading_order:
            return index
    for index, chunk in enumerate(chunks):
        if not chunk_belongs_to_paper(chunk, paper_id, block_by_id):
            continue
        if normalize_section_name(chunk.section_name) == normalized:
            return index
    return None


def section_source_block_ids(heading: PaperBlock, block_by_id: dict[int, PaperBlock]) -> list[int]:
    if heading.id is None:
        return []
    section = normalize_section_name(heading.section_name or heading.content)
    output: list[int] = []
    for block in sorted(block_by_id.values(), key=lambda item: (item.paper_id, item.order_index)):
        if block.paper_id != heading.paper_id:
            continue
        if block.order_index < heading.order_index:
            continue
        if block.order_index > heading.order_index and block.block_type == "section_title":
            break
        if block.id is None or not should_include_source_block(block) or is_visual_source_block(block):
            continue
        if normalize_section_name(block.section_name) == section:
            output.append(block.id)
    return output


def is_visual_source_block(block: PaperBlock) -> bool:
    return block.block_type in {"figure", "figure_caption", "table", "table_caption"}


def rebuild_chunk_with_source_blocks(
    template: Chunk,
    extra_source_ids: list[int],
    block_by_id: dict[int, PaperBlock],
) -> Chunk:
    source_ids = same_paper_source_ids(
        list(set(template.source_block_ids or []) | set(extra_source_ids)),
        block_by_id,
    )
    source_ids = sorted(source_ids, key=lambda block_id: block_order(block_id, block_by_id))
    if not source_ids:
        return template
    content = content_from_source_blocks(source_ids, block_by_id) or template.content
    section_name = resolve_section_name(content, template.section_name, None)
    return Chunk(
        document_id=template.document_id,
        chunk_index=min(block_order(block_id, block_by_id) for block_id in source_ids),
        content=content,
        token_count=estimate_token_count(content),
        content_hash=content_hash(
            f"section-covered:{template.document_id}:{section_name}:{','.join(map(str, source_ids))}:{content}"
        ),
        chunk_type=rebuilt_chunk_type(template),
        section_name=section_name,
        source_block_ids=source_ids,
        chunking_strategy=template.chunking_strategy,
    )


def build_coverage_chunk(template: Chunk, source_ids: list[int], block_by_id: dict[int, PaperBlock]) -> Chunk:
    source_ids = same_paper_source_ids(source_ids, block_by_id)
    if not source_ids:
        return template
    content = content_from_source_blocks(source_ids, block_by_id)
    section_name = resolve_section_name(content, block_by_id[source_ids[0]].section_name, None)
    return Chunk(
        document_id=template.document_id,
        chunk_index=min(block_order(block_id, block_by_id) for block_id in source_ids),
        content=content,
        token_count=estimate_token_count(content),
        content_hash=content_hash(
            f"section-coverage:{template.document_id}:{section_name}:{','.join(map(str, source_ids))}:{content}"
        ),
        chunk_type=rebuilt_chunk_type(template),
        section_name=section_name,
        source_block_ids=source_ids,
        chunking_strategy=template.chunking_strategy,
    )


def rebuilt_chunk_type(template: Chunk) -> str:
    if template.chunk_type in {"figure_caption", "table", FUSED_CHUNK_TYPE}:
        return template.chunk_type
    return AGENTIC_CHUNK_TYPE if template.chunking_strategy == "agentic" else template.chunk_type


def merge_overlapping_retrieval_units(
    chunks: list[Chunk],
    block_by_id: dict[int, PaperBlock] | None = None,
    *,
    threshold: float = OVERLAP_MERGE_THRESHOLD,
) -> list[Chunk]:
    if not chunks:
        return []
    block_by_id = block_by_id or {}
    ordered = sorted(chunks, key=lambda chunk: (chunk.document_id, chunk.chunk_index))
    merged: list[Chunk] = []
    pending = ordered[0]
    for chunk in ordered[1:]:
        if should_merge_overlap(pending, chunk, threshold=threshold):
            pending = merge_chunks_by_source_blocks(pending, chunk, block_by_id)
            continue
        merged.append(pending)
        pending = chunk
    merged.append(pending)
    return sorted(merged, key=lambda chunk: (chunk.document_id, chunk.chunk_index))


def should_merge_overlap(left: Chunk, right: Chunk, *, threshold: float) -> bool:
    if left.document_id != right.document_id:
        return False
    if normalize_section_name(left.section_name) != normalize_section_name(right.section_name):
        return False
    return overlap_ratio(left.source_block_ids or [], right.source_block_ids or []) >= threshold


def overlap_ratio(left_ids: list[int], right_ids: list[int]) -> float:
    left = set(left_ids)
    right = set(right_ids)
    if not left or not right:
        return 0.0
    return len(left & right) / min(len(left), len(right))


def merge_chunks_by_source_blocks(left: Chunk, right: Chunk, block_by_id: dict[int, PaperBlock]) -> Chunk:
    source_block_ids = same_paper_source_ids(
        list(set(left.source_block_ids or []) | set(right.source_block_ids or [])),
        block_by_id,
    )
    source_block_ids = sorted(source_block_ids, key=lambda block_id: block_order(block_id, block_by_id))
    content = content_from_source_blocks(source_block_ids, block_by_id) or clean_retrieval_content(f"{left.content}\n\n{right.content}")
    section_name = left.section_name or right.section_name
    chunking_strategy = left.chunking_strategy if left.chunking_strategy == right.chunking_strategy else right.chunking_strategy
    chunk_type = rebuilt_chunk_type(left) if left.chunk_type == right.chunk_type else (AGENTIC_CHUNK_TYPE if chunking_strategy == "agentic" else left.chunk_type)
    return Chunk(
        document_id=left.document_id,
        chunk_index=min(left.chunk_index, right.chunk_index),
        content=content,
        token_count=estimate_token_count(content),
        content_hash=content_hash(
            f"overlap-merged:{left.document_id}:{chunk_type}:{section_name}:{','.join(map(str, source_block_ids))}:{content}"
        ),
        chunk_type=chunk_type,
        section_name=section_name,
        source_block_ids=source_block_ids,
        chunking_strategy=chunking_strategy,
    )


def merge_body_chunks_across_visuals(chunks: list[Chunk], block_by_id: dict[int, PaperBlock]) -> list[Chunk]:
    if not chunks:
        return []
    ordered = sorted(chunks, key=lambda chunk: (chunk.document_id, chunk.chunk_index))
    items = list(ordered)
    index = 0
    while index < len(items):
        left = items[index]
        if not is_body_chunk_for_fusion(left):
            index += 1
            continue
        right_index = find_body_continuation_across_visuals(items, index, block_by_id)
        if right_index is None:
            index += 1
            continue
        right = items[right_index]
        merged = merge_chunks_by_source_blocks(left, right, block_by_id)
        items = [chunk for offset, chunk in enumerate(items) if offset not in {index, right_index}]
        items.insert(index, merged)
        index = max(index - 1, 0)
    return sorted(items, key=lambda chunk: (chunk.document_id, chunk.chunk_index))


def find_body_continuation_across_visuals(
    chunks: list[Chunk],
    index: int,
    block_by_id: dict[int, PaperBlock],
) -> int | None:
    source = chunks[index]
    skipped_visual = False
    for candidate_index in range(index + 1, min(index + 5, len(chunks))):
        candidate = chunks[candidate_index]
        if candidate.document_id != source.document_id:
            return None
        if is_visual_chunk_for_fusion(candidate):
            skipped_visual = True
            continue
        if not is_body_chunk_for_fusion(candidate):
            return None
        if not skipped_visual:
            return None
        if not same_body_continuation_context(source, candidate, block_by_id):
            return None
        if should_merge_body_continuation(source, candidate):
            return candidate_index
        return None
    return None


def same_body_continuation_context(left: Chunk, right: Chunk, block_by_id: dict[int, PaperBlock]) -> bool:
    if normalize_section_name(left.section_name) != normalize_section_name(right.section_name):
        return False
    if paper_id_for_chunk(left, block_by_id) != paper_id_for_chunk(right, block_by_id):
        return False
    return page_distance(left, right, block_by_id) <= 1


def should_merge_body_continuation(left: Chunk, right: Chunk) -> bool:
    left_text = sanitize_text(left.content).rstrip()
    right_text = sanitize_text(right.content).lstrip()
    if ends_with_unfinished_enumeration(left_text):
        return True
    if has_bad_visual_binding_boundary(left_text):
        return True
    return looks_like_sentence_continuation(right_text) and not left_text.endswith((".", "!", "?"))


def dedupe_visual_chunks(chunks: list[Chunk], block_by_id: dict[int, PaperBlock]) -> list[Chunk]:
    ordered = sorted(chunks, key=lambda chunk: (chunk.document_id, chunk.chunk_index))
    best_by_key: dict[tuple[object, ...], tuple[int, Chunk]] = {}
    duplicate_indexes: set[int] = set()
    for index, chunk in enumerate(ordered):
        if not is_visual_chunk_for_fusion(chunk):
            continue
        key = visual_dedupe_key(chunk, block_by_id)
        if key is None:
            continue
        current = best_by_key.get(key)
        if current is None:
            best_by_key[key] = (index, chunk)
            continue
        current_index, current_chunk = current
        if visual_chunk_quality_score(chunk) > visual_chunk_quality_score(current_chunk):
            duplicate_indexes.add(current_index)
            best_by_key[key] = (index, chunk)
        else:
            duplicate_indexes.add(index)
    return [chunk for index, chunk in enumerate(ordered) if index not in duplicate_indexes]


def visual_dedupe_key(chunk: Chunk, block_by_id: dict[int, PaperBlock]) -> tuple[object, ...] | None:
    refs = tuple(extract_visual_refs(chunk.content))
    if not refs:
        return None
    return (
        chunk.document_id,
        paper_id_for_chunk(chunk, block_by_id),
        normalize_section_name(chunk.section_name),
        semantic_chunk_type(chunk),
        refs,
    )


def visual_chunk_quality_score(chunk: Chunk) -> tuple[int, int, int]:
    semantic_type = semantic_chunk_type(chunk)
    confidence_score = 1
    if semantic_type == "table":
        confidence_score = 2 if table_confidence(chunk.content) == "high" else 0
    has_markdown_score = 1 if looks_like_markdown_table(chunk.content) else 0
    return (confidence_score, has_markdown_score, estimate_token_count(chunk.content))


def append_visual_fusion_chunks(chunks: list[Chunk], block_by_id: dict[int, PaperBlock]) -> list[Chunk]:
    ordered = sorted(chunks, key=lambda chunk: (chunk.document_id, chunk.chunk_index))
    body_chunks = [chunk for chunk in ordered if is_body_chunk_for_fusion(chunk)]
    visual_chunks = [chunk for chunk in ordered if is_visual_chunk_for_fusion(chunk)]
    fused: list[Chunk] = []
    for visual in visual_chunks:
        if is_low_confidence_table_chunk(visual):
            continue
        binding = find_visual_fusion_target(visual, body_chunks, block_by_id)
        if binding is None:
            continue
        body, confidence = binding
        fused.append(build_fused_visual_chunk(body, visual, confidence, block_by_id))
    return ordered + dedupe_fused_chunks(fused)


def reindex_fused_chunks_after_originals(chunks: list[Chunk]) -> list[Chunk]:
    originals = [chunk for chunk in chunks if chunk.chunk_type != FUSED_CHUNK_TYPE]
    fused = [chunk for chunk in chunks if chunk.chunk_type == FUSED_CHUNK_TYPE]
    next_index_by_document: dict[int, int] = defaultdict(int)
    for chunk in originals:
        next_index_by_document[chunk.document_id] = max(
            next_index_by_document[chunk.document_id],
            chunk.chunk_index + 1,
        )
    output = list(originals)
    for chunk in sorted(fused, key=lambda item: (item.document_id, item.chunk_index)):
        index = next_index_by_document.get(chunk.document_id, 0)
        next_index_by_document[chunk.document_id] = index + 1
        output.append(
            replace(
                chunk,
                chunk_index=index,
                content_hash=content_hash(f"fused-reindexed:{chunk.document_id}:{index}:{chunk.content_hash}"),
            )
        )
    return sorted(output, key=lambda item: (item.document_id, item.chunk_index))


def dedupe_fused_chunks(chunks: list[Chunk]) -> list[Chunk]:
    unique_by_source_pair: dict[tuple[str, ...], Chunk] = {}
    for chunk in chunks:
        source_ids = tuple((chunk.metadata or {}).get("source_chunk_ids", []))
        if not source_ids:
            source_ids = tuple(sorted(str(block_id) for block_id in (chunk.source_block_ids or [])))
        current = unique_by_source_pair.get(source_ids)
        if current is None or fused_chunk_quality_score(chunk) > fused_chunk_quality_score(current):
            unique_by_source_pair[source_ids] = chunk

    deduped: list[Chunk] = []
    for chunk in sorted(unique_by_source_pair.values(), key=lambda item: (item.document_id, item.chunk_index)):
        if any(too_similar_fused_chunks(chunk, existing) for existing in deduped):
            continue
        deduped.append(chunk)
    return deduped


def fused_chunk_quality_score(chunk: Chunk) -> tuple[int, int]:
    confidence_rank = {"high": 3, "medium": 2, "low": 1}
    confidence = str((chunk.metadata or {}).get("fusion_confidence") or "low")
    return (confidence_rank.get(confidence, 0), estimate_token_count(chunk.content))


def too_similar_fused_chunks(left: Chunk, right: Chunk) -> bool:
    if left.document_id != right.document_id:
        return False
    if normalize_section_name(left.section_name) != normalize_section_name(right.section_name):
        return False
    return duplicate_score(left.content, right.content) >= 0.92


def is_body_chunk_for_fusion(chunk: Chunk) -> bool:
    if is_figure_or_table_chunk(chunk):
        return False
    if chunk.chunk_type in {"figure_caption", "table", FUSED_CHUNK_TYPE}:
        return False
    return True


def is_visual_chunk_for_fusion(chunk: Chunk) -> bool:
    return chunk.chunk_type in {"figure_caption", "table"} or is_figure_or_table_chunk(chunk)


def is_low_confidence_table_chunk(chunk: Chunk) -> bool:
    if semantic_chunk_type(chunk) != "table":
        return False
    confidence = (chunk.metadata or {}).get("table_confidence") or table_confidence(chunk.content)
    return confidence == "low"


def find_visual_fusion_target(
    visual: Chunk,
    body_chunks: list[Chunk],
    block_by_id: dict[int, PaperBlock],
) -> tuple[Chunk, float] | None:
    visual_refs = set(extract_visual_refs(visual.content))
    explicit_candidates = [
        body
        for body in body_chunks
        if body.document_id == visual.document_id and visual_refs & set(extract_visual_refs(body.content))
    ]
    if explicit_candidates:
        same_section_candidates = [
            body
            for body in explicit_candidates
            if normalize_section_name(body.section_name) == normalize_section_name(visual.section_name)
        ]
        if same_section_candidates:
            explicit_candidates = same_section_candidates
        return max(
            ((body, visual_binding_score(body, visual, block_by_id, explicit=True)) for body in explicit_candidates),
            key=lambda item: item[1],
        )

    if is_conclusion_section(visual.section_name):
        return None

    candidates: list[tuple[Chunk, float]] = []
    for body in body_chunks:
        if body.document_id != visual.document_id:
            continue
        if is_conclusion_section(body.section_name):
            continue
        if normalize_section_name(body.section_name) != normalize_section_name(visual.section_name):
            continue
        if paper_id_for_chunk(body, block_by_id) != paper_id_for_chunk(visual, block_by_id):
            continue
        if page_distance(body, visual, block_by_id) > 1:
            continue
        score = visual_binding_score(body, visual, block_by_id, explicit=False)
        if score >= 0.18:
            candidates.append((body, score))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[1])


def visual_binding_score(body: Chunk, visual: Chunk, block_by_id: dict[int, PaperBlock], *, explicit: bool) -> float:
    similarity = semantic_similarity(body.content, visual.content)
    distance = page_distance(body, visual, block_by_id)
    section_bonus = 0.2 if normalize_section_name(body.section_name) == normalize_section_name(visual.section_name) else 0.0
    proximity_bonus = max(0.0, 0.2 - (0.1 * distance))
    explicit_bonus = 0.7 if explicit else 0.0
    return round(min(1.0, similarity + proximity_bonus + section_bonus + explicit_bonus), 3)


def semantic_similarity(left: str, right: str) -> float:
    import re

    stopwords = {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "are",
        "was",
        "were",
        "figure",
        "fig",
        "table",
    }
    left_tokens = {
        token
        for token in re.findall(r"[a-z][a-z0-9_-]{2,}", sanitize_text(left).lower())
        if token not in stopwords
    }
    right_tokens = {
        token
        for token in re.findall(r"[a-z][a-z0-9_-]{2,}", sanitize_text(right).lower())
        if token not in stopwords
    }
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def build_fused_visual_chunk(
    body: Chunk,
    visual: Chunk,
    confidence: float,
    block_by_id: dict[int, PaperBlock],
) -> Chunk:
    visual_refs = sorted(set(extract_visual_refs(body.content)) | set(extract_visual_refs(visual.content)))
    source_block_ids = sorted(set(body.source_block_ids or []) | set(visual.source_block_ids or []))
    content = clean_retrieval_content(
        f"{body.content}\n\nRelated visual/table evidence:\n{visual.content}"
    )
    confidence_label = fusion_confidence_label(confidence)
    metadata = {
        "semantic_chunk_type": FUSED_CHUNK_TYPE,
        "retrieval_only": True,
        "source_chunk_ids": [body.content_hash, visual.content_hash],
        "visual_refs": visual_refs,
        "fusion_confidence": confidence_label,
        "fusion_score": confidence,
        "source_body_block_ids": body.source_block_ids or [],
        "source_visual_block_ids": visual.source_block_ids or [],
        "paper_id": paper_id_for_chunk(body, block_by_id),
        "section": body.section_name,
    }
    return Chunk(
        document_id=body.document_id,
        chunk_index=max(body.chunk_index, visual.chunk_index) + 10_000,
        content=content,
        token_count=estimate_token_count(content),
        content_hash=content_hash(
            f"visual-fused:{body.document_id}:{body.content_hash}:{visual.content_hash}:{confidence_label}:{content}"
        ),
        chunk_type=FUSED_CHUNK_TYPE,
        section_name=body.section_name,
        source_block_ids=source_block_ids,
        chunking_strategy="agentic_fusion",
        metadata=metadata,
    )


def fusion_confidence_label(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.4:
        return "medium"
    return "low"


def paper_id_for_chunk(chunk: Chunk, block_by_id: dict[int, PaperBlock]) -> int | None:
    ids = {
        block_by_id[block_id].paper_id
        for block_id in chunk.source_block_ids or []
        if block_id in block_by_id
    }
    return next(iter(ids)) if len(ids) == 1 else None


def page_distance(left: Chunk, right: Chunk, block_by_id: dict[int, PaperBlock]) -> int:
    left_pages = chunk_pages(left, block_by_id)
    right_pages = chunk_pages(right, block_by_id)
    if not left_pages or not right_pages:
        return 0
    return min(abs(left_page - right_page) for left_page in left_pages for right_page in right_pages)


def chunk_pages(chunk: Chunk, block_by_id: dict[int, PaperBlock]) -> list[int]:
    return sorted(
        {
            block_by_id[block_id].page_number
            for block_id in chunk.source_block_ids or []
            if block_id in block_by_id and block_by_id[block_id].page_number is not None
        }
    )


def is_conclusion_section(section_name: str | None) -> bool:
    return "conclusion" in normalize_section_name(section_name)


def attach_visual_context_to_body(chunks: list[Chunk]) -> list[Chunk]:
    ordered = sorted(chunks, key=lambda chunk: (chunk.document_id, chunk.chunk_index))
    output = list(ordered)
    consumed: set[int] = set()
    for index, chunk in enumerate(ordered):
        if not is_figure_caption_chunk(chunk):
            continue
        target_index = find_visual_context_target(output, index)
        if target_index is None:
            continue
        output[target_index] = append_visual_context(output[target_index], chunk)
        consumed.add(index)
    return [chunk for index, chunk in enumerate(output) if index not in consumed]


def find_visual_context_target(chunks: list[Chunk], visual_index: int) -> int | None:
    visual = chunks[visual_index]
    for candidate_index in range(visual_index - 1, -1, -1):
        candidate = chunks[candidate_index]
        if candidate.document_id != visual.document_id:
            break
        if normalize_section_name(candidate.section_name) != normalize_section_name(visual.section_name):
            continue
        if is_figure_or_table_chunk(candidate):
            continue
        if has_bad_visual_binding_boundary(candidate.content):
            continue
        return candidate_index
    for candidate_index in range(visual_index + 1, len(chunks)):
        candidate = chunks[candidate_index]
        if candidate.document_id != visual.document_id:
            break
        if normalize_section_name(candidate.section_name) != normalize_section_name(visual.section_name):
            continue
        if is_figure_or_table_chunk(candidate):
            continue
        return candidate_index
    return None


def append_visual_context(body: Chunk, visual: Chunk) -> Chunk:
    visual_text = clean_retrieval_content(visual.content)
    content = clean_retrieval_content(f"{body.content}\n\nRelated visual context:\n{visual_text}")
    source_ids = sorted(set(body.source_block_ids or []) | set(visual.source_block_ids or []))
    visual_refs = sorted(set((body.metadata or {}).get("visual_refs", [])) | set(extract_visual_refs(visual_text)))
    metadata = {
        **(body.metadata or {}),
        "semantic_chunk_type": "body_with_visual_context",
        "visual_refs": visual_refs,
    }
    return replace(
        body,
        content=content,
        token_count=estimate_token_count(content),
        content_hash=content_hash(
            f"visual-bound:{body.document_id}:{body.section_name}:{','.join(map(str, source_ids))}:{content}"
        ),
        source_block_ids=source_ids,
        metadata=metadata,
    )


def has_bad_visual_binding_boundary(content: str) -> bool:
    text = sanitize_text(content).rstrip()
    if not text:
        return False
    return bool(
        re_match(r".*(?:\band\s*\(\d+\)|\bsuch as|\bincluding|\bwhere|\bbecause|\blet|[:,;])\s*$", text, ignore_case=True)
        or text.count("(") > text.count(")")
        or text.count("[") > text.count("]")
    )


def is_figure_caption_chunk(chunk: Chunk) -> bool:
    return bool(re_match(r"^(Figure|Fig\.)\s*\d+[:.]", sanitize_text(chunk.content).strip(), ignore_case=True))


def extract_visual_refs(content: str) -> list[str]:
    import re

    refs = re.findall(r"\b(?:Fig\.|Figure|Table)\s*\d+[A-Za-z]?\b", sanitize_text(content), flags=re.IGNORECASE)
    normalized: list[str] = []
    for ref in refs:
        match = re_match(r"^(Fig\.|Figure|Table)\s*(\d+[A-Za-z]?)$", ref.strip(), ignore_case=True)
        if not match:
            continue
        prefix = "Table" if match.group(1).lower().startswith("table") else "Fig."
        normalized.append(f"{prefix} {match.group(2)}")
    return normalized


def enrich_chunks_with_metadata(chunks: list[Chunk], block_by_id: dict[int, PaperBlock]) -> list[Chunk]:
    return [enrich_chunk_with_metadata(chunk, block_by_id) for chunk in chunks]


def enrich_chunk_with_metadata(chunk: Chunk, block_by_id: dict[int, PaperBlock]) -> Chunk:
    blocks = [block_by_id[block_id] for block_id in chunk.source_block_ids or [] if block_id in block_by_id]
    paper_ids = sorted({block.paper_id for block in blocks})
    pages = sorted({block.page_number for block in blocks if block.page_number is not None})
    visual_refs = sorted(set((chunk.metadata or {}).get("visual_refs", [])) | set(extract_visual_refs(chunk.content)))
    semantic_type = semantic_chunk_type(chunk)
    metadata = {
        **(chunk.metadata or {}),
        "paper_id": paper_ids[0] if paper_ids else None,
        "section": chunk.section_name,
        "semantic_chunk_type": semantic_type,
        "page_range": page_range(pages),
        "visual_refs": visual_refs,
        "quality_flags": quality_flags_for_chunk(chunk),
        "has_missing_formula": "<!-- formula-not-decoded" in chunk.content,
    }
    if semantic_type == "table":
        metadata["table_confidence"] = table_confidence(chunk.content)
        metadata["parent_section"] = chunk.section_name
    return replace(chunk, metadata=metadata)


def semantic_chunk_type(chunk: Chunk) -> str:
    if chunk.chunk_type == FUSED_CHUNK_TYPE or (chunk.metadata or {}).get("semantic_chunk_type") == FUSED_CHUNK_TYPE:
        return FUSED_CHUNK_TYPE
    if is_figure_caption_chunk(chunk):
        return "figure_caption"
    if looks_like_markdown_table(chunk.content) or (chunk.chunk_type == "table" and is_caption_like(chunk.content)):
        return "table"
    if (chunk.metadata or {}).get("visual_refs"):
        return "body_with_visual_context"
    return "body"


def page_range(pages: list[int]) -> str | None:
    if not pages:
        return None
    return str(pages[0]) if len(pages) == 1 else f"{pages[0]}-{pages[-1]}"


def table_confidence(content: str) -> str:
    lines = [line for line in sanitize_text(content).splitlines() if "|" in line]
    if len(lines) < 2:
        return "low"
    column_counts = {markdown_table_column_count(line) for line in lines}
    return "high" if len(column_counts) == 1 and any("---" in line for line in lines) else "low"


def markdown_table_column_count(line: str) -> int:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return len(stripped.split("|"))


def quality_flags_for_chunk(chunk: Chunk) -> list[str]:
    flags: list[str] = []
    if has_bad_visual_binding_boundary(chunk.content):
        flags.append("bad_sentence_boundary")
    if ends_with_unfinished_enumeration(chunk.content):
        flags.append("unfinished_enumeration")
    if caption_inside_sentence(chunk.content):
        flags.append("caption_inside_sentence")
    if "<!-- formula-not-decoded" in chunk.content:
        flags.append("missing_formula")
    if semantic_chunk_type(chunk) == "table" and table_confidence(chunk.content) == "low":
        flags.append("low_confidence_table")
    return flags


def caption_inside_sentence(content: str) -> bool:
    import re

    return bool(re.search(r"[a-z,;:]\s*\n+(?:Figure|Fig\.|Table)\s*\d+", sanitize_text(content)))


def print_chunk_quality_report(chunks: list[Chunk]) -> None:
    report = build_chunk_quality_report(chunks)
    safe_print(
        "Chunk quality report: "
        f"total_chunks={report['total_chunks']} "
        f"bad_boundary_chunks={report['bad_boundary_chunks']} "
        f"chunks_with_caption_inside_sentence={report['chunks_with_caption_inside_sentence']} "
        f"duplicate_neighbor_chunks={report['duplicate_neighbor_chunks']} "
        f"low_confidence_tables={report['low_confidence_tables']} "
        f"chunks_with_missing_formula={report['chunks_with_missing_formula']} "
        f"average_chunk_words={report['average_chunk_words']} "
        f"max_chunk_words={report['max_chunk_words']} "
        f"min_chunk_words={report['min_chunk_words']}"
    )
    for item in report["bad_chunks"]:
        safe_print(
            "Bad chunk: "
            f"paper_id={item['paper_id']} "
            f"chunk_id={item['chunk_id']} "
            f"section={item['section']} "
            f"reason={item['reason']} "
            f"preview={item['preview']}"
        )
    print_grouped_chunk_debug_report(chunks)


def print_grouped_chunk_debug_report(chunks: list[Chunk]) -> None:
    grouped = {
        "BODY CHUNKS": [chunk for chunk in chunks if debug_chunk_group(chunk) == "body"],
        "VISUAL CHUNKS": [chunk for chunk in chunks if debug_chunk_group(chunk) == "visual"],
        "FUSED CHUNKS": [chunk for chunk in chunks if debug_chunk_group(chunk) == "fused"],
    }
    for heading, items in grouped.items():
        safe_print(heading)
        for chunk in sorted(items, key=lambda item: (item.document_id, item.chunk_index)):
            metadata = chunk.metadata or {}
            refs = metadata.get("visual_refs") or []
            confidence = metadata.get("fusion_confidence")
            extras = []
            if refs:
                extras.append(f"visual_refs={refs}")
            if confidence:
                extras.append(f"fusion_confidence={confidence}")
            extra_text = f" {' '.join(extras)}" if extras else ""
            preview = sanitize_text(chunk.content).replace("\n", " ")[:180]
            safe_print(
                f"- document_id={chunk.document_id} chunk_index={chunk.chunk_index} "
                f"type={chunk.chunk_type} section={chunk.section_name}{extra_text} preview={preview}"
            )


def safe_print(value: str) -> None:
    try:
        print(value)
    except UnicodeEncodeError:
        import sys

        encoding = sys.stdout.encoding or "utf-8"
        print(value.encode(encoding, errors="replace").decode(encoding, errors="replace"))


def debug_chunk_group(chunk: Chunk) -> str:
    if chunk.chunk_type == FUSED_CHUNK_TYPE or (chunk.metadata or {}).get("semantic_chunk_type") == FUSED_CHUNK_TYPE:
        return "fused"
    if chunk.chunk_type in {"figure_caption", "table"} or semantic_chunk_type(chunk) in {"figure_caption", "table"}:
        return "visual"
    return "body"


def build_chunk_quality_report(chunks: list[Chunk]) -> dict[str, object]:
    word_counts = [len(sanitize_text(chunk.content).split()) for chunk in chunks]
    bad_chunks: list[dict[str, object]] = []
    duplicate_neighbor_chunks = count_duplicate_neighbor_chunks(chunks)
    for chunk in chunks:
        flags = list((chunk.metadata or {}).get("quality_flags", []))
        if flags:
            bad_chunks.append(
                {
                    "paper_id": (chunk.metadata or {}).get("paper_id"),
                    "chunk_id": chunk.chunk_index,
                    "section": chunk.section_name,
                    "reason": ",".join(flags),
                    "preview": sanitize_text(chunk.content).replace("\n", " ")[:180],
                }
            )
    return {
        "total_chunks": len(chunks),
        "bad_boundary_chunks": sum("bad_sentence_boundary" in (chunk.metadata or {}).get("quality_flags", []) for chunk in chunks),
        "chunks_with_caption_inside_sentence": sum("caption_inside_sentence" in (chunk.metadata or {}).get("quality_flags", []) for chunk in chunks),
        "duplicate_neighbor_chunks": duplicate_neighbor_chunks,
        "low_confidence_tables": sum((chunk.metadata or {}).get("table_confidence") == "low" for chunk in chunks),
        "chunks_with_missing_formula": sum(bool((chunk.metadata or {}).get("has_missing_formula")) for chunk in chunks),
        "average_chunk_words": round(sum(word_counts) / len(word_counts), 2) if word_counts else 0,
        "max_chunk_words": max(word_counts, default=0),
        "min_chunk_words": min(word_counts, default=0),
        "bad_chunks": bad_chunks[:20],
    }


def count_duplicate_neighbor_chunks(chunks: list[Chunk]) -> int:
    ordered = sorted(chunks, key=lambda chunk: (chunk.document_id, chunk.chunk_index))
    duplicates = 0
    for left, right in zip(ordered, ordered[1:], strict=False):
        if left.document_id != right.document_id:
            continue
        if normalize_section_name(left.section_name) == normalize_section_name(right.section_name):
            if duplicate_score(left.content, right.content) >= 0.82 and has_long_text_containment(left.content, right.content):
                duplicates += 1
            continue
        if same_or_adjacent_section_name(left.section_name, right.section_name):
            if duplicate_score(left.content, right.content) >= 0.82 and has_long_text_containment(left.content, right.content):
                duplicates += 1
    return duplicates


def duplicate_score(left: str, right: str) -> float:
    left_tokens = set(sanitize_text(left).lower().split())
    right_tokens = set(sanitize_text(right).lower().split())
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
    import re

    text = re.sub(r"^##\s*\d+(?:\.\d+)*\s+.*$", " ", sanitize_text(content), flags=re.MULTILINE)
    return re.sub(r"\s+", " ", text).strip().lower()


def has_enough_unique_tokens(content: str) -> bool:
    import re

    tokens = {token for token in re.findall(r"[a-z][a-z0-9_-]{2,}", content.lower())}
    return len(tokens) >= 8


def same_or_adjacent_section_name(left: str | None, right: str | None) -> bool:
    if normalize_section_name(left) == normalize_section_name(right):
        return True
    left_number = section_number_key(left)
    right_number = section_number_key(right)
    if not left_number or not right_number:
        return False
    if left_number[:-1] != right_number[:-1]:
        return False
    return abs(left_number[-1] - right_number[-1]) == 1


def section_number_key(section_name: str | None) -> tuple[int, ...] | None:
    match = re_match(r"^\s*(\d+(?:\.\d+)*)\b", sanitize_text(section_name or ""))
    if not match:
        return None
    return tuple(int(part) for part in match.group(1).split("."))


def split_large_retrieval_units(
    chunks: list[Chunk],
    block_by_id: dict[int, PaperBlock] | None = None,
    *,
    max_tokens: int = MAX_RETRIEVAL_UNIT_TOKENS,
) -> list[Chunk]:
    block_by_id = block_by_id or {}
    output: list[Chunk] = []
    for chunk in chunks:
        if chunk.token_count <= max_tokens:
            output.append(chunk)
            continue
        output.extend(split_large_chunk(chunk, block_by_id, max_tokens=max_tokens))
    return sorted(output, key=lambda item: (item.document_id, item.chunk_index))


def split_large_chunk(chunk: Chunk, block_by_id: dict[int, PaperBlock], *, max_tokens: int) -> list[Chunk]:
    source_ids = [block_id for block_id in chunk.source_block_ids or [] if block_id in block_by_id]
    if source_ids:
        return split_large_chunk_by_blocks(chunk, source_ids, block_by_id, max_tokens=max_tokens)
    return split_large_chunk_by_paragraphs(chunk, max_tokens=max_tokens)


def split_large_chunk_by_blocks(
    chunk: Chunk,
    source_ids: list[int],
    block_by_id: dict[int, PaperBlock],
    *,
    max_tokens: int,
) -> list[Chunk]:
    ordered_ids = [
        block_id
        for block_id in sorted(source_ids, key=lambda block_id: block_order(block_id, block_by_id))
        if block_id in block_by_id and should_include_source_block(block_by_id[block_id])
    ]
    if not ordered_ids:
        return []
    groups: list[list[int]] = []
    current: list[int] = []
    current_tokens = 0
    for block_id in ordered_ids:
        block_content = chunk_content_for_block(block_by_id[block_id])
        block_tokens = estimate_token_count(block_content)
        if current and current_tokens + block_tokens > max_tokens and not source_ids_end_with_unfinished_enumeration(current, block_by_id):
            groups.append(current)
            current = []
            current_tokens = 0
        current.append(block_id)
        current_tokens += block_tokens
    if current:
        groups.append(current)
    return [
        rebuild_split_chunk(chunk, group, content_from_source_blocks(group, block_by_id), index)
        for index, group in enumerate(groups)
    ]


def source_ids_end_with_unfinished_enumeration(source_ids: list[int], block_by_id: dict[int, PaperBlock]) -> bool:
    content = content_from_source_blocks(source_ids, block_by_id)
    return ends_with_unfinished_enumeration(content)


def ends_with_unfinished_enumeration(content: str) -> bool:
    text = sanitize_text(content).rstrip()
    return bool(re_match(r".*\b(?:and|or)\s*\(\d+\)\s*$", text, ignore_case=True))


def split_large_chunk_by_paragraphs(chunk: Chunk, *, max_tokens: int) -> list[Chunk]:
    paragraphs = [paragraph.strip() for paragraph in chunk.content.split("\n\n") if paragraph.strip()]
    if len(paragraphs) <= 1:
        return [chunk]
    groups: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0
    for paragraph in paragraphs:
        paragraph_tokens = estimate_token_count(paragraph)
        if current and current_tokens + paragraph_tokens > max_tokens:
            groups.append(current)
            current = []
            current_tokens = 0
        current.append(paragraph)
        current_tokens += paragraph_tokens
    if current:
        groups.append(current)
    return [
        rebuild_split_chunk(chunk, chunk.source_block_ids or [], clean_retrieval_content("\n\n".join(group)), index)
        for index, group in enumerate(groups)
    ]


def rebuild_split_chunk(template: Chunk, source_block_ids: list[int], content: str, split_index: int) -> Chunk:
    clean_content = clean_retrieval_content(content)
    return Chunk(
        document_id=template.document_id,
        chunk_index=template.chunk_index + split_index,
        content=clean_content,
        token_count=estimate_token_count(clean_content),
        content_hash=content_hash(
            f"retrieval-split:{template.document_id}:{template.section_name}:{split_index}:{','.join(map(str, source_block_ids))}:{clean_content}"
        ),
        chunk_type=rebuilt_chunk_type(template),
        section_name=template.section_name,
        source_block_ids=source_block_ids,
        chunking_strategy=template.chunking_strategy,
        metadata=template.metadata,
    )


def content_from_source_blocks(source_block_ids: list[int], block_by_id: dict[int, PaperBlock]) -> str:
    source_block_ids = same_paper_source_ids(source_block_ids, block_by_id)
    parts = [
        chunk_content_for_block(block_by_id[block_id])
        for block_id in source_block_ids
        if block_id in block_by_id and should_include_source_block(block_by_id[block_id])
    ]
    return clean_retrieval_content("\n\n".join(parts))


def same_paper_blocks(blocks: list[PaperBlock]) -> bool:
    paper_ids = {block.paper_id for block in blocks}
    return len(paper_ids) <= 1


def same_paper_source_ids(source_block_ids: list[int], block_by_id: dict[int, PaperBlock]) -> list[int]:
    known_ids = [block_id for block_id in source_block_ids if block_id in block_by_id]
    if not known_ids:
        return []
    first_paper_id = block_by_id[known_ids[0]].paper_id
    return [block_id for block_id in known_ids if block_by_id[block_id].paper_id == first_paper_id]


def chunk_belongs_to_paper(chunk: Chunk, paper_id: int, block_by_id: dict[int, PaperBlock]) -> bool:
    source_ids = [block_id for block_id in chunk.source_block_ids or [] if block_id in block_by_id]
    if not source_ids:
        return False
    return all(block_by_id[block_id].paper_id == paper_id for block_id in source_ids)


def find_template_chunk_for_paper(
    chunks: list[Chunk],
    paper_id: int,
    block_by_id: dict[int, PaperBlock],
) -> Chunk | None:
    for chunk in chunks:
        if chunk_belongs_to_paper(chunk, paper_id, block_by_id):
            return chunk
    return None


def should_include_source_block(block: PaperBlock) -> bool:
    if not block.should_embed:
        return False
    if block.block_type == "table":
        return True
    return not is_metadata_noise(block.markdown_content or block.content, block.section_name, block.order_index)


def block_order(block_id: int, block_by_id: dict[int, PaperBlock]) -> int:
    block = block_by_id.get(block_id)
    return block.order_index if block else block_id


def annotate_retrieval_units(chunks: list[Chunk]) -> list[Chunk]:
    return [annotate_retrieval_unit(chunk) for chunk in chunks]


def reindex_retrieval_units(chunks: list[Chunk]) -> list[Chunk]:
    ordered = sorted(chunks, key=lambda chunk: (chunk.document_id, chunk.chunk_index))
    counters: dict[int, int] = {}
    output: list[Chunk] = []
    for chunk in ordered:
        index = counters.get(chunk.document_id, 0)
        counters[chunk.document_id] = index + 1
        output.append(
            Chunk(
                document_id=chunk.document_id,
                chunk_index=index,
                content=chunk.content,
                token_count=chunk.token_count,
                content_hash=content_hash(
                    f"retrieval-reindexed:{chunk.document_id}:{index}:{chunk.content_hash}"
                ),
                chunk_type=chunk.chunk_type,
                section_name=chunk.section_name,
                source_block_ids=chunk.source_block_ids,
                chunking_strategy=chunk.chunking_strategy,
                retrieval_value=chunk.retrieval_value,
                query_intents=chunk.query_intents,
                keywords=chunk.keywords,
                planner_reason=chunk.planner_reason,
                metadata=chunk.metadata,
            )
        )
    return output


def annotate_retrieval_unit(chunk: Chunk) -> Chunk:
    if not ENABLE_CHUNK_METADATA:
        return strip_chunk_metadata(chunk)
    retrieval_value = infer_retrieval_value(chunk)
    query_intents = infer_query_intents(chunk)
    return Chunk(
        document_id=chunk.document_id,
        chunk_index=chunk.chunk_index,
        content=chunk.content,
        token_count=chunk.token_count,
        content_hash=content_hash(
            f"retrieval-annotated:{chunk.document_id}:{chunk.chunk_index}:{retrieval_value}:{','.join(query_intents)}:{chunk.content_hash}"
        ),
        chunk_type=chunk.chunk_type,
        section_name=chunk.section_name,
        source_block_ids=chunk.source_block_ids,
        chunking_strategy=chunk.chunking_strategy,
        retrieval_value=retrieval_value,
        query_intents=query_intents,
        keywords=chunk.keywords,
        planner_reason=chunk.planner_reason,
        metadata=chunk.metadata,
    )


def strip_chunk_metadata(chunk: Chunk) -> Chunk:
    return Chunk(
        document_id=chunk.document_id,
        chunk_index=chunk.chunk_index,
        content=chunk.content,
        token_count=chunk.token_count,
        content_hash=content_hash(f"retrieval-metadata-disabled:{chunk.document_id}:{chunk.chunk_index}:{chunk.content_hash}"),
        chunk_type=chunk.chunk_type,
        section_name=chunk.section_name,
        source_block_ids=chunk.source_block_ids,
        chunking_strategy=chunk.chunking_strategy,
        retrieval_value=None,
        query_intents=[],
        keywords=[],
        planner_reason=None,
        metadata=chunk.metadata,
    )


def infer_retrieval_value(chunk: Chunk) -> str:
    text = sanitize_text(f"{chunk.section_name or ''}\n{chunk.content}").lower()
    section = sanitize_text(chunk.section_name or "").strip().lower()
    if section == "abstract" or "introduction" in section:
        return "low"
    high_terms = (
        "performance",
        "accuracy",
        "result",
        "benchmark",
        "comparison",
        "sharpe",
        "return",
        "ratio",
        "score",
        "quantitative",
        "outperform",
        "table",
    )
    medium_terms = (
        "method",
        "model construction",
        "training",
        "factor construction",
        "feature",
        "calculate factor",
        "methodology",
    )
    low_terms = ("abstract", "introduction", "background", "motivation")
    if any(term in text for term in high_terms) and has_quantitative_signal(text):
        return "high"
    if any(term in text for term in high_terms) and any(term in text for term in ("conclusion", "best", "model performance")):
        return "high"
    if any(term in text for term in medium_terms):
        return "medium"
    if any(term in text for term in low_terms):
        return "low"
    return "medium"


def has_quantitative_signal(text: str) -> bool:
    return bool(re_match(r".*\b\d+(?:\.\d+)?%?\b.*", text)) or "|" in text


def infer_query_intents(chunk: Chunk) -> list[str]:
    text = sanitize_text(f"{chunk.section_name or ''}\n{chunk.content}").lower()
    intents: list[str] = []
    if "momentum" in text:
        intents.extend(["best momentum factor", "momentum return", "momentum sharpe ratio"])
    if "reversion" in text:
        intents.extend(["best reversion factor", "reversion return", "reversion sharpe ratio"])
    if "performance" in text or "accuracy" in text or "score" in text:
        intents.extend(["model performance", "classification accuracy", "prediction score"])
    if "factor" in text and not intents:
        intents.extend(["factor construction", "factor return", "factor comparison"])
    if "ratio" in text or "valuation" in text:
        intents.extend(["valuation ratio", "financial analysis", "sector comparison"])
    if "table" in text or "|" in text:
        intents.append("quantitative comparison")
    if not intents:
        section = sanitize_text(chunk.section_name or "").strip().lower()
        intents.append(section or "paper retrieval")
    return dedupe_preserve_order(intents)


def dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def merge_or_drop_section_titles(chunks: list[Chunk]) -> ChunkCleanupResult:
    items = list(chunks)
    consumed: set[int] = set()
    dropped = 0
    merged = 0
    for index, chunk in enumerate(items):
        if index in consumed:
            continue
        if should_drop_metadata_chunk(chunk):
            consumed.add(index)
            dropped += 1
            continue
        if not is_standalone_section_heading_chunk(chunk):
            continue
        target_index = find_next_chunk_index(items, index, consumed)
        consumed.add(index)
        if target_index is None:
            dropped += 1
            continue
        items[target_index] = merge_two_chunks(chunk, items[target_index])
        merged += 1
    return ChunkCleanupResult(
        chunks=[chunk for index, chunk in enumerate(items) if index not in consumed],
        dropped_tiny_chunks=dropped,
        merged_tiny_chunks=merged,
    )


def merge_or_drop_tiny_chunks(chunks: list[Chunk]) -> ChunkCleanupResult:
    items = list(chunks)
    consumed: set[int] = set()
    output: list[Chunk] = []
    dropped = 0
    merged = 0

    for index, chunk in enumerate(items):
        if index in consumed:
            continue
        if should_drop_metadata_chunk(chunk):
            dropped += 1
            continue
        if is_tiny_chunk(chunk):
            if output and can_merge_cleanup_chunks(output[-1], chunk):
                output[-1] = merge_two_chunks(output[-1], chunk)
                merged += 1
                continue
            target_index = find_next_chunk_index(items, index, consumed)
            if target_index is not None:
                items[target_index] = merge_two_chunks(chunk, items[target_index])
                consumed.add(index)
                merged += 1
                continue
            dropped += 1
            continue
        output.append(chunk)

    return ChunkCleanupResult(chunks=output, dropped_tiny_chunks=dropped, merged_tiny_chunks=merged)


def find_next_chunk_index(chunks: list[Chunk], index: int, consumed: set[int]) -> int | None:
    source = chunks[index]
    for candidate_index in range(index + 1, len(chunks)):
        if candidate_index in consumed:
            continue
        candidate = chunks[candidate_index]
        if should_drop_metadata_chunk(candidate):
            continue
        if is_standalone_section_heading_chunk(candidate):
            continue
        if can_merge_cleanup_chunks(source, candidate):
            return candidate_index
    return None


def can_merge_cleanup_chunks(left: Chunk, right: Chunk) -> bool:
    if left.document_id != right.document_id:
        return False
    if is_figure_or_table_chunk(left) or is_figure_or_table_chunk(right):
        return False
    left_section = normalize_section_name(left.section_name)
    right_section = normalize_section_name(right.section_name)
    if left_section == right_section:
        return True
    return (
        left_section == "unknown" and right_section in {"abstract", "introduction"}
    ) or (
        right_section == "unknown" and left_section in {"abstract", "introduction"}
    )


def normalize_section_name(section_name: str | None) -> str:
    return (section_name or "unknown").strip().lower()


def is_tiny_chunk(chunk: Chunk) -> bool:
    if chunk.token_count >= 50:
        return False
    return not looks_like_substantial_short_content(chunk.content)


def looks_like_substantial_short_content(content: str) -> bool:
    text = sanitize_text(content).strip()
    if not text:
        return False
    lowered = text.lower()
    if has_numbered_enumeration_fragment(text) or looks_like_sentence_continuation(text):
        return True
    if lowered.startswith("## "):
        return estimate_token_count(text) >= 5
    if re_match(r"^Table\s+\d+\s+reports\b", text, ignore_case=True):
        return True
    if "<!-- formula-not-decoded" in lowered:
        return True
    if is_caption_like(text):
        return True
    if lowered.startswith("## abstract") or lowered.startswith("abstract\n"):
        return estimate_token_count(text) >= 20
    if "```" in text:
        return estimate_token_count(text) >= 20
    if "|" in text and "\n" in text:
        return looks_like_markdown_table(text)
    return False


def has_numbered_enumeration_fragment(content: str) -> bool:
    text = sanitize_text(content)
    return bool(re_match(r".*\(\d+\).*\(\d+\)", text, ignore_case=True))


def looks_like_sentence_continuation(content: str) -> bool:
    text = sanitize_text(content).strip()
    if estimate_token_count(text) < 6:
        return False
    return bool(re_match(r"^(?:[a-z][a-z-]*|where|because|including|such as)\b", text, ignore_case=True))


def should_drop_metadata_chunk(chunk: Chunk) -> bool:
    section_name = (chunk.section_name or "").lower()
    if section_name == "unknown" and is_metadata_noise(chunk.content, chunk.section_name, chunk.chunk_index):
        return True
    return False


def is_standalone_section_heading_chunk(chunk: Chunk) -> bool:
    return is_section_heading_text(chunk.content)


def looks_like_metadata(content: str) -> bool:
    import re

    text = sanitize_text(content).strip()
    lowered = text.lower()
    if not text:
        return True
    if "@" in text:
        return True
    if any(term in lowered for term in ("university", "department of", "school of", "institute", "affiliation")):
        return True
    if re.fullmatch(r"(19|20)\d{2}([-/]\d{1,2}){0,2}", text):
        return True
    if len(text.split()) <= 12 and re.search(r"\b(author|authors|copyright|license)\b", lowered):
        return True
    return False


def merge_two_chunks(left: Chunk, right: Chunk) -> Chunk:
    content = clean_retrieval_content(f"{left.content}\n\n{right.content}")
    source_block_ids = sorted(set(left.source_block_ids or []) | set(right.source_block_ids or []))
    chunking_strategy = left.chunking_strategy if left.chunking_strategy == right.chunking_strategy else right.chunking_strategy
    chunk_type = AGENTIC_CHUNK_TYPE if chunking_strategy == "agentic" else (left.chunk_type if left.chunk_type == right.chunk_type else "other")
    return Chunk(
        document_id=left.document_id,
        chunk_index=min(left.chunk_index, right.chunk_index),
        content=content,
        token_count=estimate_token_count(content),
        content_hash=content_hash(
            f"agentic-merged:{left.document_id}:{chunk_type}:{left.section_name}:{','.join(map(str, source_block_ids))}:{content}"
        ),
        chunk_type=chunk_type,
        section_name=left.section_name,
        source_block_ids=source_block_ids,
        chunking_strategy=chunking_strategy,
    )


def planner_stats(
    chunks: list[Chunk],
    *,
    before_merge_count: int,
    cleanup: ChunkCleanupResult | None = None,
) -> dict[str, object]:
    cleanup = cleanup or ChunkCleanupResult(chunks=chunks)
    return {
        "before_merge_chunks": before_merge_count,
        "after_merge_chunks": len(chunks),
        "skipped": 0,
        "avg_chunk_tokens": avg_chunk_tokens(chunks),
        "dropped_tiny_chunks": cleanup.dropped_tiny_chunks,
        "merged_tiny_chunks": cleanup.merged_tiny_chunks,
        "final_avg_chunk_tokens": avg_chunk_tokens(chunks),
        "min_chunk_tokens": min((chunk.token_count for chunk in chunks), default=0),
    }


def avg_chunk_tokens(chunks: list[Chunk]) -> float:
    if not chunks:
        return 0.0
    return round(sum(chunk.token_count for chunk in chunks) / len(chunks), 2)


def build_repair_prompt(raw_output: str) -> str:
    return (
        "Fix this into valid JSON matching this schema. Return JSON only.\n"
        '{"chunks":[{"chunk_type":"abstract|background|method|experiment|table|figure_caption|code|conclusion|other","section_name":"string","source_block_ids":[1],"should_embed":true}]}\n\n'
        f"Raw output:\n{raw_output}"
    )
