from __future__ import annotations

from ragarena.chunking.block_chunker import chunk_block_documents
from ragarena.chunking.repository import DocumentRecord
from ragarena.papers.models import PaperBlock


def test_block_chunker_outputs_source_block_ids() -> None:
    document = DocumentRecord(
        id=10,
        title="Paper",
        source="https://arxiv.org/abs/2401.00001v1",
        content="arXiv ID: 2401.00001v1\nPaper content",
        content_hash="hash",
    )
    block = PaperBlock(
        id=42,
        paper_id=1,
        arxiv_id="2401.00001v1",
        block_type="abstract",
        section_name="Abstract",
        page_number=1,
        content="This paper studies agentic retrieval.",
        markdown_content="This paper studies agentic retrieval.",
        image_path=None,
        order_index=0,
        should_embed=True,
        metadata={},
        content_hash="block-hash",
    )

    chunks = chunk_block_documents([document], [block])

    assert len(chunks) == 1
    assert chunks[0].source_block_ids == [42]
    assert chunks[0].chunk_type == "abstract"
    assert chunks[0].chunking_strategy == "block"


def test_figure_caption_can_enter_chunk() -> None:
    document = DocumentRecord(
        id=10,
        title="Paper",
        source="https://arxiv.org/abs/2401.00001v1",
        content="arXiv ID: 2401.00001v1\nPaper content",
        content_hash="hash",
    )

    block = PaperBlock(
        id=42,
        paper_id=1,
        arxiv_id="2401.00001v1",
        block_type="figure_caption",
        section_name="Results",
        page_number=2,
        content="Figure 1: Retrieval architecture.",
        markdown_content="Figure 1: Retrieval architecture.",
        image_path=None,
        order_index=3,
        should_embed=True,
        metadata={},
        content_hash="block-hash",
    )

    chunks = chunk_block_documents([document], [block])

    assert chunks[0].chunk_type == "figure_caption"
    assert chunks[0].source_block_ids == [42]
