from __future__ import annotations

from ragarena.chunking.repository import DocumentRecord
from ragarena.cli.rebuild_paper import chunk_fixed_documents


def test_rebuild_fixed_chunks_use_fixed_strategy() -> None:
    chunks = chunk_fixed_documents(
        [
            DocumentRecord(
                id=1,
                title="Paper",
                source="source",
                content="## Abstract\n\n" + "retrieval " * 120,
                content_hash="hash",
            )
        ]
    )

    assert chunks
    assert all(chunk.chunking_strategy == "fixed" for chunk in chunks)
    assert all(chunk.chunk_type == "fixed" for chunk in chunks)
    assert chunks[0].section_name == "Abstract"
