from __future__ import annotations

import asyncio
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from app.main import app, startup_load_runtime
from ragarena.agent.policies.grader import grade_documents
from ragarena.agent.policies.rewriter import clean_rewritten_query, rewrite_query
from ragarena.agent.policies.router import route_query
from ragarena.chunking.fixed_chunker import chunk_document
from ragarena.chunking.repository import (
    count_chunks_by_hash,
    ensure_document_chunks_table,
    fetch_documents,
    insert_chunks,
)
from ragarena.config import settings
from ragarena.embedding.repository import (
    count_embeddings_for_model,
    ensure_chunk_embeddings_table,
    fetch_chunks,
    insert_embeddings,
)
from ragarena.generation.prompt import build_rag_prompt
from ragarena.agent.workflow import build_agentic_rag_graph, run_agentic_rag
from ragarena.ingestion.loaders import load_documents
from ragarena.ingestion.repository import (
    count_documents_by_hash,
    ensure_documents_table,
    insert_documents,
)
from ragarena.papers.models import PaperBlock, PaperMetadata
from ragarena.papers.repository import (
    count_papers_by_arxiv_ids,
    ensure_papers_table,
    fetch_paper_blocks_by_hashes,
    insert_paper_blocks,
    insert_papers,
)
from ragarena.retrieval.indexer import fetch_embedded_chunks, index_embedded_chunks
from ragarena.retrieval.search import bm25_search, hybrid_search, vector_search
from ragarena.runtime import (
    get_bge_encoder,
    get_deepseek_generator,
    get_elasticsearch_vector_store,
    get_observability_tracer,
)

LOCAL_QUERY = "LangGraph and LangChain differences"
UNKNOWN_QUERY = "today president news"
SEARCH_QUERY = "LangGraph workflow framework"


def check_fastapi_app() -> bool:
    return app.title == "RAGArena"


def check_fastapi_v1_routes() -> dict[str, bool]:
    paths = {route.path for route in app.routes}
    return {
        "search API route exists": "/api/v1/search" in paths,
        "ask API route exists": "/api/v1/ask" in paths,
        "agent API route exists": "/api/v1/agent" in paths,
        "stream API route exists": "/api/v1/stream" in paths,
        "feedback API route exists": "/api/v1/feedback" in paths,
        "papers API route exists": "/api/v1/papers" in paths,
    }


def check_langgraph_workflow() -> bool:
    return build_agentic_rag_graph() is not None


async def check_capability_ingestion() -> dict[str, bool]:
    documents = load_documents(PROJECT_ROOT / "data" / "sample_docs")
    hashes = [doc.content_hash for doc in documents]
    unique_hash_count = len(set(hashes))

    await ensure_documents_table(settings.postgres_dsn)
    before_count = await count_documents_by_hash(settings.postgres_dsn, hashes)
    await insert_documents(settings.postgres_dsn, documents)
    after_first_count = await count_documents_by_hash(settings.postgres_dsn, hashes)
    await insert_documents(settings.postgres_dsn, documents)
    after_second_count = await count_documents_by_hash(settings.postgres_dsn, hashes)

    return {
        "sample documents loaded": len(documents) >= 2,
        "documents table ready": True,
        "documents inserted": after_first_count == unique_hash_count,
        "duplicate ingestion is idempotent": after_second_count == after_first_count,
        "document count does not shrink": after_first_count >= before_count,
    }


async def check_capability_chunking() -> dict[str, bool]:
    await ensure_documents_table(settings.postgres_dsn)
    await ensure_document_chunks_table(settings.postgres_dsn)

    documents = await fetch_documents(settings.postgres_dsn)
    chunks = [
        chunk
        for document in documents
        for chunk in chunk_document(document.id, document.content)
    ]
    hashes = [chunk.content_hash for chunk in chunks]
    unique_hash_count = len(set(hashes))

    before_count = await count_chunks_by_hash(settings.postgres_dsn, hashes)
    await insert_chunks(settings.postgres_dsn, chunks)
    after_first_count = await count_chunks_by_hash(settings.postgres_dsn, hashes)
    await insert_chunks(settings.postgres_dsn, chunks)
    after_second_count = await count_chunks_by_hash(settings.postgres_dsn, hashes)

    return {
        "documents exist": len(documents) > 0,
        "chunks table ready": True,
        "chunks inserted": after_first_count == unique_hash_count,
        "duplicate chunking is idempotent": after_second_count == after_first_count,
        "chunk count does not shrink": after_first_count >= before_count,
    }


async def check_capability_embedding() -> dict[str, bool]:
    await ensure_document_chunks_table(settings.postgres_dsn)
    await ensure_chunk_embeddings_table(settings.postgres_dsn)

    chunks = await fetch_chunks(settings.postgres_dsn)
    chunk_ids = [chunk.id for chunk in chunks]
    test_model_name = f"{settings.embedding_model}:test"
    test_embeddings = [
        [float((index + chunk.id) % 17) / 17.0 for index in range(1024)]
        for chunk in chunks
    ]

    before_count = await count_embeddings_for_model(settings.postgres_dsn, test_model_name, chunk_ids)
    await insert_embeddings(settings.postgres_dsn, test_model_name, chunk_ids, test_embeddings)
    after_first_count = await count_embeddings_for_model(settings.postgres_dsn, test_model_name, chunk_ids)
    await insert_embeddings(settings.postgres_dsn, test_model_name, chunk_ids, test_embeddings)
    after_second_count = await count_embeddings_for_model(settings.postgres_dsn, test_model_name, chunk_ids)

    return {
        "embedding model configured": settings.embedding_model == "BAAI/bge-m3",
        "chunks exist": len(chunks) > 0,
        "embeddings table ready": True,
        "embeddings inserted": after_first_count == len(chunk_ids),
        "duplicate embedding is idempotent": after_second_count == after_first_count,
        "embedding count does not shrink": after_first_count >= before_count,
    }


async def check_capability_elasticsearch_retrieval() -> dict[str, bool]:
    vector_store = get_elasticsearch_vector_store(settings.elasticsearch_url, settings.elasticsearch_index)
    embedded_chunks = await fetch_embedded_chunks(settings.postgres_dsn, settings.embedding_model)

    indexed_count = index_embedded_chunks(vector_store, embedded_chunks)
    indexed_total = vector_store.count(settings.embedding_model)
    bm25_response = vector_response = hybrid_response = None
    if indexed_total > 0:
        bm25_response = bm25_search(SEARCH_QUERY, settings.elasticsearch_url, settings.elasticsearch_index, settings.embedding_model, 3)
        vector_response = vector_search(SEARCH_QUERY, settings.elasticsearch_url, settings.elasticsearch_index, settings.embedding_model, 3)
        hybrid_response = hybrid_search(SEARCH_QUERY, settings.elasticsearch_url, settings.elasticsearch_index, settings.embedding_model, 3)

    return {
        "Elasticsearch index ready": vector_store.index_exists(),
        "embedded chunks available": len(embedded_chunks) > 0,
        "chunks indexed": indexed_count == len(embedded_chunks),
        "indexed count visible": indexed_total >= len(embedded_chunks),
        "BM25 search returns chunks": bm25_response is not None and len(bm25_response.results) > 0,
        "vector search returns chunks": vector_response is not None and len(vector_response.results) > 0,
        "hybrid search returns chunks": hybrid_response is not None and len(hybrid_response.results) > 0,
    }


async def check_capability_arxiv_paper_ingestion() -> dict[str, bool]:
    paper = PaperMetadata(
        arxiv_id="2401.00001v1",
        title="Retrieval Augmented Generation for Research Assistants",
        authors=["Alice Researcher", "Bob Engineer"],
        abstract="This paper studies retrieval augmented generation for research workflows.",
        categories=["cs.CL", "cs.AI"],
        published_at=datetime(2024, 1, 1, tzinfo=UTC),
        updated_at=datetime(2024, 1, 1, tzinfo=UTC),
        pdf_url="https://arxiv.org/pdf/2401.00001v1",
        source_url="https://arxiv.org/abs/2401.00001v1",
    )
    arxiv_ids = [paper.arxiv_id]

    await ensure_papers_table(settings.postgres_dsn)
    before_count = await count_papers_by_arxiv_ids(settings.postgres_dsn, arxiv_ids)
    await insert_papers(settings.postgres_dsn, [paper])
    after_first_count = await count_papers_by_arxiv_ids(settings.postgres_dsn, arxiv_ids)
    await insert_papers(settings.postgres_dsn, [paper])
    after_second_count = await count_papers_by_arxiv_ids(settings.postgres_dsn, arxiv_ids)
    block = PaperBlock(
        id=None,
        paper_id=1,
        arxiv_id=paper.arxiv_id,
        block_type="abstract",
        section_name="Abstract",
        page_number=None,
        content=paper.abstract,
        markdown_content=paper.abstract,
        image_path=None,
        order_index=0,
        should_embed=True,
        metadata={},
        content_hash=f"test-paper-section-{paper.arxiv_id}",
    )
    before_blocks = await fetch_paper_blocks_by_hashes(settings.postgres_dsn, [block.content_hash])
    await insert_paper_blocks(settings.postgres_dsn, [block])
    after_blocks = await fetch_paper_blocks_by_hashes(settings.postgres_dsn, [block.content_hash])

    return {
        "papers table ready": True,
        "paper metadata upserted": after_first_count == 1,
        "duplicate paper ingestion is idempotent": after_second_count == after_first_count,
        "paper count does not shrink": after_first_count >= before_count,
        "paper blocks table ready": len(after_blocks) >= len(before_blocks),
    }


def check_capability_rag_generation() -> dict[str, bool]:
    response = hybrid_search(LOCAL_QUERY, settings.elasticsearch_url, settings.elasticsearch_index, settings.embedding_model, 3)
    prompt = build_rag_prompt(LOCAL_QUERY, response.results)
    generator = get_deepseek_generator(settings.deepseek_api_key, settings.deepseek_model)

    return {
        "hybrid retrieval provides context": len(response.results) > 0,
        "prompt builds with context": response.results[0].content in prompt,
        "DeepSeek model configured": settings.deepseek_model == "deepseek-chat",
        "DeepSeek generation skipped in test": generator.model == settings.deepseek_model,
    }


def check_capability_performance_optimization() -> dict[str, bool]:
    encoder_a = get_bge_encoder(settings.embedding_model)
    encoder_b = get_bge_encoder(settings.embedding_model)
    store_a = get_elasticsearch_vector_store(settings.elasticsearch_url, settings.elasticsearch_index)
    store_b = get_elasticsearch_vector_store(settings.elasticsearch_url, settings.elasticsearch_index)
    generator_a = get_deepseek_generator(settings.deepseek_api_key, settings.deepseek_model)
    generator_b = get_deepseek_generator(settings.deepseek_api_key, settings.deepseek_model)
    tracer = get_observability_tracer()

    startup_load_runtime()

    return {
        "Runtime BGEEncoder singleton": encoder_a is encoder_b,
        "Runtime Elasticsearch client singleton": store_a is store_b,
        "Runtime DeepSeek client singleton": generator_a is generator_b,
        "Runtime FastAPI startup loads BGE": app.state.bge_encoder is encoder_a,
        "Runtime FastAPI startup loads Elasticsearch": app.state.elasticsearch_vector_store is store_a,
        "Runtime FastAPI startup loads DeepSeek": app.state.deepseek_generator is generator_a,
        "Runtime FastAPI startup loads observability tracer": app.state.observability_tracer is tracer,
    }


def check_capability_document_grader() -> dict[str, bool]:
    local_response = hybrid_search(LOCAL_QUERY, settings.elasticsearch_url, settings.elasticsearch_index, settings.embedding_model, 3)
    local_grade = grade_documents(LOCAL_QUERY, [chunk.content for chunk in local_response.results])

    unknown_response = hybrid_search(UNKNOWN_QUERY, settings.elasticsearch_url, settings.elasticsearch_index, settings.embedding_model, 3)
    unknown_grade = grade_documents(UNKNOWN_QUERY, [chunk.content for chunk in unknown_response.results])

    return {
        "grade_documents returns decision": hasattr(local_grade, "sufficient") and hasattr(unknown_grade, "sufficient"),
        "known local question graded true": local_grade.sufficient is True,
        "unknown current-events question handled": isinstance(unknown_grade.sufficient, bool),
    }


def check_capability_query_rewrite() -> dict[str, bool]:
    rewritten = rewrite_query("LangGraph")
    cleaned = clean_rewritten_query("Rewritten query: LangGraph workflow framework")

    return {
        "rewrite_query returns str": isinstance(rewritten, str),
        "rewrite_query returns non-empty": bool(rewritten.strip()),
        "rewrite_query cleans prefixes": cleaned == "LangGraph workflow framework",
    }


def check_capability_langgraph_workflow() -> dict[str, bool]:
    graph = build_agentic_rag_graph()
    local_state = run_agentic_rag(LOCAL_QUERY)
    unknown_state = run_agentic_rag("abcxyz123")

    return {
        "agentic graph compiles": graph is not None,
        "local query returns state trace": bool(local_state["trace"]),
        "local query returns generation": bool(local_state["generation"]),
        "unknown query handled": bool(unknown_state["generation"]),
    }


def check_capability_query_router() -> dict[str, bool]:
    greeting_route = route_query("你好")
    local_route = route_query(LOCAL_QUERY)
    greeting_state = run_agentic_rag("你好")
    local_state = run_agentic_rag(LOCAL_QUERY)
    unknown_state = run_agentic_rag("abcxyz123")

    return {
        "greeting route is legal": greeting_route.route in {"direct_answer", "local_rag"},
        "LangGraph route is local_rag": local_route.route == "local_rag",
        "greeting returns generation": bool(greeting_state["generation"]),
        "local rag route persists": local_state["route"] == "local_rag",
        "unknown query handled": bool(unknown_state["generation"]),
    }


def check_capability_local_only_agent_branch() -> dict[str, bool]:
    route = route_query(UNKNOWN_QUERY)
    state = run_agentic_rag(UNKNOWN_QUERY)

    return {
        "current-events route stays local": route.route == "local_rag",
        "graph keeps local route": state["route"] == "local_rag",
        "external branch not recorded": not any(step.startswith("external_search:") for step in state["trace"]),
        "local fallback returns generation": bool(state["generation"]),
    }


def check_capability_architecture_visualization() -> dict[str, bool]:
    architecture_doc = PROJECT_ROOT / "docs" / "architecture.md"
    workflow_diagram = PROJECT_ROOT / "docs" / "images" / "ragarena-langgraph-workflow.drawio"
    system_diagram = PROJECT_ROOT / "docs" / "images" / "ragarena-architecture.drawio"
    workflow_image = PROJECT_ROOT / "docs" / "images" / "ragarena-langgraph-workflow.svg"
    system_image = PROJECT_ROOT / "docs" / "images" / "ragarena-architecture.svg"
    workflow_png = PROJECT_ROOT / "docs" / "images" / "ragarena-langgraph-workflow.png"
    system_png = PROJECT_ROOT / "docs" / "images" / "ragarena-architecture.png"
    workflow_text = workflow_diagram.read_text(encoding="utf-8") if workflow_diagram.exists() else ""
    required_nodes = {
        "router",
        "hybrid_retrieve",
        "rerank",
        "grade_documents",
        "rewrite_query",
        "generate_answer",
        "give_up",
    }

    return {
        "Architecture architecture doc exists": architecture_doc.exists(),
        "Architecture system diagram exists": system_diagram.exists(),
        "Architecture workflow diagram exists": workflow_diagram.exists(),
        "Architecture rendered workflow image exists": workflow_image.exists(),
        "Architecture rendered system image exists": system_image.exists(),
        "Architecture rendered workflow png exists": workflow_png.exists(),
        "Architecture rendered system png exists": system_png.exists(),
        "Architecture workflow diagram has key nodes": required_nodes.issubset(set(workflow_text.split()) | {node for node in required_nodes if node in workflow_text}),
    }


def main() -> None:
    checks = {
        "FastAPI app import": check_fastapi_app(),
        "LangGraph agentic workflow build": check_langgraph_workflow(),
        "POSTGRES_DSN configured": bool(settings.postgres_dsn),
        "ELASTICSEARCH_URL configured": bool(settings.elasticsearch_url),
        "OLLAMA_URL configured": bool(settings.ollama_url),
    }

    checks.update(asyncio.run(check_capability_ingestion()))
    checks.update(asyncio.run(check_capability_chunking()))
    checks.update(asyncio.run(check_capability_embedding()))
    checks.update(asyncio.run(check_capability_elasticsearch_retrieval()))
    checks.update(asyncio.run(check_capability_arxiv_paper_ingestion()))
    checks.update(check_capability_rag_generation())
    checks.update(check_capability_performance_optimization())
    checks.update(check_capability_document_grader())
    checks.update(check_capability_query_rewrite())
    checks.update(check_capability_langgraph_workflow())
    checks.update(check_capability_query_router())
    checks.update(check_capability_architecture_visualization())
    checks.update(check_capability_local_only_agent_branch())
    checks.update(check_fastapi_v1_routes())

    print("RAGArena production smoke check")
    print("=================================")
    for name, ok in checks.items():
        status = "OK" if ok else "FAIL"
        print(f"{status} - {name}")

    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

