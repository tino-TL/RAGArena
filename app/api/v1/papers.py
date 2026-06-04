from __future__ import annotations

from fastapi import APIRouter, Depends

from app.dependencies import RequestContext, get_request_context, latency_ms, raise_api_error
from app.schemas import PaperFetchRequest, PaperFetchResponse, PaperResponse, PapersResponse
from ragarena.config import settings
from ragarena.papers.arxiv_client import fetch_arxiv_papers
from ragarena.papers.repository import ensure_papers_table, fetch_papers
from ragarena.pipeline.steps import store_papers

router = APIRouter(tags=["papers"])


@router.get("/papers", response_model=PapersResponse)
async def list_papers(
    limit: int = 20,
    context: RequestContext = Depends(get_request_context),
) -> PapersResponse:
    try:
        await ensure_papers_table(settings.postgres_dsn)
        papers = await fetch_papers(settings.postgres_dsn, limit=limit)
    except Exception as exc:
        raise_api_error(
            code="papers_list_failed",
            message=str(exc),
            context=context,
        )

    return PapersResponse(
        request_id=context.request_id,
        latency_ms=latency_ms(context),
        papers=[serialize_paper(paper) for paper in papers],
    )


@router.post("/papers/fetch", response_model=PaperFetchResponse)
async def fetch_papers_endpoint(
    request: PaperFetchRequest,
    context: RequestContext = Depends(get_request_context),
) -> PaperFetchResponse:
    try:
        papers = fetch_arxiv_papers(request.query, max_results=request.max_results)
        stored = await store_papers(papers, sync_documents=request.sync_documents)
    except Exception as exc:
        raise_api_error(
            code="papers_fetch_failed",
            message=str(exc),
            context=context,
        )

    return PaperFetchResponse(
        request_id=context.request_id,
        latency_ms=latency_ms(context),
        fetched=len(papers),
        stored=stored,
        synced_documents=request.sync_documents,
    )


def serialize_paper(paper) -> PaperResponse:
    return PaperResponse(
        id=paper.id,
        arxiv_id=paper.arxiv_id,
        title=paper.title,
        authors=paper.authors,
        abstract=paper.abstract,
        categories=paper.categories,
        published_at=paper.published_at.isoformat() if paper.published_at else None,
        updated_at=paper.updated_at.isoformat() if paper.updated_at else None,
        pdf_url=paper.pdf_url,
        source_url=paper.source_url,
    )
