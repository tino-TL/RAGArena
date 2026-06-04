from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.agent import router as agent_router
from app.api.v1.ask import router as ask_router
from app.api.v1.feedback import router as feedback_router
from app.api.v1.health import router as health_router
from app.api.v1.papers import router as papers_router
from app.api.v1.search import router as search_router
from app.api.v1.stream import router as stream_router

router = APIRouter(prefix="/api/v1")
router.include_router(health_router)
router.include_router(papers_router)
router.include_router(search_router)
router.include_router(ask_router)
router.include_router(stream_router)
router.include_router(agent_router)
router.include_router(feedback_router)
