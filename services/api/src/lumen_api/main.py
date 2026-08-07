"""FastAPI application factory."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from lumen_api import (
    api_keys,
    baselines,
    certification,
    global_patterns,
    health,
    me,
    proposals,
    public,
    runs,
    schedules,
    sources,
    trust,
    usage,
)
from lumen_api.errors import register_error_handlers
from lumen_api.settings import get_settings

logger = logging.getLogger("lumen")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    resolved = settings.resolved_llm_mode

    logger.info(
        "lumen api starting — environment=%s llm=%s embeddings=%s",
        settings.environment,
        resolved,
        settings.embedding_provider,
    )
    if resolved == "mock":
        logger.warning(
            "No LLM credential configured. Running the deterministic MockProvider: "
            "agents will profile real data and propose real pipelines, but the "
            "wording is generated locally. Set ANTHROPIC_API_KEY to use Claude."
        )

    yield

    from lumen_api.db.session import dispose_engines

    await dispose_engines()


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Lumen API",
        version="0.1.0",
        description="Agentic data cleaning, pipelines and EDA.",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_error_handlers(app)
    app.include_router(health.router)
    app.include_router(me.router)
    app.include_router(usage.router)
    app.include_router(sources.router)
    app.include_router(runs.router)
    app.include_router(proposals.router)
    app.include_router(schedules.router)
    app.include_router(certification.router)
    app.include_router(api_keys.router)
    app.include_router(public.router)
    app.include_router(trust.router)
    app.include_router(global_patterns.router)
    app.include_router(baselines.router)

    return app


app = create_app()
