"""Liveness, readiness, and one endpoint that answers "is this thing configured".

`/healthz` must never touch a dependency — it answers "is the process alive",
which is the only question a restart policy should act on.

`/readyz` checks the dependencies a request actually needs.

`/v1/config` is the one an operator reads after pasting their keys in: it says
which LLM provider is live, whether embeddings are available, and whether
Supabase is reachable — without ever echoing a secret.
"""

from __future__ import annotations

from fastapi import APIRouter
from redis.asyncio import Redis

from lumen_api.db.session import ping
from lumen_api.errors import Misconfigured
from lumen_api.settings import get_settings

router = APIRouter(tags=["health"])


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/readyz")
async def readyz() -> dict[str, str]:
    settings = get_settings()

    try:
        await ping()
    except Exception as exc:  # noqa: BLE001 — the operator needs the cause, not a stack
        raise Misconfigured(f"Database unreachable: {type(exc).__name__}") from exc

    redis = Redis.from_url(settings.redis_url)
    try:
        await redis.ping()
    except Exception as exc:  # noqa: BLE001
        raise Misconfigured(f"Redis unreachable: {type(exc).__name__}") from exc
    finally:
        await redis.aclose()

    return {"status": "ok"}


@router.get("/v1/config")
async def config() -> dict[str, object]:
    """What this deployment is actually running. No secret is ever included."""
    settings = get_settings()
    return {
        "environment": settings.environment,
        "llm": {
            "mode": settings.llm_mode,
            "resolved": settings.resolved_llm_mode,
            "anthropic_configured": settings.has_anthropic,
            "groq_configured": settings.has_groq,
            "models": {
                "reasoning": settings.model_reasoning,
                "specialist": settings.model_specialist,
                "fast": settings.model_fast,
            },
            # The one thing an operator most wants to know after adding a key.
            "keyless_fallback_active": settings.resolved_llm_mode == "mock",
        },
        "embeddings": {
            "provider": settings.embedding_provider,
            "model": settings.embedding_model,
            "dimensions": settings.embedding_dimensions,
        },
        "supabase": {
            "url": settings.supabase_url,
            "configured": bool(settings.supabase_anon_key.get_secret_value()),
        },
        "storage_bucket": settings.storage_bucket,
    }
