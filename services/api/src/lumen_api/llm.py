"""Resolve an `LLMProvider` from this org's settings.

Its own module rather than living in `agents/runner.py` (where it started):
`services/worker`'s jobs (`sentinel.py`, `glossary.py`) and `impact.py` all
need it too, and none of them should have to import `agents/runner.py` —
which itself imports `agents/registry.py` — just to build a provider.
`impact.py` importing `runner.py` while `registry.py` imports `impact.py`
is exactly the cycle that moving this here avoids.
"""

from __future__ import annotations

from lumen.llm.registry import ModelTiers, Tier, get_provider
from lumen_api.settings import get_settings


def provider(tier: Tier = "reasoning"):
    settings = get_settings()
    return get_provider(
        tier,
        anthropic_key=(
            settings.anthropic_api_key.get_secret_value() if settings.has_anthropic else None
        ),
        groq_key=(settings.groq_api_key.get_secret_value() if settings.has_groq else None),
        tiers=ModelTiers(
            reasoning=settings.model_reasoning,
            specialist=settings.model_specialist,
            fast=settings.model_fast,
        ),
        mode=settings.llm_mode,
        bridge_inbox=settings.llm_bridge_inbox,
    )
