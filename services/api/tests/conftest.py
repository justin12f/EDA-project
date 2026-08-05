"""Test isolation.

A settings test that reads the developer's `.env` is not a test — it passes or
fails depending on whose machine it runs on. Everything here exists to make the
unit suite hermetic: no ambient dotenv, no ambient environment variables.
"""

from __future__ import annotations

import os

import pytest
import pytest_asyncio

from lumen_api.settings import Settings, get_settings

# Anything that could change what Settings resolves to.
_LEAKY_VARS = (
    "ENVIRONMENT",
    "SUPABASE_URL",
    "SUPABASE_ANON_KEY",
    "SUPABASE_KEY",
    "SUPABASE_SERVICE_ROLE_KEY",
    "SUPABASE_JWT_SECRET",
    "DATABASE_URL",
    "LLM_MODE",
    "ANTHROPIC_API_KEY",
    "GROQ_API_KEY",
    "API_KEY_groq",
    "EMBEDDING_PROVIDER",
    "REDIS_URL",
)


@pytest.fixture(autouse=True)
def _isolate_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _LEAKY_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ENVIRONMENT", "test")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def make_settings():
    """Build Settings from explicit values only — never from a file on disk."""

    def _factory(**overrides) -> Settings:
        overrides.setdefault("environment", "test")
        # A test may point at a specific file on purpose (the .env.example
        # check); everything else gets no file at all.
        env_file = overrides.pop("_env_file", None)
        return Settings(_env_file=env_file, **overrides)

    return _factory


@pytest_asyncio.fixture(autouse=True)
async def _dispose_engines_between_tests():
    """An asyncpg pool belongs to the loop that opened it.

    pytest-asyncio gives each test a fresh loop, so a pool left over from the
    previous one yields connections bound to a closed loop — the failure mode
    where every test passes alone and half fail together.
    """
    from lumen_api.db.session import dispose_engines

    yield
    await dispose_engines()
