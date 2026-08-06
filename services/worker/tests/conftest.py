"""Test isolation — the same rules as `services/api`'s own conftest.

`lumen_worker` has no Settings class of its own; every job in it resolves
configuration through `lumen_api.settings.get_settings()`, so the hermetic
rules that module's tests need apply here unchanged.
"""

from __future__ import annotations

import pytest
import pytest_asyncio

from lumen_api.settings import get_settings

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


@pytest_asyncio.fixture(autouse=True)
async def _dispose_engines_between_tests():
    """See `services/api/tests/conftest.py` — same asyncpg-pool-per-loop reason."""
    from lumen_api.db.session import dispose_engines

    yield
    await dispose_engines()
