"""The endpoints an operator hits first: is it alive, and is it configured?

These run with no Supabase project, no Redis and no API key — everything here
must answer from settings alone, which is exactly the property that lets a new
contributor verify their `.env` before anything else is running.
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from lumen_api.main import create_app


def client() -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=create_app()), base_url="http://test")


async def test_healthz_never_touches_a_dependency():
    async with client() as http:
        response = await http.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_openapi_is_served():
    async with client() as http:
        response = await http.get("/openapi.json")
    assert response.status_code == 200
    assert response.json()["info"]["title"] == "Lumen API"


async def test_config_reports_the_keyless_fallback_when_no_key_is_set():
    async with client() as http:
        response = await http.get("/v1/config")

    assert response.status_code == 200
    body = response.json()
    assert body["llm"]["mode"] == "auto"
    assert body["embeddings"]["dimensions"] == 384


async def test_config_never_echoes_a_secret():
    """A config endpoint that leaks a key is worse than no config endpoint."""
    async with client() as http:
        response = await http.get("/v1/config")

    text = response.text.lower()
    for forbidden in ("sk-ant", "gsk_", "service_role", "jwt_secret", "password"):
        assert forbidden not in text


# ── Settings ────────────────────────────────────────────────────────────────


def test_auto_resolves_to_mock_without_credentials(make_settings):
    settings = make_settings(environment="test", anthropic_api_key=None, groq_api_key=None)
    assert settings.resolved_llm_mode == "mock"
    assert settings.has_anthropic is False


def test_auto_resolves_to_anthropic_once_a_key_is_present(make_settings):
    settings = make_settings(environment="test", anthropic_api_key="sk-ant-real-value")
    assert settings.resolved_llm_mode == "anthropic"
    assert settings.has_anthropic is True


def test_auto_falls_through_to_groq_when_only_that_is_configured(make_settings):
    settings = make_settings(environment="test", anthropic_api_key=None, groq_api_key="gsk_real")
    assert settings.resolved_llm_mode == "groq"


def test_a_placeholder_is_not_a_credential(make_settings):
    """Copying .env.example and forgetting to fill it in must not read as configured."""
    settings = make_settings(
        environment="test",
        supabase_url="https://YOUR-PROJECT-REF.supabase.co",
        anthropic_api_key="changeme",
    )
    assert settings.has_anthropic is False
    assert settings.resolved_llm_mode == "mock"


def test_production_refuses_to_start_with_placeholders(make_settings):
    with pytest.raises(ValueError) as excinfo:
        make_settings(
            environment="prod",
            supabase_url="https://YOUR-PROJECT-REF.supabase.co",
            supabase_anon_key="",
            supabase_service_role_key="",
            supabase_jwt_secret="",
            database_url="postgresql+asyncpg://postgres:YOUR-DB-PASSWORD@db.YOUR-PROJECT-REF.supabase.co:5432/postgres",
        )

    message = str(excinfo.value)
    for expected in ("SUPABASE_URL", "SUPABASE_ANON_KEY", "SUPABASE_JWT_SECRET", "DATABASE_URL"):
        assert expected in message


def test_production_starts_when_supabase_is_configured_even_with_no_llm_key(make_settings):
    """No API key is not a misconfiguration — it selects the keyless provider."""
    settings = make_settings(
        environment="prod",
        supabase_url="https://abcdefgh.supabase.co",
        supabase_anon_key="anon-real",
        supabase_service_role_key="service-real",
        supabase_jwt_secret="jwt-real",
        database_url="postgresql+asyncpg://postgres:realpass@db.abcdefgh.supabase.co:5432/postgres",
    )
    assert settings.resolved_llm_mode == "mock"


def test_production_rejects_an_explicit_provider_without_its_key(make_settings):
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        make_settings(
            environment="prod",
            llm_mode="anthropic",
            supabase_url="https://abcdefgh.supabase.co",
            supabase_anon_key="anon-real",
            supabase_service_role_key="service-real",
            supabase_jwt_secret="jwt-real",
            database_url="postgresql+asyncpg://postgres:realpass@db.abcdefgh.supabase.co:5432/postgres",
        )


def test_a_trailing_comment_is_not_a_credential(make_settings):
    """Regression: python-dotenv folds `KEY=   # note` into the value.

    Before this check, copying .env.example and leaving the key blank reported
    "anthropic key set", selected the Anthropic provider, and failed at the
    first API call with a 401 — sending the operator to debug their Anthropic
    account instead of their .env.
    """
    settings = make_settings(
        environment="test",
        anthropic_api_key="# console.anthropic.com → API keys (sk-ant-...)",
        groq_api_key="# console.groq.com → API keys",
    )
    assert settings.has_anthropic is False
    assert settings.has_groq is False
    assert settings.resolved_llm_mode == "mock"


def test_a_value_that_is_not_shaped_like_a_key_is_not_a_key(make_settings):
    settings = make_settings(environment="test", anthropic_api_key="my-anthropic-key")
    assert settings.has_anthropic is False

    settings = make_settings(environment="test", groq_api_key="not-a-groq-key")
    assert settings.has_groq is False


def test_correctly_shaped_keys_are_accepted(make_settings):
    settings = make_settings(
        environment="test",
        anthropic_api_key="sk-ant-api03-abcdef",
        groq_api_key="gsk_abcdef",
    )
    assert settings.has_anthropic is True
    assert settings.has_groq is True
    assert settings.resolved_llm_mode == "anthropic"


def test_the_shipped_env_example_reads_as_unconfigured(make_settings):
    """Whatever the file says, a fresh copy must select the keyless path."""
    from pathlib import Path

    example = Path(__file__).resolve().parents[3] / ".env.example"
    assert example.exists(), ".env.example must ship with the repository"

    settings = make_settings(environment="test", _env_file=str(example))
    assert settings.has_anthropic is False
    assert settings.has_groq is False
    assert settings.resolved_llm_mode == "mock"


def test_sync_dsn_strips_the_async_driver(make_settings):
    settings = make_settings(
        environment="test",
        database_url="postgresql+asyncpg://u:p@h:5432/db",
    )
    assert settings.sync_database_url == "postgresql://u:p@h:5432/db"
