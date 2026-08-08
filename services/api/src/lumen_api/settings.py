"""Typed application settings.

Every environment variable the API and the worker read is declared here, once.
Two rules this module exists to enforce:

1. **Secrets are `SecretStr`.** They never render in a log line, a traceback or
   a `repr()` by accident — you have to call `.get_secret_value()`, which is a
   visible act at the call site.
2. **Production fails fast.** `ENVIRONMENT=prod` with a placeholder or missing
   credential raises at import, not on the first request that needs it. A
   half-configured deployment that serves 500s is worse than one that refuses
   to boot.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Environment = Literal["dev", "test", "prod"]
LLMMode = Literal["auto", "anthropic", "groq", "mock", "bridge"]
EmbeddingProviderName = Literal["fastembed", "none"]

_REPO_ROOT = Path(__file__).resolve().parents[4]

# A value matching any of these is a placeholder somebody forgot to replace, not
# a setting. Case-insensitive, and deliberately broad: `your_supabase_url_here`,
# `YOUR-PROJECT-REF` and `changeme` are all the same mistake.
_PLACEHOLDER_RE = re.compile(
    r"your[-_ ]|[-_]here\b|YOUR-PROJECT-REF|YOUR-DB-PASSWORD|changeme|<[^>]+>|\bxxx+\b",
    re.IGNORECASE,
)

# Prefixes each provider's keys actually carry. A value that does not start with
# one is not a key, whatever it is — most often it is a trailing comment that
# python-dotenv folded into the value, which would otherwise send the app down
# the "credential configured" path with a comment as the credential.
_KEY_PREFIXES = {
    "anthropic": ("sk-ant-",),
    "groq": ("gsk_",),
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(_REPO_ROOT / ".env", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        # A field with a validation_alias would otherwise reject its own name as
        # a constructor argument — which silently ignored `Settings(groq_api_key=…)`
        # instead of failing, so tests passed values that never landed.
        populate_by_name=True,
    )

    environment: Environment = "dev"

    # ── Supabase ────────────────────────────────────────────────────────────
    #
    # The alias lists accept the names this project used before the rename, so
    # an existing .env keeps working through the migration instead of silently
    # reading as unconfigured.
    supabase_url: str = "http://127.0.0.1:54321"
    supabase_anon_key: SecretStr = Field(
        default=SecretStr(""),
        validation_alias=AliasChoices("SUPABASE_ANON_KEY", "SUPABASE_KEY"),
    )
    supabase_service_role_key: SecretStr = SecretStr("")
    # Empty is a valid, and increasingly the normal, configuration: projects using
    # asymmetric JWT signing keys (ES256/RS256) publish a JWKS instead of sharing a
    # secret, and there is no secret to paste. See `jwt_verification`.
    supabase_jwt_secret: SecretStr = SecretStr("")
    storage_bucket: str = "lumen"

    database_url: str = "postgresql+asyncpg://postgres:postgres@127.0.0.1:54322/postgres"

    # ── Tenant database (ADR-0024) ────────────────────────────────────────────
    #
    # ADR-0024: customer data lives on its own Postgres instance, never in
    # the Supabase database that holds organizations, api_keys and
    # subscriptions. Unset means the Architect is disabled, which is the
    # correct state for a checkout that has not provisioned one.
    tenant_database_url: SecretStr | None = None
    # Above this row count a table is read through DuckDB rather than
    # polars (spec §3.7). A compute choice only — results are identical
    # either side, which a parity test enforces.
    duckdb_row_threshold: int = 5_000_000

    @property
    def has_tenant_db(self) -> bool:
        return self.tenant_database_url is not None

    # ── LLM ─────────────────────────────────────────────────────────────────
    llm_mode: LLMMode = "auto"
    anthropic_api_key: SecretStr | None = None
    groq_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("GROQ_API_KEY", "API_KEY_groq"),
    )

    model_reasoning: str = "claude-opus-5"
    model_specialist: str = "claude-sonnet-5"
    model_fast: str = "qwen/qwen3.6-27b"

    llm_bridge_inbox: str = ".llm-bridge"

    agent_max_iterations: int = 12
    agent_deadline_seconds: float = 180.0
    agent_max_total_tokens: int = 120_000

    # ── Billing (ADR-0004) ──────────────────────────────────────────────────
    #
    # Absent by default, same posture as the LLM keys: a plan-change proposal
    # still validates and records a decision with no key configured, it just
    # cannot hand back a real Checkout URL — the accept path says so plainly
    # rather than failing or pretending to redirect somewhere real.
    stripe_secret_key: SecretStr | None = None
    stripe_webhook_secret: SecretStr | None = None
    app_base_url: str = "http://localhost:3000"

    @property
    def has_stripe(self) -> bool:
        raw = _raw(self.stripe_secret_key)
        return bool(raw) and _is_real(raw) and raw.startswith("sk_")

    # ── Embeddings ──────────────────────────────────────────────────────────
    embedding_provider: EmbeddingProviderName = "fastembed"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dimensions: int = 384
    embedding_cache_dir: str = ".cache/embeddings"

    # ── Queue ───────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"

    # ── HTTP ────────────────────────────────────────────────────────────────
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])

    # ── Derived ─────────────────────────────────────────────────────────────

    @property
    def has_anthropic(self) -> bool:
        return _is_key(self.anthropic_api_key, "anthropic")

    @property
    def has_groq(self) -> bool:
        return _is_key(self.groq_api_key, "groq")

    @property
    def resolved_llm_mode(self) -> LLMMode:
        """What `auto` actually means right now.

        Exposed so `/healthz` and the startup banner can tell the operator which
        provider is live without them having to reason about the fallback rules.
        """
        if self.llm_mode != "auto":
            return self.llm_mode
        if self.has_anthropic:
            return "anthropic"
        if self.has_groq:
            return "groq"
        return "mock"

    @property
    def jwt_verification(self) -> Literal["shared_secret", "jwks"]:
        """How access tokens get verified.

        A shared secret is offline and cheapest, so it wins when present. Its
        absence is not a misconfiguration — asymmetric projects have no secret
        to configure, and the JWKS endpoint is derived from `supabase_url`.
        """
        return "shared_secret" if _is_real(self.supabase_jwt_secret) else "jwks"

    @property
    def jwks_url(self) -> str:
        return f"{self.supabase_url.rstrip('/')}/auth/v1/.well-known/jwks.json"

    @property
    def sync_database_url(self) -> str:
        """The same DSN for libraries that cannot speak asyncpg (polars, pandas)."""
        return self.database_url.replace("+asyncpg", "")

    # ── Validation ──────────────────────────────────────────────────────────

    @model_validator(mode="after")
    def _reject_placeholders_in_production(self) -> Settings:
        if self.environment != "prod":
            return self

        missing: list[str] = []
        for name, value in (
            ("SUPABASE_URL", self.supabase_url),
            ("SUPABASE_ANON_KEY", self.supabase_anon_key),
            ("SUPABASE_SERVICE_ROLE_KEY", self.supabase_service_role_key),
            ("DATABASE_URL", self.database_url),
        ):
            if not _is_real(value):
                missing.append(name)

        # SUPABASE_JWT_SECRET is deliberately absent from that list. Projects on
        # asymmetric signing keys have no secret, and demanding one would refuse
        # to boot a correctly configured deployment. Verification falls back to
        # the JWKS endpoint, which only needs SUPABASE_URL — already checked.

        if self.llm_mode == "anthropic" and not self.has_anthropic:
            missing.append("ANTHROPIC_API_KEY (required by LLM_MODE=anthropic)")
        if self.llm_mode == "groq" and not self.has_groq:
            missing.append("GROQ_API_KEY (required by LLM_MODE=groq)")

        if missing:
            raise ValueError(
                "ENVIRONMENT=prod but these are unset or still placeholders: "
                + ", ".join(missing)
                + ". Fill them in .env — see .env.example."
            )
        return self


def _raw(value: SecretStr | str | None) -> str:
    if value is None:
        return ""
    return (value.get_secret_value() if isinstance(value, SecretStr) else value).strip()


def _is_real(value: SecretStr | str | None) -> bool:
    """True when a setting holds something an operator actually filled in."""
    raw = _raw(value)
    if not raw:
        return False
    # A value that starts with '#' is a comment python-dotenv folded in, not a
    # setting. Same for a bare '='-less fragment carrying whitespace.
    if raw.startswith("#"):
        return False
    return _PLACEHOLDER_RE.search(raw) is None


def _is_key(value: SecretStr | str | None, provider: str) -> bool:
    """True when a setting holds something shaped like that provider's key.

    Stricter than `_is_real` on purpose: reporting "key configured" for a value
    that cannot be a key sends the operator to the provider's dashboard to debug
    a 401, when the real problem is in their `.env`.
    """
    raw = _raw(value)
    if not _is_real(raw):
        return False
    prefixes = _KEY_PREFIXES.get(provider, ())
    if not prefixes:
        return True
    return raw.startswith(prefixes)


@lru_cache
def get_settings() -> Settings:
    return Settings()
