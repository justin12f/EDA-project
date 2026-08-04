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

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Environment = Literal["dev", "test", "prod"]
LLMMode = Literal["auto", "anthropic", "groq", "mock", "bridge"]
EmbeddingProviderName = Literal["fastembed", "none"]

_REPO_ROOT = Path(__file__).resolve().parents[4]

# Substrings that mark a value as still-a-placeholder rather than a credential.
_PLACEHOLDERS = ("YOUR-PROJECT-REF", "YOUR-DB-PASSWORD", "changeme", "xxx")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(_REPO_ROOT / ".env", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    environment: Environment = "dev"

    # ── Supabase ────────────────────────────────────────────────────────────
    supabase_url: str = "http://127.0.0.1:54321"
    supabase_anon_key: SecretStr = SecretStr("")
    supabase_service_role_key: SecretStr = SecretStr("")
    supabase_jwt_secret: SecretStr = SecretStr("")
    storage_bucket: str = "lumen"

    database_url: str = (
        "postgresql+asyncpg://postgres:postgres@127.0.0.1:54322/postgres"
    )

    # ── LLM ─────────────────────────────────────────────────────────────────
    llm_mode: LLMMode = "auto"
    anthropic_api_key: SecretStr | None = None
    groq_api_key: SecretStr | None = None

    model_reasoning: str = "claude-opus-5"
    model_specialist: str = "claude-sonnet-5"
    model_fast: str = "qwen/qwen3.6-27b"

    llm_bridge_inbox: str = ".llm-bridge"

    agent_max_iterations: int = 12
    agent_deadline_seconds: float = 180.0
    agent_max_total_tokens: int = 120_000

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
        return _is_real(self.anthropic_api_key)

    @property
    def has_groq(self) -> bool:
        return _is_real(self.groq_api_key)

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
            ("SUPABASE_JWT_SECRET", self.supabase_jwt_secret),
            ("DATABASE_URL", self.database_url),
        ):
            if not _is_real(value):
                missing.append(name)

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


def _is_real(value: SecretStr | str | None) -> bool:
    if value is None:
        return False
    raw = value.get_secret_value() if isinstance(value, SecretStr) else value
    raw = raw.strip()
    if not raw:
        return False
    return not any(marker in raw for marker in _PLACEHOLDERS)


@lru_cache
def get_settings() -> Settings:
    return Settings()
