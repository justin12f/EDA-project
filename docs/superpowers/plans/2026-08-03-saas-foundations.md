# SaaS Foundations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the CLI-only Python EDA engine into a multi-tenant SaaS backend foundation: one importable `lumen` package, a FastAPI service, a Postgres control plane with row-level security, and organization-scoped authentication.

**Architecture:** All existing top-level Python packages move under `engine/src/lumen/` so imports become `lumen.<domain>` — this also removes the current shadowing of the stdlib `statistics` module. `services/api` (FastAPI) depends on `lumen` and owns HTTP, auth, and tenancy. Every control-plane table carries `org_id` and a PostgreSQL RLS policy keyed on the `app.current_org` session variable, set per transaction. Sessions are opaque tokens in Redis; passwords are Argon2id.

**Tech Stack:** Python 3.11, uv, FastAPI, uvicorn, SQLAlchemy 2.x (async), Alembic, asyncpg, Redis 7, argon2-cffi, pydantic-settings, pytest + pytest-asyncio, Docker Compose, Postgres 15.

## Global Constraints

- Python 3.11. Dependencies managed with `uv` (`uv sync`, `uv run`), never bare `pip`.
- The engine stays free of web and tenant concerns. Nothing under `engine/src/lumen/` may import FastAPI, SQLAlchemy models from `services/api`, or reference `org_id`.
- Every new control-plane table has `org_id uuid NOT NULL` (except `users`, `organizations`, `sessions`) and an RLS policy. A test enumerates `pg_tables` and fails if any lacks `rowsecurity`.
- The application database role must **not** have `BYPASSRLS`. Migrations use a separate role.
- Money and identity values are never logged. Password hashes and data-source credentials never appear in any API response.
- All timestamps are `timestamptz`, stored UTC.
- Every commit message uses Conventional Commits (`feat:`, `fix:`, `refactor:`, `test:`, `chore:`).
- Run all commands from the repository root unless a task says otherwise.

---

## File Structure

```
engine/
  pyproject.toml                  lumen package definition
  src/lumen/
    __init__.py
    core/            backend.py, abstract_factory.py, inyeccion.py     (moved)
    readers/         base.py, polars_impl.py, spark_impl.py, ...       (moved)
    data_cleaning/   pipeline, steps, factories                        (moved)
    analyze_data/    analyzers                                         (moved)
    statistics/      11 domains                                        (moved — no longer shadows stdlib)
    preproccesing/   encoders, scalers, model_pre_input                (moved)
    models/          evaluation/  algorithms/  parsers/                (moved)
    agents/          master_factory.py, context_creator.py             (moved)
    model_tools/     tools                                             (moved)
services/api/
  pyproject.toml
  src/lumen_api/
    __init__.py
    main.py                       app factory, router registration, lifespan
    settings.py                   pydantic-settings, all env vars typed
    db/
      base.py                     DeclarativeBase, naming convention
      session.py                  async engine, session factory, org-scoped tx
      models/
        identity.py               User, Organization, Membership
        source.py                 DataSource
      __init__.py
    auth/
      password.py                 Argon2id hash/verify
      sessions.py                 Redis-backed opaque session tokens
      dependencies.py             current_user, current_org, require_role
      router.py                   /v1/auth/*
    orgs/router.py                /v1/orgs/*
    health.py                     /healthz, /readyz
    errors.py                     RFC 9457 problem+json handlers
  alembic/
    env.py
    versions/0001_identity.py
    versions/0002_rls.py
  tests/
    conftest.py                   db fixture, client fixture, org factories
    test_health.py
    test_rls.py
    test_auth.py
    test_orgs.py
infra/
  docker-compose.yml              db, redis, minio, api
  postgres/init/01-roles.sql      app role without BYPASSRLS
.env.example
Makefile
```

---

### Task 1: Consolidate the engine into a `lumen` package

**Files:**
- Create: `engine/pyproject.toml`, `engine/src/lumen/__init__.py`, `scripts/migrate_imports.py`
- Move: `agents/`, `algorithms/`, `analyze_data/`, `api/`, `core/`, `database/`, `data_cleaning/`, `evaluation/`, `models/`, `model_tools/`, `parsers/`, `preproccesing/`, `readers/`, `statistics/` → `engine/src/lumen/`
- Modify: every moved `.py` file's import statements
- Test: `engine/tests/test_package_imports.py`

**Interfaces:**
- Consumes: nothing
- Produces: importable package `lumen` with `lumen.agents.master_factory.AgentMasterFactory`, `lumen.core.backend.Backend | DEFAULT_BACKEND | validate_backend`, `lumen.readers.inyeccion.ReadersInyeccionDependency`, `lumen.data_cleaning.data_cleaning_pipeline.PipelineBuilder | DataCleaningPipeline`, `lumen.statistics.inyeccion.StatisticsInyeccionDependency`

- [ ] **Step 1: Write the failing test**

Create `engine/tests/test_package_imports.py`:

```python
"""The engine must be importable as one package and must not shadow the stdlib."""
import statistics as stdlib_statistics


def test_stdlib_statistics_is_not_shadowed():
    assert stdlib_statistics.mean([1, 2, 3]) == 2
    assert "lumen" not in (stdlib_statistics.__file__ or "")


def test_master_factory_imports():
    from lumen.agents.master_factory import AgentMasterFactory

    master = AgentMasterFactory("polars")
    assert master.backend == "polars"


def test_domain_layers_resolve():
    from lumen.agents.master_factory import AgentMasterFactory

    master = AgentMasterFactory("pandas")
    assert master.readers().backend == "pandas"
    assert master.cleaning().backend == "pandas"
    assert master.analyzers().backend == "pandas"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --directory engine pytest tests/test_package_imports.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lumen'`

- [ ] **Step 3: Create the package skeleton**

Create `engine/pyproject.toml`:

```toml
[project]
name = "lumen-engine"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
  "numpy>=1.26",
  "pandas>=2.2",
  "polars>=1.0",
  "pyarrow>=16",
  "scikit-learn>=1.5",
  "python-dotenv>=1.0",
  "langchain-core>=0.3",
]

[project.optional-dependencies]
spark = ["pyspark>=3.5"]
dev = ["pytest>=8", "pytest-asyncio>=0.24"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/lumen"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

Create `engine/src/lumen/__init__.py`:

```python
"""Lumen analytics engine — backend-agnostic data profiling, cleaning and statistics."""

__version__ = "0.1.0"
```

- [ ] **Step 4: Move the packages**

Run from the repository root:

```bash
mkdir -p engine/src/lumen engine/tests
for d in agents algorithms analyze_data api core database data_cleaning evaluation models model_tools parsers preproccesing readers statistics; do
  git mv "$d" "engine/src/lumen/$d"
done
find engine/src/lumen -name '__pycache__' -type d -prune -exec rm -rf {} +
```

- [ ] **Step 5: Rewrite imports with a script**

Create `scripts/migrate_imports.py`:

```python
"""Rewrite top-level engine imports to the lumen namespace. Idempotent."""
from __future__ import annotations

import re
import sys
from pathlib import Path

PACKAGES = (
    "agents", "algorithms", "analyze_data", "api", "core", "database",
    "data_cleaning", "evaluation", "models", "model_tools", "parsers",
    "preproccesing", "readers", "statistics",
)

FROM_RE = re.compile(rf"^(\s*)from\s+({'|'.join(PACKAGES)})(\.|\s)", re.M)
IMPORT_RE = re.compile(rf"^(\s*)import\s+({'|'.join(PACKAGES)})(\.|\s|$)", re.M)


def rewrite(text: str) -> str:
    text = FROM_RE.sub(r"\1from lumen.\2\3", text)
    text = IMPORT_RE.sub(r"\1import lumen.\2\3", text)
    return text


def main(root: str) -> int:
    changed = 0
    for path in Path(root).rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        original = path.read_text(encoding="utf-8")
        updated = rewrite(original)
        if updated != original:
            path.write_text(updated, encoding="utf-8")
            changed += 1
    print(f"rewrote {changed} files under {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "engine/src/lumen"))
```

Run:

```bash
python scripts/migrate_imports.py engine/src/lumen
python scripts/migrate_imports.py tests
```

- [ ] **Step 6: Move the existing test suite and install**

```bash
git mv tests engine/tests_legacy
mkdir -p engine/tests
git mv engine/tests_legacy/* engine/tests/ && rmdir engine/tests_legacy
uv sync --directory engine --extra dev
```

- [ ] **Step 7: Run the new test to verify it passes**

Run: `uv run --directory engine pytest tests/test_package_imports.py -v`
Expected: PASS — 3 passed

- [ ] **Step 8: Run the full legacy suite and record the baseline**

Run: `uv run --directory engine pytest tests -v --tb=short`
Expected: the same set of passes and failures as before the move. Spark tests may fail or skip if `pyspark` is not installed — that is acceptable and pre-existing. Any test that fails with `ModuleNotFoundError` naming a moved package is a migration defect: find the missed import and fix it.

- [ ] **Step 9: Delete the stale top-level entry point**

Delete `main.py`, `test_agent.py`, `test_spark.py`, `generate_analyzers_backends.py`, and `gbm_config.py` from the repository root — they are CLI-era scripts superseded by the API. Move the sample CSVs into `engine/tests/fixtures/data/`.

```bash
git rm main.py test_agent.py test_spark.py generate_analyzers_backends.py gbm_config.py
mkdir -p engine/tests/fixtures/data
git mv shopping_trends.csv clean_shopping_trends.csv dirty_data.csv "GBM - Acciones.csv" clean_gbm_acciones.csv gbm_clean_enriched.csv cleaning_report_shopping_trends.csv.json engine/tests/fixtures/data/
```

- [ ] **Step 10: Commit**

```bash
git add -A
git commit -m "refactor: consolidate engine into lumen package, drop stdlib statistics shadowing"
```

---

### Task 2: FastAPI service skeleton with typed settings

**Files:**
- Create: `services/api/pyproject.toml`, `services/api/src/lumen_api/__init__.py`, `services/api/src/lumen_api/settings.py`, `services/api/src/lumen_api/main.py`, `services/api/src/lumen_api/health.py`, `services/api/src/lumen_api/errors.py`, `.env.example`
- Test: `services/api/tests/test_health.py`, `services/api/tests/conftest.py`

**Interfaces:**
- Consumes: `lumen` package from Task 1
- Produces: `lumen_api.main.create_app() -> FastAPI`; `lumen_api.settings.Settings` with fields `database_url: PostgresDsn`, `redis_url: RedisDsn`, `session_ttl_seconds: int`, `anthropic_api_key: SecretStr | None`, `groq_api_key: SecretStr | None`, `storage_endpoint: str`, `storage_bucket: str`, `storage_access_key: SecretStr`, `storage_secret_key: SecretStr`, `environment: Literal["dev","test","prod"]`; `lumen_api.settings.get_settings() -> Settings` (lru_cached); `lumen_api.errors.ProblemDetail`

- [ ] **Step 1: Write the failing test**

Create `services/api/tests/test_health.py`:

```python
import pytest
from httpx import ASGITransport, AsyncClient

from lumen_api.main import create_app


@pytest.mark.asyncio
async def test_healthz_returns_ok():
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_openapi_is_served():
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/openapi.json")
    assert response.status_code == 200
    assert response.json()["info"]["title"] == "Lumen API"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --directory services/api pytest tests/test_health.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lumen_api'`

- [ ] **Step 3: Create the project definition**

Create `services/api/pyproject.toml`:

```toml
[project]
name = "lumen-api"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
  "lumen-engine",
  "fastapi>=0.115",
  "uvicorn[standard]>=0.32",
  "pydantic>=2.9",
  "pydantic-settings>=2.6",
  "sqlalchemy[asyncio]>=2.0",
  "asyncpg>=0.30",
  "alembic>=1.14",
  "redis>=5.2",
  "argon2-cffi>=23.1",
  "python-multipart>=0.0.12",
  "anthropic>=0.40",
  "openai>=1.55",
  "boto3>=1.35",
]

[project.optional-dependencies]
dev = ["pytest>=8", "pytest-asyncio>=0.24", "httpx>=0.28", "asgi-lifespan>=2.1"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/lumen_api"]

[tool.uv.sources]
lumen-engine = { path = "../../engine", editable = true }

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
asyncio_mode = "auto"
```

- [ ] **Step 4: Write settings**

Create `services/api/src/lumen_api/settings.py`:

```python
"""Typed application settings. Every environment variable the API reads is declared here."""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    environment: Literal["dev", "test", "prod"] = "dev"

    database_url: str = "postgresql+asyncpg://lumen_app:lumen_app@localhost:5432/lumen"
    redis_url: str = "redis://localhost:6379/0"

    session_ttl_seconds: int = 60 * 60 * 24 * 14
    session_cookie_name: str = "lumen_session"

    anthropic_api_key: SecretStr | None = None
    groq_api_key: SecretStr | None = None

    storage_endpoint: str = "http://localhost:9000"
    storage_bucket: str = "lumen"
    storage_access_key: SecretStr = SecretStr("minioadmin")
    storage_secret_key: SecretStr = SecretStr("minioadmin")

    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

- [ ] **Step 5: Write error handling**

Create `services/api/src/lumen_api/errors.py`:

```python
"""RFC 9457 problem+json error responses."""
from __future__ import annotations

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel


class ProblemDetail(BaseModel):
    type: str = "about:blank"
    title: str
    status: int
    detail: str | None = None


class AppError(Exception):
    """Base for errors the API converts into problem+json."""

    status_code: int = status.HTTP_400_BAD_REQUEST
    title: str = "Request failed"

    def __init__(self, detail: str | None = None) -> None:
        super().__init__(detail or self.title)
        self.detail = detail


class NotFound(AppError):
    status_code = status.HTTP_404_NOT_FOUND
    title = "Not found"


class Unauthorized(AppError):
    status_code = status.HTTP_401_UNAUTHORIZED
    title = "Not authenticated"


class Forbidden(AppError):
    status_code = status.HTTP_403_FORBIDDEN
    title = "Not permitted"


class Conflict(AppError):
    status_code = status.HTTP_409_CONFLICT
    title = "Conflict"


def _problem(status_code: int, title: str, detail: str | None) -> JSONResponse:
    body = ProblemDetail(title=title, status=status_code, detail=detail)
    return JSONResponse(
        status_code=status_code,
        content=body.model_dump(),
        media_type="application/problem+json",
    )


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error(_: Request, exc: AppError) -> JSONResponse:
        return _problem(exc.status_code, exc.title, exc.detail)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        return _problem(422, "Validation failed", str(exc.errors()))
```

- [ ] **Step 6: Write health and the app factory**

Create `services/api/src/lumen_api/health.py`:

```python
from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    """Liveness — the process is up. Never touches a dependency."""
    return {"status": "ok"}


@router.get("/readyz")
async def readyz() -> dict[str, str]:
    """Readiness — dependencies are reachable. Wired in Task 3."""
    return {"status": "ok"}
```

Create `services/api/src/lumen_api/main.py`:

```python
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from lumen_api import health
from lumen_api.errors import register_error_handlers
from lumen_api.settings import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Lumen API",
        version="0.1.0",
        description="Agentic data cleaning, pipelines and EDA.",
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
    return app


app = create_app()
```

Create `services/api/src/lumen_api/__init__.py` (empty file) and `services/api/tests/conftest.py`:

```python
import pytest


@pytest.fixture(autouse=True)
def _test_env(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "test")
```

- [ ] **Step 7: Install and run the tests**

Run:
```bash
uv sync --directory services/api --extra dev
uv run --directory services/api pytest tests/test_health.py -v
```
Expected: PASS — 2 passed

- [ ] **Step 8: Write `.env.example`**

Create `.env.example` at the repository root:

```dotenv
ENVIRONMENT=dev

DATABASE_URL=postgresql+asyncpg://lumen_app:lumen_app@localhost:5432/lumen
DATABASE_MIGRATION_URL=postgresql+asyncpg://lumen_migrator:lumen_migrator@localhost:5432/lumen
REDIS_URL=redis://localhost:6379/0

SESSION_TTL_SECONDS=1209600
SESSION_COOKIE_NAME=lumen_session

ANTHROPIC_API_KEY=
GROQ_API_KEY=

STORAGE_ENDPOINT=http://localhost:9000
STORAGE_BUCKET=lumen
STORAGE_ACCESS_KEY=minioadmin
STORAGE_SECRET_KEY=minioadmin

CORS_ORIGINS=["http://localhost:3000"]
```

- [ ] **Step 9: Commit**

```bash
git add services/api .env.example
git commit -m "feat: add FastAPI service skeleton with typed settings and problem+json errors"
```

---

### Task 3: Postgres control plane with row-level security

**Files:**
- Create: `infra/docker-compose.yml`, `infra/postgres/init/01-roles.sql`, `services/api/src/lumen_api/db/base.py`, `services/api/src/lumen_api/db/session.py`, `services/api/src/lumen_api/db/models/identity.py`, `services/api/src/lumen_api/db/models/source.py`, `services/api/alembic.ini`, `services/api/alembic/env.py`, `services/api/alembic/versions/0001_identity.py`, `services/api/alembic/versions/0002_rls.py`
- Modify: `services/api/src/lumen_api/health.py` (real readiness), `services/api/src/lumen_api/main.py` (lifespan)
- Test: `services/api/tests/test_rls.py`, `services/api/tests/conftest.py`

**Interfaces:**
- Consumes: `lumen_api.settings.get_settings`
- Produces: `lumen_api.db.base.Base`; `lumen_api.db.session.engine`, `session_factory`, `async def org_session(org_id: UUID) -> AsyncIterator[AsyncSession]`, `async def admin_session() -> AsyncIterator[AsyncSession]`; models `User(id, email, password_hash, display_name, created_at)`, `Organization(id, name, slug, created_at)`, `Membership(id, org_id, user_id, role)`, `DataSource(id, org_id, name, kind, dsn_encrypted, table_name, row_count, status, created_at)`

- [ ] **Step 1: Write the failing test**

Create `services/api/tests/test_rls.py`:

```python
"""Tenant isolation must be enforced by the database, not by application code."""
import uuid

import pytest
from sqlalchemy import text

from lumen_api.db.session import admin_session, org_session
from lumen_api.db.models.source import DataSource


@pytest.mark.asyncio
async def test_every_org_scoped_table_has_rls_enabled():
    async with admin_session() as session:
        rows = await session.execute(
            text(
                """
                SELECT c.relname
                FROM pg_class c
                JOIN pg_namespace n ON n.oid = c.relnamespace
                JOIN information_schema.columns col
                  ON col.table_name = c.relname AND col.table_schema = n.nspname
                WHERE n.nspname = 'public'
                  AND c.relkind = 'r'
                  AND col.column_name = 'org_id'
                  AND c.relrowsecurity = false
                """
            )
        )
        offenders = [r[0] for r in rows]
    assert offenders == [], f"tables with org_id but no RLS: {offenders}"


@pytest.mark.asyncio
async def test_org_cannot_read_another_orgs_rows(two_orgs):
    org_a, org_b = two_orgs

    async with org_session(org_a) as session:
        session.add(
            DataSource(
                id=uuid.uuid4(), org_id=org_a, name="a-source",
                kind="csv", table_name="a_tbl", status="idle",
            )
        )
        await session.commit()

    async with org_session(org_b) as session:
        result = await session.execute(text("SELECT count(*) FROM data_sources"))
        assert result.scalar_one() == 0

    async with org_session(org_a) as session:
        result = await session.execute(text("SELECT count(*) FROM data_sources"))
        assert result.scalar_one() == 1


@pytest.mark.asyncio
async def test_app_role_cannot_bypass_rls():
    async with admin_session() as session:
        result = await session.execute(
            text("SELECT rolbypassrls FROM pg_roles WHERE rolname = 'lumen_app'")
        )
        assert result.scalar_one() is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --directory services/api pytest tests/test_rls.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lumen_api.db'`

- [ ] **Step 3: Bring up Postgres and Redis with the right roles**

Create `infra/postgres/init/01-roles.sql`:

```sql
-- Migrations run as the owner; the application runs as a role RLS applies to.
CREATE ROLE lumen_migrator LOGIN PASSWORD 'lumen_migrator';
CREATE ROLE lumen_app      LOGIN PASSWORD 'lumen_app' NOBYPASSRLS;

CREATE DATABASE lumen OWNER lumen_migrator;

\connect lumen

GRANT USAGE ON SCHEMA public TO lumen_app;
ALTER DEFAULT PRIVILEGES FOR ROLE lumen_migrator IN SCHEMA public
  GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO lumen_app;
ALTER DEFAULT PRIVILEGES FOR ROLE lumen_migrator IN SCHEMA public
  GRANT USAGE, SELECT ON SEQUENCES TO lumen_app;
```

Create `infra/docker-compose.yml`:

```yaml
name: lumen

services:
  db:
    image: postgres:15-alpine
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: postgres
    ports: ["5432:5432"]
    volumes:
      - pgdata:/var/lib/postgresql/data
      - ./postgres/init:/docker-entrypoint-initdb.d:ro
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 3s
      retries: 20

  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 3s
      retries: 20

  storage:
    image: minio/minio:latest
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    ports: ["9000:9000", "9001:9001"]
    volumes:
      - miniodata:/data

volumes:
  pgdata:
  miniodata:
```

Run: `docker compose -f infra/docker-compose.yml up -d db redis storage`
Verify: `docker compose -f infra/docker-compose.yml ps` shows all three healthy.

- [ ] **Step 4: Write the declarative base and models**

Create `services/api/src/lumen_api/db/base.py`:

```python
from __future__ import annotations

from sqlalchemy import MetaData
from sqlalchemy.orm import DeclarativeBase

NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)
```

Create `services/api/src/lumen_api/db/models/identity.py`:

```python
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import UUID as PgUUID
from sqlalchemy.orm import Mapped, mapped_column

from lumen_api.db.base import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(PgUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String(320), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    display_name: Mapped[str] = mapped_column(String(120), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Organization(Base):
    __tablename__ = "organizations"

    id: Mapped[uuid.UUID] = mapped_column(PgUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    slug: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    plan_code: Mapped[str] = mapped_column(String(16), nullable=False, default="free")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Membership(Base):
    __tablename__ = "memberships"
    __table_args__ = (UniqueConstraint("org_id", "user_id"),)

    id: Mapped[uuid.UUID] = mapped_column(PgUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    org_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False, default="member")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
```

Create `services/api/src/lumen_api/db/models/source.py`:

```python
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import BigInteger, DateTime, ForeignKey, String, Text, func
from sqlalchemy.dialects.postgresql import UUID as PgUUID
from sqlalchemy.orm import Mapped, mapped_column

from lumen_api.db.base import Base


class DataSource(Base):
    __tablename__ = "data_sources"

    id: Mapped[uuid.UUID] = mapped_column(PgUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    org_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    kind: Mapped[str] = mapped_column(String(16), nullable=False)  # postgres|mysql|csv|json|parquet
    dsn_encrypted: Mapped[str | None] = mapped_column(Text, nullable=True)
    object_uri: Mapped[str | None] = mapped_column(Text, nullable=True)
    table_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    row_count: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="idle")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
```

Create `services/api/src/lumen_api/db/models/__init__.py`:

```python
from lumen_api.db.models.identity import Membership, Organization, User
from lumen_api.db.models.source import DataSource

__all__ = ["User", "Organization", "Membership", "DataSource"]
```

- [ ] **Step 5: Write the org-scoped session**

Create `services/api/src/lumen_api/db/session.py`:

```python
"""Database sessions.

`org_session` is the only session application code may use for tenant data. It opens a
transaction and sets `app.current_org` with SET LOCAL, so the value is scoped to that
transaction and cannot leak to the next borrower of a pooled connection.
"""
from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from lumen_api.settings import get_settings

_settings = get_settings()

engine = create_async_engine(_settings.database_url, pool_pre_ping=True, pool_size=10)
session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


@asynccontextmanager
async def org_session(org_id: uuid.UUID) -> AsyncIterator[AsyncSession]:
    """Yield a session whose every statement is confined to `org_id` by RLS."""
    async with session_factory() as session:
        await session.begin()
        await session.execute(
            text("SELECT set_config('app.current_org', :org, true)"),
            {"org": str(org_id)},
        )
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


@asynccontextmanager
async def admin_session() -> AsyncIterator[AsyncSession]:
    """Yield a session with no org context. Only for auth, signup and diagnostics."""
    async with session_factory() as session:
        async with session.begin():
            yield session
```

- [ ] **Step 6: Write the migrations**

Create `services/api/alembic.ini` with `script_location = alembic` and `sqlalchemy.url =` left blank (env.py supplies it).

Create `services/api/alembic/env.py`:

```python
from __future__ import annotations

import asyncio
import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy.ext.asyncio import async_engine_from_config
from sqlalchemy import pool

from lumen_api.db.base import Base
from lumen_api.db import models  # noqa: F401  (registers metadata)

config = context.config
if config.config_file_name:
    fileConfig(config.config_file_name)

config.set_main_option(
    "sqlalchemy.url",
    os.environ.get(
        "DATABASE_MIGRATION_URL",
        "postgresql+asyncpg://lumen_migrator:lumen_migrator@localhost:5432/lumen",
    ),
)
target_metadata = Base.metadata


def _run(connection):
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_async():
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(_run)
    await connectable.dispose()


asyncio.run(run_async())
```

Create `services/api/alembic/versions/0001_identity.py` — generate it, do not hand-write the tables:

```bash
uv run --directory services/api alembic revision --autogenerate -m "identity and sources" --rev-id 0001
```

Then create `services/api/alembic/versions/0002_rls.py` by hand:

```python
"""Enable row-level security on every org-scoped table."""
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None

ORG_SCOPED_TABLES = ("data_sources", "memberships")


def upgrade() -> None:
    for table in ORG_SCOPED_TABLES:
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        op.execute(
            f"""
            CREATE POLICY {table}_org_isolation ON {table}
            USING (org_id = current_setting('app.current_org', true)::uuid)
            WITH CHECK (org_id = current_setting('app.current_org', true)::uuid)
            """
        )


def downgrade() -> None:
    for table in ORG_SCOPED_TABLES:
        op.execute(f"DROP POLICY IF EXISTS {table}_org_isolation ON {table}")
        op.execute(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY")
```

Run: `uv run --directory services/api alembic upgrade head`

- [ ] **Step 7: Add the test fixtures**

Append to `services/api/tests/conftest.py`:

```python
import uuid

import pytest_asyncio
from sqlalchemy import text

from lumen_api.db.session import admin_session


@pytest_asyncio.fixture
async def two_orgs():
    """Create two organizations and clean them up afterwards."""
    org_a, org_b = uuid.uuid4(), uuid.uuid4()
    async with admin_session() as session:
        for oid, slug in ((org_a, f"a-{oid_short(org_a)}"), (org_b, f"b-{oid_short(org_b)}")):
            await session.execute(
                text(
                    "INSERT INTO organizations (id, name, slug, plan_code) "
                    "VALUES (:id, :name, :slug, 'free')"
                ),
                {"id": oid, "name": slug, "slug": slug},
            )
    yield org_a, org_b
    async with admin_session() as session:
        await session.execute(
            text("DELETE FROM organizations WHERE id = ANY(:ids)"),
            {"ids": [org_a, org_b]},
        )


def oid_short(value: uuid.UUID) -> str:
    return value.hex[:8]
```

- [ ] **Step 8: Run the tests to verify they pass**

Run: `uv run --directory services/api pytest tests/test_rls.py -v`
Expected: PASS — 3 passed. If `test_org_cannot_read_another_orgs_rows` fails with rows visible, the policy is missing `FORCE ROW LEVEL SECURITY` (table owners bypass policies without it).

- [ ] **Step 9: Wire real readiness**

Replace the body of `readyz` in `services/api/src/lumen_api/health.py`:

```python
from fastapi import APIRouter
from redis.asyncio import Redis
from sqlalchemy import text

from lumen_api.db.session import engine
from lumen_api.settings import get_settings

router = APIRouter(tags=["health"])


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/readyz")
async def readyz() -> dict[str, str]:
    async with engine.connect() as conn:
        await conn.execute(text("SELECT 1"))
    redis = Redis.from_url(get_settings().redis_url)
    try:
        await redis.ping()
    finally:
        await redis.aclose()
    return {"status": "ok"}
```

- [ ] **Step 10: Commit**

```bash
git add infra services/api
git commit -m "feat: add postgres control plane with row-level tenant isolation"
```

---

### Task 4: Password hashing and Redis-backed sessions

**Files:**
- Create: `services/api/src/lumen_api/auth/__init__.py`, `services/api/src/lumen_api/auth/password.py`, `services/api/src/lumen_api/auth/sessions.py`
- Test: `services/api/tests/test_password.py`, `services/api/tests/test_sessions.py`

**Interfaces:**
- Consumes: `lumen_api.settings.get_settings`
- Produces: `hash_password(plain: str) -> str`, `verify_password(plain: str, hashed: str) -> bool`, `needs_rehash(hashed: str) -> bool`; `SessionStore` with `async create(user_id: UUID, org_id: UUID) -> str`, `async read(token: str) -> SessionData | None`, `async revoke(token: str) -> None`, `async revoke_all_for_user(user_id: UUID) -> None`; `SessionData(user_id: UUID, org_id: UUID)`

- [ ] **Step 1: Write the failing tests**

Create `services/api/tests/test_password.py`:

```python
from lumen_api.auth.password import hash_password, verify_password


def test_hash_is_argon2id_and_salted():
    a = hash_password("correct horse battery staple")
    b = hash_password("correct horse battery staple")
    assert a.startswith("$argon2id$")
    assert a != b, "identical passwords must not produce identical hashes"


def test_verify_accepts_the_right_password():
    hashed = hash_password("s3cret-pass")
    assert verify_password("s3cret-pass", hashed) is True


def test_verify_rejects_the_wrong_password():
    hashed = hash_password("s3cret-pass")
    assert verify_password("wrong-pass", hashed) is False


def test_verify_rejects_a_malformed_hash():
    assert verify_password("anything", "not-a-hash") is False
```

Create `services/api/tests/test_sessions.py`:

```python
import uuid

import pytest

from lumen_api.auth.sessions import SessionStore


@pytest.mark.asyncio
async def test_create_then_read_round_trips():
    store = SessionStore()
    user_id, org_id = uuid.uuid4(), uuid.uuid4()
    token = await store.create(user_id, org_id)
    assert len(token) >= 43, "token must carry at least 256 bits of entropy"

    data = await store.read(token)
    assert data is not None
    assert data.user_id == user_id
    assert data.org_id == org_id
    await store.revoke(token)


@pytest.mark.asyncio
async def test_revoked_token_reads_as_none():
    store = SessionStore()
    token = await store.create(uuid.uuid4(), uuid.uuid4())
    await store.revoke(token)
    assert await store.read(token) is None


@pytest.mark.asyncio
async def test_unknown_token_reads_as_none():
    store = SessionStore()
    assert await store.read("nonexistent-token") is None


@pytest.mark.asyncio
async def test_revoke_all_kills_every_session_for_a_user():
    store = SessionStore()
    user_id = uuid.uuid4()
    org_id = uuid.uuid4()
    t1 = await store.create(user_id, org_id)
    t2 = await store.create(user_id, org_id)
    await store.revoke_all_for_user(user_id)
    assert await store.read(t1) is None
    assert await store.read(t2) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --directory services/api pytest tests/test_password.py tests/test_sessions.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lumen_api.auth'`

- [ ] **Step 3: Implement password hashing**

Create `services/api/src/lumen_api/auth/password.py`:

```python
"""Argon2id password hashing.

Parameters follow the OWASP 2024 baseline for Argon2id: 19 MiB memory, 2 iterations,
1 degree of parallelism.
"""
from __future__ import annotations

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerификationError, VerifyMismatchError

_hasher = PasswordHasher(time_cost=2, memory_cost=19456, parallelism=1)


def hash_password(plain: str) -> str:
    return _hasher.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return _hasher.verify(hashed, plain)
    except (VerifyMismatchError, InvalidHashError, VerificationError):
        return False


def needs_rehash(hashed: str) -> bool:
    try:
        return _hasher.check_needs_rehash(hashed)
    except InvalidHashError:
        return True
```

> Implementer note: the import line above must read
> `from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError`.
> Type it out; do not copy a mangled identifier.

- [ ] **Step 4: Implement the session store**

Create `services/api/src/lumen_api/auth/sessions.py`:

```python
"""Opaque server-side sessions in Redis.

The browser only ever holds a random token. Everything else lives server-side, so a
logout is a delete and a compromised token can be revoked instantly.
"""
from __future__ import annotations

import secrets
import uuid
from dataclasses import dataclass

from redis.asyncio import Redis

from lumen_api.settings import get_settings

_KEY = "session:{token}"
_USER_INDEX = "user-sessions:{user_id}"


@dataclass(frozen=True)
class SessionData:
    user_id: uuid.UUID
    org_id: uuid.UUID


class SessionStore:
    def __init__(self, redis: Redis | None = None) -> None:
        settings = get_settings()
        self._redis = redis or Redis.from_url(settings.redis_url, decode_responses=True)
        self._ttl = settings.session_ttl_seconds

    async def create(self, user_id: uuid.UUID, org_id: uuid.UUID) -> str:
        token = secrets.token_urlsafe(32)
        key = _KEY.format(token=token)
        index = _USER_INDEX.format(user_id=user_id)
        pipe = self._redis.pipeline()
        pipe.hset(key, mapping={"user_id": str(user_id), "org_id": str(org_id)})
        pipe.expire(key, self._ttl)
        pipe.sadd(index, token)
        pipe.expire(index, self._ttl)
        await pipe.execute()
        return token

    async def read(self, token: str) -> SessionData | None:
        raw = await self._redis.hgetall(_KEY.format(token=token))
        if not raw:
            return None
        return SessionData(
            user_id=uuid.UUID(raw["user_id"]), org_id=uuid.UUID(raw["org_id"])
        )

    async def switch_org(self, token: str, org_id: uuid.UUID) -> None:
        key = _KEY.format(token=token)
        if not await self._redis.exists(key):
            return
        await self._redis.hset(key, "org_id", str(org_id))

    async def revoke(self, token: str) -> None:
        data = await self.read(token)
        pipe = self._redis.pipeline()
        pipe.delete(_KEY.format(token=token))
        if data:
            pipe.srem(_USER_INDEX.format(user_id=data.user_id), token)
        await pipe.execute()

    async def revoke_all_for_user(self, user_id: uuid.UUID) -> None:
        index = _USER_INDEX.format(user_id=user_id)
        tokens = await self._redis.smembers(index)
        pipe = self._redis.pipeline()
        for token in tokens:
            pipe.delete(_KEY.format(token=token))
        pipe.delete(index)
        await pipe.execute()
```

Create an empty `services/api/src/lumen_api/auth/__init__.py`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run --directory services/api pytest tests/test_password.py tests/test_sessions.py -v`
Expected: PASS — 8 passed

- [ ] **Step 6: Commit**

```bash
git add services/api/src/lumen_api/auth services/api/tests/test_password.py services/api/tests/test_sessions.py
git commit -m "feat: add argon2id password hashing and redis session store"
```

---

### Task 5: Signup, login, logout and the current-user dependency

**Files:**
- Create: `services/api/src/lumen_api/auth/dependencies.py`, `services/api/src/lumen_api/auth/router.py`, `services/api/src/lumen_api/auth/schemas.py`
- Modify: `services/api/src/lumen_api/main.py` (register the router)
- Test: `services/api/tests/test_auth.py`

**Interfaces:**
- Consumes: `hash_password`, `verify_password`, `SessionStore`, `admin_session`, `org_session`, models from Task 3
- Produces: dependencies `current_session(request) -> SessionData`, `current_user(...) -> User`, `current_org_id(...) -> UUID`, `require_role(*roles) -> Callable`; routes `POST /v1/auth/signup`, `POST /v1/auth/login`, `POST /v1/auth/logout`, `GET /v1/auth/me`; schemas `SignupRequest(email, password, display_name, org_name)`, `LoginRequest(email, password)`, `MeResponse(user_id, email, display_name, org_id, org_name, role, plan_code)`

- [ ] **Step 1: Write the failing test**

Create `services/api/tests/test_auth.py`:

```python
import uuid

import pytest
from httpx import ASGITransport, AsyncClient

from lumen_api.main import create_app


@pytest.fixture
def client():
    app = create_app()
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


def _signup_payload():
    tag = uuid.uuid4().hex[:8]
    return {
        "email": f"ana+{tag}@lumen.dev",
        "password": "correct horse battery staple",
        "display_name": "Ana Kovač",
        "org_name": f"Acme {tag}",
    }


@pytest.mark.asyncio
async def test_signup_creates_user_org_and_owner_membership(client):
    payload = _signup_payload()
    async with client as c:
        response = await c.post("/v1/auth/signup", json=payload)
        assert response.status_code == 201
        body = response.json()
        assert body["email"] == payload["email"]
        assert body["role"] == "owner"
        assert c.cookies.get("lumen_session")

        me = await c.get("/v1/auth/me")
        assert me.status_code == 200
        assert me.json()["org_name"] == payload["org_name"]


@pytest.mark.asyncio
async def test_signup_rejects_a_duplicate_email(client):
    payload = _signup_payload()
    async with client as c:
        assert (await c.post("/v1/auth/signup", json=payload)).status_code == 201
        second = await c.post("/v1/auth/signup", json=payload)
        assert second.status_code == 409


@pytest.mark.asyncio
async def test_login_succeeds_and_wrong_password_fails(client):
    payload = _signup_payload()
    async with client as c:
        await c.post("/v1/auth/signup", json=payload)
        await c.post("/v1/auth/logout")

        bad = await c.post(
            "/v1/auth/login", json={"email": payload["email"], "password": "wrong"}
        )
        assert bad.status_code == 401

        good = await c.post(
            "/v1/auth/login",
            json={"email": payload["email"], "password": payload["password"]},
        )
        assert good.status_code == 200
        assert (await c.get("/v1/auth/me")).status_code == 200


@pytest.mark.asyncio
async def test_me_requires_authentication(client):
    async with client as c:
        assert (await c.get("/v1/auth/me")).status_code == 401


@pytest.mark.asyncio
async def test_logout_invalidates_the_session(client):
    payload = _signup_payload()
    async with client as c:
        await c.post("/v1/auth/signup", json=payload)
        assert (await c.post("/v1/auth/logout")).status_code == 204
        assert (await c.get("/v1/auth/me")).status_code == 401


@pytest.mark.asyncio
async def test_password_hash_never_leaves_the_api(client):
    payload = _signup_payload()
    async with client as c:
        signup = await c.post("/v1/auth/signup", json=payload)
        me = await c.get("/v1/auth/me")
    assert "password" not in signup.text.lower()
    assert "argon2" not in signup.text.lower()
    assert "password" not in me.text.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --directory services/api pytest tests/test_auth.py -v`
Expected: FAIL — 404 on `/v1/auth/signup`

- [ ] **Step 3: Write the schemas**

Create `services/api/src/lumen_api/auth/schemas.py`:

```python
from __future__ import annotations

import uuid

from pydantic import BaseModel, EmailStr, Field


class SignupRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=12, max_length=256)
    display_name: str = Field(min_length=1, max_length=120)
    org_name: str = Field(min_length=1, max_length=120)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=256)


class MeResponse(BaseModel):
    user_id: uuid.UUID
    email: EmailStr
    display_name: str
    org_id: uuid.UUID
    org_name: str
    role: str
    plan_code: str
```

- [ ] **Step 4: Write the dependencies**

Create `services/api/src/lumen_api/auth/dependencies.py`:

```python
from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy import select

from lumen_api.auth.sessions import SessionData, SessionStore
from lumen_api.db.models.identity import Membership, User
from lumen_api.db.session import admin_session
from lumen_api.errors import Forbidden, Unauthorized
from lumen_api.settings import get_settings


def get_session_store() -> SessionStore:
    return SessionStore()


async def current_session(
    request: Request,
    store: Annotated[SessionStore, Depends(get_session_store)],
) -> SessionData:
    token = request.cookies.get(get_settings().session_cookie_name)
    if not token:
        raise Unauthorized("No session cookie")
    data = await store.read(token)
    if data is None:
        raise Unauthorized("Session expired or revoked")
    return data


async def current_user(
    session: Annotated[SessionData, Depends(current_session)],
) -> User:
    async with admin_session() as db:
        user = await db.get(User, session.user_id)
    if user is None:
        raise Unauthorized("User no longer exists")
    return user


async def current_org_id(
    session: Annotated[SessionData, Depends(current_session)],
) -> uuid.UUID:
    return session.org_id


async def current_role(
    session: Annotated[SessionData, Depends(current_session)],
) -> str:
    async with admin_session() as db:
        result = await db.execute(
            select(Membership.role).where(
                Membership.org_id == session.org_id,
                Membership.user_id == session.user_id,
            )
        )
        role = result.scalar_one_or_none()
    if role is None:
        raise Forbidden("Not a member of this organization")
    return role


def require_role(*allowed: str) -> Callable[[str], str]:
    async def _check(role: Annotated[str, Depends(current_role)]) -> str:
        if role not in allowed:
            raise Forbidden(f"Requires one of: {', '.join(allowed)}")
        return role

    return _check
```

- [ ] **Step 5: Write the router**

Create `services/api/src/lumen_api/auth/router.py`:

```python
from __future__ import annotations

import re
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from lumen_api.auth.dependencies import current_session, get_session_store
from lumen_api.auth.password import hash_password, verify_password
from lumen_api.auth.schemas import LoginRequest, MeResponse, SignupRequest
from lumen_api.auth.sessions import SessionData, SessionStore
from lumen_api.db.models.identity import Membership, Organization, User
from lumen_api.db.session import admin_session
from lumen_api.errors import Conflict, Unauthorized
from lumen_api.settings import get_settings

router = APIRouter(prefix="/v1/auth", tags=["auth"])


def _slugify(value: str) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "org"
    return f"{base[:48]}-{uuid.uuid4().hex[:6]}"


def _set_cookie(response: Response, token: str) -> None:
    settings = get_settings()
    response.set_cookie(
        settings.session_cookie_name,
        token,
        max_age=settings.session_ttl_seconds,
        httponly=True,
        secure=settings.environment == "prod",
        samesite="lax",
        path="/",
    )


async def _build_me(user: User, org: Organization, role: str) -> MeResponse:
    return MeResponse(
        user_id=user.id,
        email=user.email,
        display_name=user.display_name,
        org_id=org.id,
        org_name=org.name,
        role=role,
        plan_code=org.plan_code,
    )


@router.post("/signup", response_model=MeResponse, status_code=status.HTTP_201_CREATED)
async def signup(
    payload: SignupRequest,
    response: Response,
    store: Annotated[SessionStore, Depends(get_session_store)],
) -> MeResponse:
    user = User(
        email=payload.email.lower(),
        password_hash=hash_password(payload.password),
        display_name=payload.display_name,
    )
    org = Organization(name=payload.org_name, slug=_slugify(payload.org_name))
    try:
        async with admin_session() as db:
            db.add(user)
            db.add(org)
            await db.flush()
            db.add(Membership(org_id=org.id, user_id=user.id, role="owner"))
    except IntegrityError as exc:
        raise Conflict("An account with that email already exists") from exc

    token = await store.create(user.id, org.id)
    _set_cookie(response, token)
    return await _build_me(user, org, "owner")


@router.post("/login", response_model=MeResponse)
async def login(
    payload: LoginRequest,
    response: Response,
    store: Annotated[SessionStore, Depends(get_session_store)],
) -> MeResponse:
    async with admin_session() as db:
        result = await db.execute(select(User).where(User.email == payload.email.lower()))
        user = result.scalar_one_or_none()
        if user is None or not verify_password(payload.password, user.password_hash):
            raise Unauthorized("Invalid email or password")
        membership = await db.execute(
            select(Membership).where(Membership.user_id == user.id).limit(1)
        )
        member = membership.scalar_one_or_none()
        if member is None:
            raise Unauthorized("Account has no organization")
        org = await db.get(Organization, member.org_id)

    token = await store.create(user.id, org.id)
    _set_cookie(response, token)
    return await _build_me(user, org, member.role)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(
    request: Request,
    response: Response,
    store: Annotated[SessionStore, Depends(get_session_store)],
) -> Response:
    settings = get_settings()
    token = request.cookies.get(settings.session_cookie_name)
    if token:
        await store.revoke(token)
    response.delete_cookie(settings.session_cookie_name, path="/")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/me", response_model=MeResponse)
async def me(session: Annotated[SessionData, Depends(current_session)]) -> MeResponse:
    async with admin_session() as db:
        user = await db.get(User, session.user_id)
        org = await db.get(Organization, session.org_id)
        role_row = await db.execute(
            select(Membership.role).where(
                Membership.org_id == session.org_id,
                Membership.user_id == session.user_id,
            )
        )
        role = role_row.scalar_one_or_none()
    if user is None or org is None or role is None:
        raise Unauthorized("Session no longer valid")
    return await _build_me(user, org, role)
```

- [ ] **Step 6: Register the router**

In `services/api/src/lumen_api/main.py`, add the import and the include:

```python
from lumen_api.auth import router as auth_router
...
    app.include_router(health.router)
    app.include_router(auth_router.router)
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `uv run --directory services/api pytest tests/test_auth.py -v`
Expected: PASS — 6 passed

- [ ] **Step 8: Run the whole API suite**

Run: `uv run --directory services/api pytest tests -v`
Expected: PASS — all tests from Tasks 2–5 pass.

- [ ] **Step 9: Commit**

```bash
git add services/api
git commit -m "feat: add signup, login, logout and session-backed auth dependencies"
```

---

### Task 6: Data source CRUD scoped to the organization

**Files:**
- Create: `services/api/src/lumen_api/sources/__init__.py`, `services/api/src/lumen_api/sources/schemas.py`, `services/api/src/lumen_api/sources/router.py`
- Modify: `services/api/src/lumen_api/main.py`
- Test: `services/api/tests/test_sources.py`

**Interfaces:**
- Consumes: `current_org_id`, `require_role`, `org_session`, `DataSource`
- Produces: routes `GET /v1/sources`, `POST /v1/sources`, `GET /v1/sources/{id}`, `DELETE /v1/sources/{id}`; schemas `DataSourceCreate(name, kind, table_name)`, `DataSourceOut(id, name, kind, table_name, row_count, status, created_at)`

- [ ] **Step 1: Write the failing test**

Create `services/api/tests/test_sources.py`:

```python
import uuid

import pytest
from httpx import ASGITransport, AsyncClient

from lumen_api.main import create_app


async def _signed_up_client() -> AsyncClient:
    client = AsyncClient(transport=ASGITransport(app=create_app()), base_url="http://test")
    tag = uuid.uuid4().hex[:8]
    await client.post(
        "/v1/auth/signup",
        json={
            "email": f"user+{tag}@lumen.dev",
            "password": "correct horse battery staple",
            "display_name": "User",
            "org_name": f"Org {tag}",
        },
    )
    return client


@pytest.mark.asyncio
async def test_create_then_list_returns_the_source():
    client = await _signed_up_client()
    async with client as c:
        created = await c.post(
            "/v1/sources",
            json={"name": "users_2024.csv", "kind": "csv", "table_name": "users_2024"},
        )
        assert created.status_code == 201
        assert created.json()["name"] == "users_2024.csv"

        listed = await c.get("/v1/sources")
        assert listed.status_code == 200
        assert [s["name"] for s in listed.json()] == ["users_2024.csv"]


@pytest.mark.asyncio
async def test_a_second_org_sees_none_of_the_first_orgs_sources():
    first = await _signed_up_client()
    async with first as c:
        await c.post("/v1/sources", json={"name": "secret.csv", "kind": "csv", "table_name": "s"})

    second = await _signed_up_client()
    async with second as c:
        listed = await c.get("/v1/sources")
        assert listed.status_code == 200
        assert listed.json() == []


@pytest.mark.asyncio
async def test_sources_require_authentication():
    client = AsyncClient(transport=ASGITransport(app=create_app()), base_url="http://test")
    async with client as c:
        assert (await c.get("/v1/sources")).status_code == 401


@pytest.mark.asyncio
async def test_unknown_source_id_is_404():
    client = await _signed_up_client()
    async with client as c:
        response = await c.get(f"/v1/sources/{uuid.uuid4()}")
        assert response.status_code == 404


@pytest.mark.asyncio
async def test_invalid_kind_is_rejected():
    client = await _signed_up_client()
    async with client as c:
        response = await c.post(
            "/v1/sources", json={"name": "x", "kind": "mongodb", "table_name": "x"}
        )
        assert response.status_code == 422
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --directory services/api pytest tests/test_sources.py -v`
Expected: FAIL — 404 on `/v1/sources`

- [ ] **Step 3: Write the schemas**

Create `services/api/src/lumen_api/sources/schemas.py`:

```python
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SourceKind = Literal["postgres", "mysql", "csv", "json", "parquet"]


class DataSourceCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    kind: SourceKind
    table_name: str | None = Field(default=None, max_length=128)


class DataSourceOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    kind: SourceKind
    table_name: str | None
    row_count: int | None
    status: str
    created_at: datetime
```

- [ ] **Step 4: Write the router**

Create `services/api/src/lumen_api/sources/router.py`:

```python
from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, status
from sqlalchemy import select

from lumen_api.auth.dependencies import current_org_id, require_role
from lumen_api.db.models.source import DataSource
from lumen_api.db.session import org_session
from lumen_api.errors import NotFound
from lumen_api.sources.schemas import DataSourceCreate, DataSourceOut

router = APIRouter(prefix="/v1/sources", tags=["sources"])

OrgId = Annotated[uuid.UUID, Depends(current_org_id)]


@router.get("", response_model=list[DataSourceOut])
async def list_sources(org_id: OrgId) -> list[DataSourceOut]:
    async with org_session(org_id) as db:
        result = await db.execute(select(DataSource).order_by(DataSource.created_at))
        return [DataSourceOut.model_validate(row) for row in result.scalars()]


@router.post("", response_model=DataSourceOut, status_code=status.HTTP_201_CREATED)
async def create_source(
    payload: DataSourceCreate,
    org_id: OrgId,
    _: Annotated[str, Depends(require_role("owner", "admin", "member"))],
) -> DataSourceOut:
    source = DataSource(
        org_id=org_id,
        name=payload.name,
        kind=payload.kind,
        table_name=payload.table_name,
        status="idle",
    )
    async with org_session(org_id) as db:
        db.add(source)
        await db.flush()
        await db.refresh(source)
        return DataSourceOut.model_validate(source)


@router.get("/{source_id}", response_model=DataSourceOut)
async def get_source(source_id: uuid.UUID, org_id: OrgId) -> DataSourceOut:
    async with org_session(org_id) as db:
        source = await db.get(DataSource, source_id)
        if source is None:
            raise NotFound("Data source not found")
        return DataSourceOut.model_validate(source)


@router.delete("/{source_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_source(
    source_id: uuid.UUID,
    org_id: OrgId,
    _: Annotated[str, Depends(require_role("owner", "admin"))],
) -> None:
    async with org_session(org_id) as db:
        source = await db.get(DataSource, source_id)
        if source is None:
            raise NotFound("Data source not found")
        await db.delete(source)
```

Create an empty `services/api/src/lumen_api/sources/__init__.py`. Register in `main.py`:

```python
from lumen_api.sources import router as sources_router
...
    app.include_router(sources_router.router)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run --directory services/api pytest tests/test_sources.py -v`
Expected: PASS — 5 passed

- [ ] **Step 6: Run the whole suite and commit**

```bash
uv run --directory services/api pytest tests -v
git add services/api
git commit -m "feat: add org-scoped data source endpoints"
```

---

### Task 7: Developer entry points and containerized API

**Files:**
- Create: `Makefile`, `services/api/Dockerfile`, `engine/Dockerfile`
- Modify: `infra/docker-compose.yml` (add the `api` service), delete the repository-root `Dockerfile` and `docker-compose.yml`
- Test: `services/api/tests/test_smoke_container.py` (marked `integration`)

**Interfaces:**
- Consumes: everything above
- Produces: `make dev`, `make test`, `make migrate`, `make fmt`; `lumen-api` image serving on `:8000`

- [ ] **Step 1: Write the failing test**

Create `services/api/tests/test_smoke_container.py`:

```python
"""Runs only when the compose stack is up: `make dev` then `pytest -m integration`."""
import os

import httpx
import pytest

pytestmark = pytest.mark.integration

API = os.environ.get("LUMEN_API_URL", "http://localhost:8000")


def test_container_serves_healthz():
    response = httpx.get(f"{API}/healthz", timeout=10)
    assert response.status_code == 200


def test_container_serves_readyz_with_live_dependencies():
    response = httpx.get(f"{API}/readyz", timeout=10)
    assert response.status_code == 200
```

Register the marker by adding to `services/api/pyproject.toml` under `[tool.pytest.ini_options]`:

```toml
markers = ["integration: requires the docker compose stack"]
addopts = "-m 'not integration'"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --directory services/api pytest tests/test_smoke_container.py -m integration -v`
Expected: FAIL — `httpx.ConnectError` (nothing listening on 8000)

- [ ] **Step 3: Write the Dockerfile**

Create `services/api/Dockerfile`:

```dockerfile
FROM python:3.11-slim AS base
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY engine /app/engine
COPY services/api /app/services/api

WORKDIR /app/services/api
RUN uv sync --frozen --no-dev || uv sync --no-dev

EXPOSE 8000
CMD ["uv", "run", "uvicorn", "lumen_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 4: Add the API service to compose**

Append to `infra/docker-compose.yml` under `services:`:

```yaml
  api:
    build:
      context: ..
      dockerfile: services/api/Dockerfile
    environment:
      ENVIRONMENT: dev
      DATABASE_URL: postgresql+asyncpg://lumen_app:lumen_app@db:5432/lumen
      DATABASE_MIGRATION_URL: postgresql+asyncpg://lumen_migrator:lumen_migrator@db:5432/lumen
      REDIS_URL: redis://redis:6379/0
      STORAGE_ENDPOINT: http://storage:9000
      CORS_ORIGINS: '["http://localhost:3000"]'
    ports: ["8000:8000"]
    depends_on:
      db: { condition: service_healthy }
      redis: { condition: service_healthy }
```

- [ ] **Step 5: Write the Makefile**

Create `Makefile` at the repository root:

```makefile
COMPOSE := docker compose -f infra/docker-compose.yml

.PHONY: dev down migrate test test-engine test-api test-integration fmt

dev:
	$(COMPOSE) up -d --build
	@echo "API   http://localhost:8000/docs"
	@echo "MinIO http://localhost:9001"

down:
	$(COMPOSE) down

migrate:
	uv run --directory services/api alembic upgrade head

test: test-engine test-api

test-engine:
	uv run --directory engine pytest tests -q

test-api:
	uv run --directory services/api pytest tests -q

test-integration:
	uv run --directory services/api pytest tests -m integration -q

fmt:
	uv run --directory services/api ruff format src tests
	uv run --directory engine ruff format src tests
```

- [ ] **Step 6: Remove the superseded root compose files**

```bash
git rm docker-compose.yml Dockerfile .dockerignore
```

- [ ] **Step 7: Bring the stack up and run the integration test**

```bash
make dev
make migrate
uv run --directory services/api pytest tests/test_smoke_container.py -m integration -v
```
Expected: PASS — 2 passed.
If `readyz` returns 500, check `docker compose -f infra/docker-compose.yml logs api` — the usual cause is the `lumen` database not existing because the init SQL only runs on a fresh volume. Fix with `docker compose -f infra/docker-compose.yml down -v` then `make dev`.

- [ ] **Step 8: Commit**

```bash
git add Makefile services/api/Dockerfile infra
git commit -m "chore: containerize the api and add developer entry points"
```

---

## Self-Review

**Spec coverage** — ADR-0001 topology: Tasks 1, 2, 7 (monorepo, API, compose; worker deferred to the agent-layer plan, which is where its first job type appears). ADR-0002 tenancy: Tasks 3, 5, 6 (orgs, memberships, RLS, encrypted DSN column present and never serialized). Envelope encryption of `dsn_encrypted` is **deliberately deferred** — the column exists and is excluded from `DataSourceOut`, but no external DSN can be created until connectors ship in a later plan, so nothing unencrypted can be written.

**Placeholder scan** — every step carries runnable code or an exact command. The one prose-only instruction (Task 4, Step 3) is an explicit warning about a mangled identifier, not a deferred decision.

**Type consistency** — `SessionData(user_id, org_id)` is produced in Task 4 and consumed unchanged in Task 5. `current_org_id -> UUID` is produced in Task 5 and consumed in Task 6. `DataSource` fields used in Task 6 (`org_id, name, kind, table_name, row_count, status, created_at`) all exist in the Task 3 model. `Organization.plan_code` is defined in Task 3 and read by `MeResponse` in Task 5.

**Known follow-ups for later plans** — `lumen_api/db/models/source.py` gains `object_uri` here but nothing writes it until the upload endpoint in the agent-layer plan; `require_role` is defined in Task 5 and first exercised in Task 6.
