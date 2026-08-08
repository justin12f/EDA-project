"""The tenant Postgres instance.

Separate from Supabase on purpose (spec D1). The control plane —
organizations, memberships, subscriptions, api_keys, proposals — stays
there; this instance holds nothing but customer data. The security argument
is structural rather than permissive: an agent writing DDL here has no
connection to a control-plane table, so the blast radius is bounded by what
the instance *contains*, not by what a REVOKE forbids.

Per-org roles are still required. They solve a different problem: keeping
tenants out of each other's schemas *within* this instance.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from lumen_api.settings import get_settings

_engines: dict[int, AsyncEngine] = {}


def tenant_schema_name(org_id: uuid.UUID) -> str:
    """`tenant_<32 hex>` — 39 bytes, comfortably inside Postgres's 63."""
    return f"tenant_{org_id.hex}"


def tenant_raw_schema_name(org_id: uuid.UUID) -> str:
    """Staging. Permanent, not a temporary buffer: the raw-data browser
    reads it while a schema is awaiting review, and a failed promotion must
    be retryable without re-downloading the origin."""
    return f"{tenant_schema_name(org_id)}_raw"


def tenant_role_name(org_id: uuid.UUID) -> str:
    return f"{tenant_schema_name(org_id)}_role"


def get_tenant_engine() -> AsyncEngine:
    """Cached per event loop, for the reason `db/session.py` documents at
    length: an asyncpg connection belongs to the loop that opened it, and a
    module-level pool works in production but breaks under pytest-asyncio,
    which gives every test a fresh loop.
    """
    settings = get_settings()
    if not settings.has_tenant_db:
        raise RuntimeError(
            "TENANT_DATABASE_URL is not configured — the Data Architect is disabled."
        )

    try:
        key = id(asyncio.get_running_loop())
    except RuntimeError:
        key = 0

    engine = _engines.get(key)
    if engine is None:
        engine = create_async_engine(
            settings.tenant_database_url.get_secret_value(),
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=5,
            pool_recycle=1800,
        )
        _engines[key] = engine
    return engine


async def dispose_tenant_engines() -> None:
    for engine in list(_engines.values()):
        await engine.dispose()
    _engines.clear()


@asynccontextmanager
async def tenant_session(org_id: uuid.UUID) -> AsyncIterator[AsyncSession]:
    """A session scoped to one org's schemas by Postgres grants.

    Cannot write `proposals`, `artifact_dependencies` or anything else in
    the control plane — those are on a different instance, so no transaction
    spans both. Callers that need both use two sessions in sequence and
    write the control-plane record LAST (spec §3.2).
    """
    schema = tenant_schema_name(org_id)
    factory = async_sessionmaker(get_tenant_engine(), expire_on_commit=False, class_=AsyncSession)

    async with factory() as session:
        await session.begin()
        await session.execute(text(f'SET LOCAL ROLE "{tenant_role_name(org_id)}"'))
        await session.execute(
            text(f'SET LOCAL search_path = "{schema}", "{tenant_raw_schema_name(org_id)}"')
        )
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def ensure_tenant_schema(org_id: uuid.UUID) -> None:
    """Create this org's role and schemas if they are missing.

    Lazy and idempotent rather than part of signup, because orgs already
    exist without these objects and a migration cannot create a role per
    future org. Runs at the head of every ingestion job. Safe under
    concurrency: every statement is IF NOT EXISTS or wrapped in the
    duplicate_object guard every migration in this repo already uses.
    """
    schema = tenant_schema_name(org_id)
    raw = tenant_raw_schema_name(org_id)
    role = tenant_role_name(org_id)

    engine = get_tenant_engine()
    async with engine.begin() as conn:
        await conn.execute(
            text(
                f"DO $$ BEGIN CREATE ROLE \"{role}\" NOLOGIN; "
                f"EXCEPTION WHEN duplicate_object THEN NULL; END $$"
            )
        )
        for name in (schema, raw):
            await conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{name}" AUTHORIZATION "{role}"'))
            await conn.execute(text(f'GRANT ALL ON SCHEMA "{name}" TO "{role}"'))
        # The connecting role must be able to SET ROLE to it.
        await conn.execute(text(f'GRANT "{role}" TO CURRENT_USER'))
