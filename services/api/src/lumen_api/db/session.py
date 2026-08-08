"""Database sessions against Supabase Postgres.

The API and the worker connect with the service role, which by itself bypasses
nothing but *is* the table owner — so every tenant statement must be run as
`authenticated` with the caller's JWT claims applied, exactly as PostgREST does
it. That is what makes `auth.uid()` resolve and every RLS policy fire.

Three entry points, and the names are the whole safety story:

    user_session(user_id)   tenant work on behalf of a signed-in person.
                            RLS applies. Use this for ~everything.

    service_session()       no identity, no RLS. Signup bootstrap and
                            cross-tenant maintenance only. Every call site is
                            in this module's docstring or it is a bug.

    engine                  raw connections for health checks.

    tenant_session(org_id)  lives in lumen_api.tenant_db, not this module,
                            because it connects to a different instance
                            entirely. Customer data only; it can reach no
                            table defined in this database.

`SET LOCAL` is deliberate: the setting dies with the transaction, so a pooled
connection can never hand one tenant's identity to the next borrower. This is
also why the DSN must point at Supabase's *session* pooler (5432) and not the
transaction pooler (6543), which does not carry `SET LOCAL` across statements.

Legitimate `service_session()` call sites:
  * `lumen_api.orgs.bootstrap` — reading an org for a user who has just signed
    up and may not yet have a membership row visible to themselves.
  * `lumen_worker.jobs.*` — a job runs with no HTTP request behind it; it
    re-establishes identity with `user_session(run.created_by)` before touching
    tenant rows, and only uses the service session to look the run up.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from lumen_api.settings import get_settings

_engines: dict[int, AsyncEngine] = {}


def get_engine() -> AsyncEngine:
    """The engine for the running event loop.

    Cached per loop rather than per process. An asyncpg connection belongs to the
    loop that opened it, so a single module-level pool works in production — one
    loop for the process lifetime — and breaks under pytest-asyncio, where each
    test gets a fresh loop and inherits pooled connections bound to a closed one.
    That surfaced as tests passing alone and failing together, which is the most
    expensive kind of failure to debug.
    """
    settings = get_settings()
    try:
        key = id(asyncio.get_running_loop())
    except RuntimeError:  # no loop yet — the engine will be built on first use
        key = 0

    engine = _engines.get(key)
    if engine is None:
        engine = create_async_engine(
            settings.database_url,
            pool_pre_ping=True,
            pool_size=10,
            max_overflow=5,
            # Supabase closes idle server-side connections; recycle before it does.
            pool_recycle=1800,
        )
        _engines[key] = engine
    return engine


def _session_factory() -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(get_engine(), expire_on_commit=False, class_=AsyncSession)


async def dispose_engines() -> None:
    """Close every pool. Call on shutdown, and between tests."""
    for engine in list(_engines.values()):
        await engine.dispose()
    _engines.clear()


@asynccontextmanager
async def user_session(user_id: uuid.UUID) -> AsyncIterator[AsyncSession]:
    """Yield a session in which RLS sees `user_id` as `auth.uid()`."""
    claims = json.dumps({"sub": str(user_id), "role": "authenticated"})

    async with _session_factory()() as session:
        await session.begin()
        # Both forms: older auth.uid() reads request.jwt.claim.sub, newer reads
        # the request.jwt.claims JSON. Setting both keeps this working across
        # Supabase platform upgrades.
        await session.execute(text("SET LOCAL ROLE authenticated"))
        await session.execute(
            text("SELECT set_config('request.jwt.claims', :claims, true)"),
            {"claims": claims},
        )
        await session.execute(
            text("SELECT set_config('request.jwt.claim.sub', :sub, true)"),
            {"sub": str(user_id)},
        )
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


@asynccontextmanager
async def service_session() -> AsyncIterator[AsyncSession]:
    """Yield a session with no user identity. RLS does not constrain it.

    Read this module's docstring before adding a call site.
    """
    async with _session_factory()() as session:
        async with session.begin():
            yield session


async def ping() -> bool:
    async with get_engine().connect() as connection:
        await connection.execute(text("SELECT 1"))
    return True
