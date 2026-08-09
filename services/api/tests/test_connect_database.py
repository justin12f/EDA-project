"""Connecting a customer database: structure first, data on demand (D10)."""

from __future__ import annotations

import uuid

import httpx
import pytest
import pytest_asyncio
from sqlalchemy import text

from lumen_api.auth.dependencies import Identity
from lumen_api.db.session import user_session
from lumen_api.settings import get_settings
from lumen_api.sources_db import DatabaseSourceCreate, connect_database, list_source_tables
from lumen_api.tenant_db import get_tenant_engine

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not get_settings().has_tenant_db, reason="TENANT_DATABASE_URL is not configured"
    ),
]

# identity is copied from test_architect_design.py — a cross-file import
# doesn't work in this package (see the note there).


def _admin_headers() -> dict[str, str]:
    key = get_settings().supabase_service_role_key.get_secret_value()
    return {"apikey": key, "Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _create_user() -> uuid.UUID:
    settings = get_settings()
    response = httpx.post(
        f"{settings.supabase_url}/auth/v1/admin/users",
        headers=_admin_headers(),
        json={
            "email": f"lumen-connectdb-{uuid.uuid4().hex[:12]}@example.com",
            "password": uuid.uuid4().hex + "Aa1!",
            "email_confirm": True,
            "user_metadata": {"display_name": "Connect DB Tester", "org_name": "Connect DB Org"},
        },
        timeout=30,
    )
    response.raise_for_status()
    return uuid.UUID(response.json()["id"])


def _delete_user(user_id: uuid.UUID) -> None:
    settings = get_settings()
    httpx.delete(
        f"{settings.supabase_url}/auth/v1/admin/users/{user_id}", headers=_admin_headers(), timeout=30
    )


async def _identity_of(user_id: uuid.UUID) -> Identity:
    async with user_session(user_id) as db:
        row = (await db.execute(text("select * from public.current_identity()"))).mappings().first()
    return Identity(
        user_id=row["user_id"], email=row["email"], display_name=row["display_name"],
        avatar_url=row["avatar_url"], org_id=row["org_id"], org_name=row["org_name"],
        org_slug=row["org_slug"], plan_code=row["plan_code"], role=str(row["role"]),
    )


@pytest_asyncio.fixture
async def identity():
    user_id = _create_user()
    try:
        yield await _identity_of(user_id)
    finally:
        _delete_user(user_id)


@pytest.fixture
async def customer_database():
    """A schema on the tenant instance standing in for a customer's own
    database — the adapter cannot tell the difference."""
    name = f"customer_db_{uuid.uuid4().hex[:8]}"
    engine = get_tenant_engine()
    async with engine.begin() as conn:
        await conn.execute(text(f'CREATE SCHEMA "{name}"'))
        await conn.execute(text(f'CREATE TABLE "{name}".people (id text PRIMARY KEY, nm text)'))
        await conn.execute(text(f'CREATE TABLE "{name}".huge (id bigint PRIMARY KEY)'))
        await conn.execute(text(f"INSERT INTO \"{name}\".people VALUES ('p1', 'Ada')"))
    try:
        yield name
    finally:
        async with engine.begin() as conn:
            await conn.execute(text(f'DROP SCHEMA "{name}" CASCADE'))


async def test_connecting_stores_the_dsn_encrypted(identity, customer_database):
    dsn = get_settings().tenant_database_url.get_secret_value()
    result = await connect_database(
        DatabaseSourceCreate(name="CRM", kind="postgres", dsn=dsn, schema=customer_database),
        identity,
    )

    async with user_session(identity.user_id) as db:
        stored = (
            await db.execute(
                text("select dsn_encrypted from public.data_sources where id = :id"),
                {"id": uuid.UUID(result["id"])},
            )
        ).scalar_one()
    assert stored.startswith("v1:")
    assert "postgres" not in stored


async def test_the_dsn_is_never_returned(identity, customer_database):
    dsn = get_settings().tenant_database_url.get_secret_value()
    result = await connect_database(
        DatabaseSourceCreate(name="CRM", kind="postgres", dsn=dsn, schema=customer_database),
        identity,
    )
    assert "dsn" not in result
    assert dsn not in str(result)


async def test_structure_is_mirrored_without_copying_data(identity, customer_database):
    """The diagram appears in seconds; no bytes move until a table is
    selected. That is the whole argument for D10."""
    dsn = get_settings().tenant_database_url.get_secret_value()
    created = await connect_database(
        DatabaseSourceCreate(name="CRM", kind="postgres", dsn=dsn, schema=customer_database),
        identity,
    )

    tables = await list_source_tables(uuid.UUID(created["id"]), identity)
    names = {t["name"] for t in tables["tables"]}
    assert names == {"people", "huge"}
    assert all(t["imported"] is False for t in tables["tables"])


async def test_the_real_primary_key_is_reported(identity, customer_database):
    dsn = get_settings().tenant_database_url.get_secret_value()
    created = await connect_database(
        DatabaseSourceCreate(name="CRM", kind="postgres", dsn=dsn, schema=customer_database),
        identity,
    )
    tables = await list_source_tables(uuid.UUID(created["id"]), identity)
    people = next(t for t in tables["tables"] if t["name"] == "people")
    assert people["primary_key"] == ["id"]
    assert people["declared"] is True
