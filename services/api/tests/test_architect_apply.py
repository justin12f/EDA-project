"""Applying a schema: DDL runs, then the control-plane record."""

from __future__ import annotations

import uuid

import httpx
import polars as pl
import pytest
import pytest_asyncio
from sqlalchemy import text

from lumen_api.architect import apply_schema, design_schema, propose_schema
from lumen_api.auth.dependencies import Identity
from lumen_api.db.session import user_session
from lumen_api.proposals import DecisionRequest, decide_proposal
from lumen_api.settings import get_settings
from lumen_api.tenant_db import (
    ensure_tenant_schema,
    raw_table_name,
    tenant_raw_schema_name,
    tenant_schema_name,
    tenant_session,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not get_settings().has_tenant_db, reason="TENANT_DATABASE_URL is not configured"
    ),
]

# identity, _create_user, _delete_user, _identity_of and _stage are copied
# verbatim from test_architect_design.py. A `from tests.test_architect_design
# import ...` was tried first — `services/api/tests` has no `__init__.py`, so
# pytest's own "pythonpath = [src]" config never puts `tests` on sys.path as
# a package, and the import fails with `ModuleNotFoundError: No module named
# 'tests'`. The copy is the point, not a workaround for the copy.


def _admin_headers() -> dict[str, str]:
    key = get_settings().supabase_service_role_key.get_secret_value()
    return {"apikey": key, "Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _create_user() -> uuid.UUID:
    settings = get_settings()
    response = httpx.post(
        f"{settings.supabase_url}/auth/v1/admin/users",
        headers=_admin_headers(),
        json={
            "email": f"lumen-arch-{uuid.uuid4().hex[:12]}@example.com",
            "password": uuid.uuid4().hex + "Aa1!",
            "email_confirm": True,
            "user_metadata": {"display_name": "Architect Tester", "org_name": "Architect Org"},
        },
        timeout=30,
    )
    response.raise_for_status()
    return uuid.UUID(response.json()["id"])


def _delete_user(user_id: uuid.UUID) -> None:
    settings = get_settings()
    httpx.delete(
        f"{settings.supabase_url}/auth/v1/admin/users/{user_id}",
        headers=_admin_headers(),
        timeout=30,
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


async def _stage(identity: Identity, table: str, frame: pl.DataFrame) -> uuid.UUID:
    """Put a frame in staging and register a data_sources row for it."""
    await ensure_tenant_schema(identity.org_id)
    source_id = uuid.uuid4()
    async with user_session(identity.user_id) as db:
        await db.execute(
            text(
                "insert into public.data_sources (id, org_id, name, kind, status, table_name) "
                "values (:id, :org, :name, 'csv', 'idle', :table)"
            ),
            {"id": source_id, "org": identity.org_id, "name": f"{table}.csv", "table": table},
        )
    dsn = get_settings().tenant_database_url.get_secret_value()
    frame.write_database(
        table_name=f"{tenant_raw_schema_name(identity.org_id)}.{raw_table_name(source_id, table)}",
        connection=dsn,
        if_table_exists="replace",
    )
    return source_id


async def _columns(org_id: uuid.UUID, table: str) -> dict[str, str]:
    async with tenant_session(org_id) as db:
        rows = (
            await db.execute(
                text(
                    "select column_name, data_type from information_schema.columns "
                    "where table_schema = :schema and table_name = :table"
                ),
                {"schema": tenant_schema_name(org_id), "table": table},
            )
        ).mappings().all()
    return {r["column_name"]: r["data_type"] for r in rows}


async def test_applying_creates_the_table_with_real_types(identity):
    source_id = await _stage(
        identity, "customers", pl.DataFrame({"id": ["c1"], "balance": [1.5]})
    )
    spec = await design_schema(identity.org_id, identity.user_id, source_id)
    await apply_schema(identity.org_id, identity.user_id, spec)

    columns = await _columns(identity.org_id, "customers")
    assert columns["id"] == "text"
    assert columns["balance"] == "double precision"


async def test_the_primary_key_is_really_enforced(identity):
    source_id = await _stage(identity, "customers", pl.DataFrame({"id": ["c1", "c2"]}))
    spec = await design_schema(identity.org_id, identity.user_id, source_id)
    await apply_schema(identity.org_id, identity.user_id, spec)

    with pytest.raises(Exception):
        async with tenant_session(identity.org_id) as db:
            await db.execute(text("insert into customers (id) values ('x'), ('x')"))


async def test_the_foreign_key_is_really_enforced(identity):
    await _stage(identity, "customers", pl.DataFrame({"id": ["c1"]}))
    orders_id = await _stage(
        identity, "orders", pl.DataFrame({"id": ["o1"], "customer_id": ["c1"]})
    )
    spec = await design_schema(identity.org_id, identity.user_id, orders_id)
    await apply_schema(identity.org_id, identity.user_id, spec)

    with pytest.raises(Exception) as caught:
        async with tenant_session(identity.org_id) as db:
            await db.execute(
                text("insert into orders (id, customer_id) values ('o9', 'ghost')")
            )
    assert "foreign key" in str(caught.value).lower()


async def test_accepting_the_proposal_applies_it(identity):
    source_id = await _stage(identity, "customers", pl.DataFrame({"id": ["c1"]}))
    spec = await design_schema(identity.org_id, identity.user_id, source_id)
    proposal_id = await propose_schema(identity.org_id, identity.user_id, source_id, spec)

    result = await decide_proposal(proposal_id, DecisionRequest(decision="accept"), identity)
    assert result["status"] == "applied"
    assert await _columns(identity.org_id, "customers")


async def test_row_count_is_written(identity):
    """data_sources.row_count is SELECTed by three read paths and written by
    none — permanently NULL until now."""
    source_id = await _stage(
        identity, "customers", pl.DataFrame({"id": ["c1", "c2", "c3"]})
    )
    spec = await design_schema(identity.org_id, identity.user_id, source_id)
    await apply_schema(identity.org_id, identity.user_id, spec)

    async with user_session(identity.user_id) as db:
        count = (
            await db.execute(
                text("select row_count from public.data_sources where id = :id"),
                {"id": source_id},
            )
        ).scalar_one()
    assert count == 3
