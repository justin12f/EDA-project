"""design_schema against the live instances."""

from __future__ import annotations

import uuid

import httpx
import polars as pl
import pytest
import pytest_asyncio
from sqlalchemy import text

from lumen_api.architect import design_schema
from lumen_api.auth.dependencies import Identity
from lumen_api.db.session import user_session
from lumen_api.settings import get_settings
from lumen_api.tenant_db import ensure_tenant_schema, tenant_raw_schema_name, tenant_session

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not get_settings().has_tenant_db, reason="TENANT_DATABASE_URL is not configured"
    ),
]


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
        table_name=f"{tenant_raw_schema_name(identity.org_id)}.{table}",
        connection=dsn,
        if_table_exists="replace",
    )
    return source_id


async def test_a_single_source_gets_typed_columns_and_a_primary_key(identity):
    source_id = await _stage(
        identity, "customers", pl.DataFrame({"id": ["c1", "c2"], "balance": [1.5, 2.5]})
    )
    spec = await design_schema(identity.org_id, identity.user_id, source_id)

    table = next(t for t in spec.tables if t.name == "customers")
    assert table.primary_key == ("id",)
    assert table.pk_rationale
    assert {c.name for c in table.columns} == {"id", "balance"}


async def test_a_cross_source_relationship_is_detected(identity):
    await _stage(identity, "customers", pl.DataFrame({"id": ["c1", "c2"]}))
    orders_id = await _stage(
        identity, "orders", pl.DataFrame({"id": ["o1", "o2"], "customer_id": ["c1", "c2"]})
    )
    spec = await design_schema(identity.org_id, identity.user_id, orders_id)

    keys = [k for k in spec.foreign_keys if k.from_table == "orders"]
    assert len(keys) == 1
    assert (keys[0].to_table, keys[0].to_column) == ("customers", "id")
    assert keys[0].enforced is True


async def test_column_names_are_sanitised(identity):
    source_id = await _stage(
        identity, "weird", pl.DataFrame({"Customer ID": ["a"], "2024 Revenue": [1.0]})
    )
    spec = await design_schema(identity.org_id, identity.user_id, source_id)
    names = {c.name for c in spec.tables[0].columns}
    assert names == {"customer_id", "col_2024_revenue"}


async def test_the_returned_spec_validates(identity):
    source_id = await _stage(identity, "customers", pl.DataFrame({"id": ["c1"]}))
    spec = await design_schema(identity.org_id, identity.user_id, source_id)
    spec.validate()
