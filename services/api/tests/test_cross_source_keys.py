"""Testing items 11 and 13: a foreign key that crosses between two of the
customer's own databases, and the layout choice round-tripping.

This is the test that proves D11 is real rather than decorative. Postgres
enforces a foreign key across schemas within a database and never across
databases — putting every source of an org in one database is precisely
what makes this possible.
"""

from __future__ import annotations

import uuid

import httpx
import polars as pl
import pytest
import pytest_asyncio
from sqlalchemy import text

from lumen_api.architect import apply_schema, design_schema
from lumen_api.auth.dependencies import Identity
from lumen_api.db.session import user_session
from lumen_api.settings import get_settings
from lumen_api.tenant_db import ensure_tenant_schema, raw_table_name, tenant_raw_schema_name, tenant_session

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not get_settings().has_tenant_db, reason="TENANT_DATABASE_URL is not configured"
    ),
]

# identity and _stage are copied from test_architect_design.py — a
# cross-file import doesn't work in this package (see the note there).


def _admin_headers() -> dict[str, str]:
    key = get_settings().supabase_service_role_key.get_secret_value()
    return {"apikey": key, "Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _create_user() -> uuid.UUID:
    settings = get_settings()
    response = httpx.post(
        f"{settings.supabase_url}/auth/v1/admin/users",
        headers=_admin_headers(),
        json={
            "email": f"lumen-crosskey-{uuid.uuid4().hex[:12]}@example.com",
            "password": uuid.uuid4().hex + "Aa1!",
            "email_confirm": True,
            "user_metadata": {"display_name": "Cross Key Tester", "org_name": "Cross Key Org"},
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


async def test_a_key_spans_two_separately_connected_sources(identity):
    """`customers` arrives from one source and `orders` from another. The
    relationship between them must be enforced, not merely drawn."""
    await _stage(identity, "customers", pl.DataFrame({"id": ["c1", "c2"]}))
    orders_id = await _stage(
        identity, "orders", pl.DataFrame({"id": ["o1"], "customer_id": ["c1"]})
    )

    spec = await design_schema(identity.org_id, identity.user_id, orders_id)
    await apply_schema(identity.org_id, identity.user_id, spec)

    key = next(k for k in spec.foreign_keys if k.from_table == "orders")
    assert key.to_table == "customers"
    assert key.enforced is True
    # The two tables came from two different data_sources rows.
    assert {t.source_id for t in spec.tables} != {orders_id}

    with pytest.raises(Exception) as caught:
        async with tenant_session(identity.org_id) as db:
            await db.execute(
                text("insert into orders (id, customer_id) values ('o2', 'ghost')")
            )
    assert "foreign key" in str(caught.value).lower()


async def test_two_sources_with_the_same_table_name_do_not_overwrite(identity):
    await _stage(identity, "users", pl.DataFrame({"id": ["a"]}))
    second = await _stage(identity, "users", pl.DataFrame({"id": ["b"], "extra": [1]}))

    spec = await design_schema(identity.org_id, identity.user_id, second)
    names = {t.name for t in spec.tables}
    assert len(names) == 2, f"one source overwrote the other: {names}"


async def test_the_layout_is_recorded_on_the_spec(identity):
    source_id = await _stage(identity, "customers", pl.DataFrame({"id": ["c1"]}))
    spec = await design_schema(identity.org_id, identity.user_id, source_id)
    assert spec.layout == "merged"
