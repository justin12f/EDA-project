"""resolve() reads SQL now. The contract is unchanged, which is the point:
ADR-0008 through ADR-0013 all consume a DataFrame and none of them change."""

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
from lumen_api.tenant_db import ensure_tenant_schema, tenant_raw_schema_name

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not get_settings().has_tenant_db, reason="TENANT_DATABASE_URL is not configured"
    ),
]

# identity and _stage are copied from test_architect_design.py — a
# cross-file import doesn't work in this package (see the note there).
# ingest_to_staging itself lives in lumen_worker, which services/api cannot
# import (the dependency runs the other way: lumen-worker depends on
# lumen-api, not vice versa) — _stage() writes straight to the raw schema,
# the same shortcut test_architect_design.py and test_architect_apply.py
# already take for this exact reason.


def _admin_headers() -> dict[str, str]:
    key = get_settings().supabase_service_role_key.get_secret_value()
    return {"apikey": key, "Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _create_user() -> uuid.UUID:
    settings = get_settings()
    response = httpx.post(
        f"{settings.supabase_url}/auth/v1/admin/users",
        headers=_admin_headers(),
        json={
            "email": f"lumen-substrate-{uuid.uuid4().hex[:12]}@example.com",
            "password": uuid.uuid4().hex + "Aa1!",
            "email_confirm": True,
            "user_metadata": {"display_name": "Substrate Tester", "org_name": "Substrate Org"},
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
        table_name=f"{tenant_raw_schema_name(identity.org_id)}.{table}",
        connection=dsn,
        if_table_exists="replace",
    )
    return source_id


@pytest_asyncio.fixture
async def applied_customers(identity):
    source_id = await _stage(identity, "customers", pl.DataFrame({"id": ["c1", "c2"]}))
    spec = await design_schema(identity.org_id, identity.user_id, source_id)
    await apply_schema(identity.org_id, identity.user_id, spec)
    return source_id


async def test_resolve_returns_the_applied_table(identity, applied_customers):
    from lumen_api.datasets.store import HandleStore

    # The plan's own sketch of this test called latest_for_source() without
    # ever creating a handle — dataset_handles rows only exist once
    # something calls put(), the way agents/registry.py's read_source tool
    # does with source_id set. Reproducing that here: apply_schema() alone
    # writes nothing to dataset_handles.
    store = HandleStore(identity.org_id, identity.user_id)
    await store.put(
        pl.DataFrame({"id": ["c1", "c2"]}), label="customers", source_id=applied_customers
    )

    handle = await store.latest_for_source(applied_customers)
    frame = await store.resolve(handle.rid)

    materialised = frame.collect() if hasattr(frame, "collect") else frame
    assert materialised.height > 0


async def test_the_orphan_modules_are_gone():
    with pytest.raises(ModuleNotFoundError):
        __import__("lumen.database.postgres_manager")
    with pytest.raises(ModuleNotFoundError):
        __import__("lumen.agents.postgres_admin_agent")
