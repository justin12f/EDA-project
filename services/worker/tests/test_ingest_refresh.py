"""Re-ingest. A file is a snapshot, so the table becomes exactly the file."""

from __future__ import annotations

import asyncio
import uuid

import httpx
import pytest
import pytest_asyncio
from sqlalchemy import text

from lumen_api.architect import apply_schema, design_schema
from lumen_api.auth.dependencies import Identity
from lumen_api.datasets.store import SupabaseStorage
from lumen_api.db.session import user_session
from lumen_api.settings import get_settings
from lumen_api.tenant_db import tenant_session
from lumen_worker.ingest import ingest_to_staging, refresh_source

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not get_settings().has_tenant_db, reason="TENANT_DATABASE_URL is not configured"
    ),
]

# identity, _FakeRedis, _seed and the applied-schema setup are copied from
# tests/test_ingest.py — a cross-file `from tests.test_ingest import ...`
# fails the same way it did for test_architect_apply.py: this package has
# no __init__.py, so pytest's pythonpath=[src] never makes `tests` an
# importable package.


def _admin_headers() -> dict[str, str]:
    key = get_settings().supabase_service_role_key.get_secret_value()
    return {"apikey": key, "Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _create_user() -> uuid.UUID:
    settings = get_settings()
    response = httpx.post(
        f"{settings.supabase_url}/auth/v1/admin/users",
        headers=_admin_headers(),
        json={
            "email": f"lumen-refresh-{uuid.uuid4().hex[:12]}@example.com",
            "password": uuid.uuid4().hex + "Aa1!",
            "email_confirm": True,
            "user_metadata": {"display_name": "Refresh Tester", "org_name": "Refresh Org"},
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
        user_id=row["user_id"],
        email=row["email"],
        display_name=row["display_name"],
        avatar_url=row["avatar_url"],
        org_id=row["org_id"],
        org_name=row["org_name"],
        org_slug=row["org_slug"],
        plan_code=row["plan_code"],
        role=str(row["role"]),
    )


@pytest_asyncio.fixture
async def identity():
    user_id = _create_user()
    try:
        yield await _identity_of(user_id)
    finally:
        _delete_user(user_id)


class _FakeRedis:
    def __init__(self) -> None:
        self.jobs: list[tuple] = []

    async def enqueue_job(self, *args, **kwargs) -> None:
        self.jobs.append(args)


async def _seed_source(identity: Identity, name: str, content: bytes) -> tuple[uuid.UUID, str]:
    path = f"org/{identity.org_id}/uploads/{uuid.uuid4().hex}{name[name.rfind('.'):]}"
    await SupabaseStorage().upload(path, content, "text/csv")
    source_id = uuid.uuid4()
    async with user_session(identity.user_id) as db:
        await db.execute(
            text(
                "insert into public.data_sources "
                "(id, org_id, name, kind, status, object_path) "
                "values (:id, :org, :name, 'csv', 'idle', :path)"
            ),
            {"id": source_id, "org": identity.org_id, "name": name, "path": path},
        )
    return source_id, path


async def _reupload(path: str, content: bytes) -> None:
    """Supabase Storage's overwrite is not immediately read-your-writes — a
    download right after an `x-upsert` upload can still return the previous
    version for a moment. Poll until the object actually is the one just
    written, same as test_sentinel_diagnosis.py::_upload_and_wait."""
    storage = SupabaseStorage()
    await storage.upload(path, content, "text/csv")
    for _ in range(150):
        if await storage.download(path) == content:
            return
        await asyncio.sleep(1)
    raise TimeoutError(f"Storage never converged on the new content at {path}")


@pytest_asyncio.fixture
async def applied_customers(identity):
    source_id, path = await _seed_source(identity, "customers.csv", b"id\nc1\nc2\n")
    ctx = {"redis": _FakeRedis()}
    await ingest_to_staging(ctx, str(source_id), str(identity.org_id), str(identity.user_id))
    spec = await design_schema(identity.org_id, identity.user_id, source_id)
    await apply_schema(identity.org_id, identity.user_id, spec)
    return source_id, path


@pytest_asyncio.fixture
async def applied_pair(identity):
    customers_id, customers_path = await _seed_source(identity, "customers.csv", b"id\nc1\n")
    orders_id, orders_path = await _seed_source(
        identity, "orders.csv", b"id,customer_id\no1,c1\n"
    )
    ctx = {"redis": _FakeRedis()}
    await ingest_to_staging(ctx, str(customers_id), str(identity.org_id), str(identity.user_id))
    await ingest_to_staging(ctx, str(orders_id), str(identity.org_id), str(identity.user_id))
    spec = await design_schema(identity.org_id, identity.user_id, orders_id)
    await apply_schema(identity.org_id, identity.user_id, spec)
    return customers_id, customers_path, orders_id


async def test_a_replaced_snapshot_drops_rows_deleted_at_origin(identity, applied_customers):
    """An upsert would keep 'c2' forever. A file is a full snapshot, so a
    row the customer deleted at origin must disappear here too."""
    source_id, path = applied_customers  # seeded with c1,c2 and applied
    await _reupload(path, b"id\nc1\n")

    await refresh_source(
        {"redis": _FakeRedis()}, str(source_id), str(identity.org_id), str(identity.user_id)
    )

    async with tenant_session(identity.org_id) as db:
        ids = (await db.execute(text("select id from customers order by id"))).scalars().all()
    assert ids == ["c1"]


async def test_replacing_a_parent_does_not_violate_a_child_foreign_key(identity, applied_pair):
    """The reason foreign keys are DEFERRABLE and the replace runs inside
    one transaction: clearing customers while orders still references it
    would fail immediately otherwise."""
    customers_id, customers_path, _ = applied_pair
    await _reupload(customers_path, b"id\nc1\nc2\n")

    result = await refresh_source(
        {"redis": _FakeRedis()}, str(customers_id), str(identity.org_id), str(identity.user_id)
    )
    assert result["status"] == "refreshed"


async def test_a_violation_leaves_the_table_intact_and_records_drift(identity, applied_pair):
    """D5: an orphan row is a data-quality finding, not a crash. The
    transaction rolls back, so the table still holds what it held."""
    customers_id, customers_path, _ = applied_pair
    await _reupload(customers_path, b"id\nc9\n")  # orphans every order

    result = await refresh_source(
        {"redis": _FakeRedis()}, str(customers_id), str(identity.org_id), str(identity.user_id)
    )
    assert result["status"] == "constraint_violation"

    async with tenant_session(identity.org_id) as db:
        ids = (await db.execute(text("select id from customers order by id"))).scalars().all()
    assert "c1" in ids

    async with user_session(identity.user_id) as db:
        kinds = (
            await db.execute(
                text(
                    "select kind from public.drift_events "
                    "where source_id = :id order by occurred_at desc limit 1"
                ),
                {"id": customers_id},
            )
        ).scalars().all()
    assert kinds == ["schema_constraint"]
