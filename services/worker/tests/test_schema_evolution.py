"""Evolution. Reversible changes may earn autonomy; destructive ones never do."""

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
from lumen_api.tenant_db import tenant_schema_name, tenant_session
from lumen_worker.ingest import evolve_schema, ingest_to_staging

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not get_settings().has_tenant_db, reason="TENANT_DATABASE_URL is not configured"
    ),
]

# identity, _FakeRedis, _seed_source and _reupload are copied from
# test_ingest_refresh.py — see the note there on why a cross-file import
# doesn't work in this package.


def _admin_headers() -> dict[str, str]:
    key = get_settings().supabase_service_role_key.get_secret_value()
    return {"apikey": key, "Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _create_user() -> uuid.UUID:
    settings = get_settings()
    response = httpx.post(
        f"{settings.supabase_url}/auth/v1/admin/users",
        headers=_admin_headers(),
        json={
            "email": f"lumen-evolve-{uuid.uuid4().hex[:12]}@example.com",
            "password": uuid.uuid4().hex + "Aa1!",
            "email_confirm": True,
            "user_metadata": {"display_name": "Evolve Tester", "org_name": "Evolve Org"},
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
    """Supabase Storage's overwrite is not immediately read-your-writes —
    poll until the object actually is the one just written."""
    storage = SupabaseStorage()
    await storage.upload(path, content, "text/csv")
    for _ in range(150):
        if await storage.download(path) == content:
            return
        await asyncio.sleep(1)
    raise TimeoutError(f"Storage never converged on the new content at {path}")


async def _apply(identity: Identity, source_id: uuid.UUID) -> None:
    ctx = {"redis": _FakeRedis()}
    await ingest_to_staging(ctx, str(source_id), str(identity.org_id), str(identity.user_id))
    spec = await design_schema(identity.org_id, identity.user_id, source_id)
    await apply_schema(identity.org_id, identity.user_id, spec)


@pytest_asyncio.fixture
async def applied_customers(identity):
    """A single-column `customers` table — an `email` column added later is
    purely additive."""
    source_id, path = await _seed_source(identity, "customers.csv", b"id\nc1\n")
    await _apply(identity, source_id)
    return source_id, path


@pytest_asyncio.fixture
async def applied_typed(identity):
    """`amount` is text at origin (a non-numeric value forces that
    inference) — replaced later with an all-numeric column, narrowing it."""
    source_id, path = await _seed_source(identity, "customers.csv", b"id,amount\nc1,abc\n")
    await _apply(identity, source_id)
    return source_id, path


@pytest_asyncio.fixture
async def applied_two_column(identity):
    """`legacy` disappears at origin later — D7 forbids dropping it."""
    source_id, path = await _seed_source(identity, "customers.csv", b"id,legacy\nc1,old\n")
    await _apply(identity, source_id)
    return source_id, path


async def _earn_trust(identity, pattern: str, streak: int = 20) -> None:
    """Pre-seed what is_auto_apply_eligible requires. The real accept/reject
    cycle is exercised in services/api/tests/test_trust_learning.py; this
    file's subject is evolution, which needs trust as a precondition."""
    async with user_session(identity.user_id) as db:
        await db.execute(
            text(
                "insert into public.pattern_trust_scores "
                "(org_id, pattern_signature, approvals, rejections, "
                " consecutive_approvals, score) "
                "values (:org, :pattern, :streak, 0, :streak, 0.9) "
                "on conflict (org_id, pattern_signature) do update set "
                "  consecutive_approvals = :streak, auto_apply_enabled = true"
            ),
            {"org": identity.org_id, "pattern": pattern, "streak": streak},
        )


async def _columns(org_id: uuid.UUID, table: str) -> set[str]:
    async with tenant_session(org_id) as db:
        rows = (
            await db.execute(
                text(
                    "select column_name from information_schema.columns "
                    "where table_schema = :schema and table_name = :table"
                ),
                {"schema": tenant_schema_name(org_id), "table": table},
            )
        ).scalars().all()
    return set(rows)


async def test_an_additive_change_auto_applies_when_trusted(identity, applied_customers):
    source_id, path = applied_customers
    await _earn_trust(identity, "schema_migration:additive")
    await _reupload(path, b"id,email\nc1,a@example.com\n")

    result = await evolve_schema(
        {"redis": _FakeRedis()}, str(source_id), str(identity.org_id), str(identity.user_id)
    )
    assert result["status"] == "applied"
    assert "email" in await _columns(identity.org_id, "customers")


async def test_the_same_change_awaits_review_when_not_trusted(identity, applied_customers):
    """The genuinely new behaviour: a reversible migration is eligible for
    autonomy, not entitled to it."""
    source_id, path = applied_customers
    await _reupload(path, b"id,email\nc1,a@example.com\n")

    result = await evolve_schema(
        {"redis": _FakeRedis()}, str(source_id), str(identity.org_id), str(identity.user_id)
    )
    assert result["status"] == "proposed"
    assert "email" not in await _columns(identity.org_id, "customers")


async def test_a_destructive_change_is_always_proposed_even_at_maximum_trust(
    identity, applied_typed
):
    """No trust level skips this. ADR-0017 §3 makes irreversibility a
    ceiling, and D7 keeps it there."""
    source_id, path = applied_typed  # a column currently typed text
    await _earn_trust(identity, "schema_migration:destructive", streak=500)
    await _reupload(path, b"id,amount\nc1,1\n")  # text -> integer, narrowing

    result = await evolve_schema(
        {"redis": _FakeRedis()}, str(source_id), str(identity.org_id), str(identity.user_id)
    )
    assert result["status"] == "proposed"


async def test_a_column_absent_at_origin_is_kept(identity, applied_two_column):
    """D7 forbids DROP COLUMN outright. The column stays and is marked
    deprecated in the spec."""
    source_id, path = applied_two_column  # id, legacy
    await _earn_trust(identity, "schema_migration:additive")
    await _reupload(path, b"id\nc1\n")

    await evolve_schema(
        {"redis": _FakeRedis()}, str(source_id), str(identity.org_id), str(identity.user_id)
    )
    assert "legacy" in await _columns(identity.org_id, "customers")
