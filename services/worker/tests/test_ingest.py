"""Staging load. Data must be visible before any schema is approved (D4)."""

from __future__ import annotations

import uuid

import httpx
import polars as pl
import pytest
import pytest_asyncio
from sqlalchemy import text

from lumen_api.auth.dependencies import Identity
from lumen_api.datasets.store import SupabaseStorage
from lumen_api.db.session import user_session
from lumen_api.settings import get_settings
from lumen_api.tenant_db import tenant_raw_schema_name, tenant_schema_name, tenant_session
from lumen_worker.ingest import ingest_to_staging

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
            "email": f"lumen-ingest-{uuid.uuid4().hex[:12]}@example.com",
            "password": uuid.uuid4().hex + "Aa1!",
            "email_confirm": True,
            "user_metadata": {"display_name": "Ingest Tester", "org_name": "Ingest Org"},
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


async def _seed(identity, name: str, content: bytes) -> uuid.UUID:
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
    return source_id


async def test_a_csv_lands_in_staging(identity):
    source_id = await _seed(identity, "orders.csv", b"id,amount\n1,10\n2,20\n")
    ctx = {"redis": _FakeRedis()}

    result = await ingest_to_staging(
        ctx, str(source_id), str(identity.org_id), str(identity.user_id)
    )
    assert result["status"] == "staged"
    assert result["rows"] == 2

    async with tenant_session(identity.org_id) as db:
        count = (await db.execute(text("select count(*) from orders"))).scalar_one()
    assert count == 2


async def test_staging_enqueues_the_design_job(identity):
    source_id = await _seed(identity, "orders.csv", b"id\n1\n")
    ctx = {"redis": _FakeRedis()}
    await ingest_to_staging(ctx, str(source_id), str(identity.org_id), str(identity.user_id))
    assert any(job[0] == "design_schema_job" for job in ctx["redis"].jobs)


async def test_an_unsupported_format_marks_the_source_in_error(identity):
    source_id = await _seed(identity, "notes.docx", b"not a dataset")
    ctx = {"redis": _FakeRedis()}

    result = await ingest_to_staging(
        ctx, str(source_id), str(identity.org_id), str(identity.user_id)
    )
    assert result["status"] == "error"

    async with user_session(identity.user_id) as db:
        status = (
            await db.execute(
                text("select status from public.data_sources where id = :id"),
                {"id": source_id},
            )
        ).scalar_one()
    assert str(status) == "error"


async def test_an_unsupported_format_does_not_enqueue_a_design(identity):
    source_id = await _seed(identity, "notes.docx", b"nope")
    ctx = {"redis": _FakeRedis()}
    await ingest_to_staging(ctx, str(source_id), str(identity.org_id), str(identity.user_id))
    assert ctx["redis"].jobs == []


async def test_a_source_with_no_file_is_skipped(identity):
    source_id = uuid.uuid4()
    async with user_session(identity.user_id) as db:
        await db.execute(
            text(
                "insert into public.data_sources (id, org_id, name, kind, status) "
                "values (:id, :org, 'empty.csv', 'csv', 'idle')"
            ),
            {"id": source_id, "org": identity.org_id},
        )
    result = await ingest_to_staging(
        {"redis": _FakeRedis()}, str(source_id), str(identity.org_id), str(identity.user_id)
    )
    assert result["status"] == "skipped"


# ── the design job ──────────────────────────────────────────────────────

from lumen_worker.ingest import design_schema_job  # noqa: E402


async def test_the_design_job_creates_an_awaiting_review_proposal(identity):
    source_id = await _seed(identity, "customers.csv", b"id,name\nc1,Acme\nc2,Globex\n")
    ctx = {"redis": _FakeRedis()}
    await ingest_to_staging(ctx, str(source_id), str(identity.org_id), str(identity.user_id))

    result = await design_schema_job(
        ctx, str(source_id), str(identity.org_id), str(identity.user_id)
    )
    assert result["status"] == "proposed"

    async with user_session(identity.user_id) as db:
        row = (
            await db.execute(
                text(
                    "select kind, status from public.proposals where id = :id"
                ),
                {"id": uuid.UUID(result["proposal_id"])},
            )
        ).mappings().first()
    assert row["kind"] == "schema_design"
    assert str(row["status"]) == "awaiting_review"


async def test_nothing_is_created_in_the_modelled_schema_before_acceptance(identity):
    """D4: staging is immediate, promotion is approved. A table appearing
    before a human said yes would break the whole propose-then-apply spine."""
    source_id = await _seed(identity, "customers.csv", b"id\nc1\n")
    ctx = {"redis": _FakeRedis()}
    await ingest_to_staging(ctx, str(source_id), str(identity.org_id), str(identity.user_id))
    await design_schema_job(ctx, str(source_id), str(identity.org_id), str(identity.user_id))

    async with tenant_session(identity.org_id) as db:
        count = (
            await db.execute(
                text(
                    "select count(*) from information_schema.tables "
                    "where table_schema = :schema"
                ),
                {"schema": tenant_schema_name(identity.org_id)},
            )
        ).scalar_one()
    assert count == 0
