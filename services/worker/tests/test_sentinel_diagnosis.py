"""process_schedule -> diagnose_drift, end to end against the live project.

Two real ticks — one to record a baseline, one against a changed file at the
same object_path (what a re-synced source looks like) — so the drift event
`diagnose_drift` reads is one `lumen.sentinel.detect_drift` actually produced,
not a row constructed by the test. This is what caught `process_schedule`
never persisting the baseline its own next tick needs: a test that inserted a
`drift_events` row directly would never have exercised that gap.

Runs on MockProvider (LLM_MODE=mock) — see `lumen.llm.mock_provider`'s
`_diagnose_drift` for what it does with the Sentinel's `propose_patch` /
`cannot_diagnose` registry specifically.
"""

from __future__ import annotations

import uuid

import httpx
import pytest
import pytest_asyncio
from sqlalchemy import text

from lumen_api.auth.dependencies import Identity
from lumen_api.datasets.store import SupabaseStorage
from lumen_api.db.session import service_session, user_session
from lumen_api.settings import get_settings
from lumen_worker.sentinel import diagnose_drift, process_schedule

pytestmark = pytest.mark.integration

BASELINE_CSV = b"id,country_code,email_hash\n1,DE,a1\n2,US,b2\n3,FR,c3\n"
# Adds 'signup_source', empty for most rows: additive schema drift with a small
# row-level footprint — the narrow case `auto_apply_enabled` exists for.
ADDITIVE_CSV = b"id,country_code,email_hash,signup_source\n1,DE,a1,ads\n2,US,b2,\n3,FR,c3,\n"
# 'email_hash' is gone outright — not a mechanical fix.
REMOVED_COLUMN_CSV = b"id,country_code\n1,DE\n2,US\n3,FR\n"
# Same columns; country_code's null rate jumps well past the 5% threshold.
NULL_SHIFT_CSV = b"id,country_code,email_hash\n1,,a1\n2,,b2\n3,FR,c3\n"


class _FakeRedis:
    async def enqueue_job(self, *args, **kwargs) -> None:
        return None


_CTX = {"redis": _FakeRedis()}


async def _upload_and_wait(path: str, content: bytes) -> None:
    """Supabase Storage's overwrite is not immediately read-your-writes — a
    download right after an `x-upsert` upload can still return the previous
    version for a moment. A real scheduled tick never notices (the next one
    fires an hour or more later); this test would race it every time without
    waiting for the object to actually be the one just written.
    """
    import asyncio

    storage = SupabaseStorage()
    await storage.upload(path, content, "text/csv")
    # Measured directly against this project's Storage/CDN (Cloudflare in
    # front of it): convergence is sometimes immediate and sometimes well
    # past 30s, varying by which edge node a fresh connection lands on —
    # `SupabaseStorage` opens a new client per call, same as production code,
    # so this polls the exact path a real tick would if it ever raced this
    # (it never does — the next tick is an hour or more later). Budgeted
    # generously because this loop only ever costs real time, never
    # correctness.
    for _ in range(150):
        if await storage.download(path) == content:
            return
        await asyncio.sleep(1)
    raise TimeoutError(f"Storage never converged on the new content at {path}")


def _admin_headers() -> dict[str, str]:
    key = get_settings().supabase_service_role_key.get_secret_value()
    return {"apikey": key, "Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _create_user() -> uuid.UUID:
    settings = get_settings()
    response = httpx.post(
        f"{settings.supabase_url}/auth/v1/admin/users",
        headers=_admin_headers(),
        json={
            "email": f"lumen-sentinel-{uuid.uuid4().hex[:12]}@example.com",
            "password": uuid.uuid4().hex + "Aa1!",
            "email_confirm": True,
            "user_metadata": {"display_name": "Sentinel Tester", "org_name": "Sentinel Org"},
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


@pytest.fixture(autouse=True)
def _force_mock_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_MODE", "mock")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


async def _seed_scheduled_source(identity: Identity, *, auto_apply: bool) -> tuple[uuid.UUID, uuid.UUID, str]:
    path = f"org/{identity.org_id}/uploads/{uuid.uuid4().hex}.csv"
    await _upload_and_wait(path, BASELINE_CSV)

    source_id = uuid.uuid4()
    async with user_session(identity.user_id) as db:
        await db.execute(
            text(
                "insert into public.data_sources "
                "(id, org_id, name, kind, status, object_path, table_name) "
                "values (:id, :org, 'watched.csv', 'csv', 'idle', :path, 'watched')"
            ),
            {"id": source_id, "org": identity.org_id, "path": path},
        )
        schedule_id = (
            await db.execute(
                text(
                    "insert into public.data_source_schedules "
                    "(org_id, source_id, auto_apply_enabled, created_by) "
                    "values (:org, :source, :auto, :user) returning id"
                ),
                {
                    "org": identity.org_id,
                    "source": source_id,
                    "auto": auto_apply,
                    "user": identity.user_id,
                },
            )
        ).scalar_one()
    return source_id, schedule_id, path


async def _tick(schedule_id: uuid.UUID, identity: Identity, source_id: uuid.UUID) -> dict:
    return await process_schedule(
        _CTX, str(schedule_id), str(identity.org_id), str(source_id), str(identity.user_id)
    )


async def _drift_event_for(source_id: uuid.UUID) -> dict:
    async with service_session() as db:
        row = (
            await db.execute(
                text(
                    "select id, status, proposal_id from public.drift_events "
                    "where source_id = :source order by occurred_at desc limit 1"
                ),
                {"source": source_id},
            )
        ).mappings().first()
    return dict(row)


async def _proposal(proposal_id: uuid.UUID, identity: Identity) -> dict:
    async with user_session(identity.user_id) as db:
        row = (
            await db.execute(
                text("select status, kind from public.proposals where id = :id"), {"id": proposal_id}
            )
        ).mappings().first()
    return dict(row)


async def test_a_second_tick_with_no_change_reports_no_drift(identity):
    source_id, schedule_id, _path = await _seed_scheduled_source(identity, auto_apply=False)
    first = await _tick(schedule_id, identity, source_id)
    assert first["status"] == "baseline_recorded"

    second = await _tick(schedule_id, identity, source_id)
    assert second["status"] == "no_drift"


async def test_additive_schema_drift_is_proposed_and_auto_applied_when_opted_in(identity):
    source_id, schedule_id, path = await _seed_scheduled_source(identity, auto_apply=True)
    assert (await _tick(schedule_id, identity, source_id))["status"] == "baseline_recorded"

    await _upload_and_wait(path, ADDITIVE_CSV)
    second = await _tick(schedule_id, identity, source_id)
    assert second["status"] == "drift_detected"

    result = await diagnose_drift(_CTX, second["drift_event_id"], str(identity.org_id), str(identity.user_id))
    assert result["status"] == "applied"

    event = await _drift_event_for(source_id)
    assert event["status"] == "resolved"
    assert event["proposal_id"] is not None
    proposal = await _proposal(event["proposal_id"], identity)
    assert proposal == {"status": "applied", "kind": "pipeline_patch"}


async def test_additive_schema_drift_awaits_review_without_auto_apply_opt_in(identity):
    source_id, schedule_id, path = await _seed_scheduled_source(identity, auto_apply=False)
    await _tick(schedule_id, identity, source_id)
    await _upload_and_wait(path, ADDITIVE_CSV)
    second = await _tick(schedule_id, identity, source_id)

    result = await diagnose_drift(_CTX, second["drift_event_id"], str(identity.org_id), str(identity.user_id))
    assert result["status"] == "proposed"

    event = await _drift_event_for(source_id)
    assert event["status"] == "proposed"
    proposal = await _proposal(event["proposal_id"], identity)
    assert proposal == {"status": "awaiting_review", "kind": "pipeline_patch"}


async def test_a_removed_column_is_left_for_manual_review_not_guessed_at(identity):
    source_id, schedule_id, path = await _seed_scheduled_source(identity, auto_apply=True)
    await _tick(schedule_id, identity, source_id)
    await _upload_and_wait(path, REMOVED_COLUMN_CSV)
    second = await _tick(schedule_id, identity, source_id)

    result = await diagnose_drift(_CTX, second["drift_event_id"], str(identity.org_id), str(identity.user_id))
    assert result["status"] == "cannot_diagnose"

    event = await _drift_event_for(source_id)
    assert event["status"] == "dismissed"
    assert event["proposal_id"] is None


async def test_a_null_rate_shift_is_proposed_but_never_auto_applied(identity):
    source_id, schedule_id, path = await _seed_scheduled_source(identity, auto_apply=True)
    await _tick(schedule_id, identity, source_id)
    await _upload_and_wait(path, NULL_SHIFT_CSV)
    second = await _tick(schedule_id, identity, source_id)

    result = await diagnose_drift(_CTX, second["drift_event_id"], str(identity.org_id), str(identity.user_id))
    # medium confidence: a real proposal, but auto_apply_enabled=True must not
    # matter here — only the narrow additive-schema case is allowed to apply
    # itself.
    assert result["status"] == "proposed"

    event = await _drift_event_for(source_id)
    assert event["status"] == "proposed"
    proposal = await _proposal(event["proposal_id"], identity)
    assert proposal["status"] == "awaiting_review"
