"""GET/PUT .../schedule and GET .../drift-events — ADR-0008's one new API
surface for a person, as opposed to the worker. Against the live project: RLS
on both tables, and the upsert's `on conflict (source_id)` path, are exactly
the kind of thing that passes a mock and fails for real.
"""

from __future__ import annotations

import json
import uuid

import httpx
import pytest
import pytest_asyncio
from sqlalchemy import text

from lumen_api.auth.dependencies import Identity
from lumen_api.db.session import service_session, user_session
from lumen_api.schedules import ScheduleUpdate, get_schedule, list_drift_events, set_schedule
from lumen_api.settings import get_settings

pytestmark = pytest.mark.integration


def _admin_headers() -> dict[str, str]:
    key = get_settings().supabase_service_role_key.get_secret_value()
    return {"apikey": key, "Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _create_user() -> uuid.UUID:
    settings = get_settings()
    response = httpx.post(
        f"{settings.supabase_url}/auth/v1/admin/users",
        headers=_admin_headers(),
        json={
            "email": f"lumen-schedules-{uuid.uuid4().hex[:12]}@example.com",
            "password": uuid.uuid4().hex + "Aa1!",
            "email_confirm": True,
            "user_metadata": {"display_name": "Schedules Tester", "org_name": "Schedules Org"},
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


async def _seed_source(identity: Identity) -> uuid.UUID:
    source_id = uuid.uuid4()
    async with user_session(identity.user_id) as db:
        await db.execute(
            text(
                "insert into public.data_sources "
                "(id, org_id, name, kind, status, object_path, table_name) "
                "values (:id, :org, 'watched.csv', 'csv', 'idle', 'org/x/watched.csv', 'watched')"
            ),
            {"id": source_id, "org": identity.org_id},
        )
    return source_id


async def test_get_schedule_with_none_configured_returns_the_off_shape(identity):
    source_id = await _seed_source(identity)
    result = await get_schedule(source_id, identity)
    assert result == {
        "id": None,
        "source_id": str(source_id),
        "enabled": False,
        "auto_apply_enabled": False,
        "cron": None,
        "last_run_at": None,
        "next_run_at": None,
    }


async def test_put_schedule_creates_then_upserts_on_a_second_call(identity):
    source_id = await _seed_source(identity)

    created = await set_schedule(source_id, ScheduleUpdate(enabled=True, auto_apply_enabled=False), identity)
    assert created["enabled"] is True
    assert created["auto_apply_enabled"] is False
    assert created["id"] is not None

    updated = await set_schedule(source_id, ScheduleUpdate(enabled=True, auto_apply_enabled=True), identity)
    # Same row — the unique(source_id) constraint's on-conflict path, not a
    # second insert that would have raised.
    assert updated["id"] == created["id"]
    assert updated["auto_apply_enabled"] is True


async def test_list_drift_events_returns_newest_first(identity):
    source_id = await _seed_source(identity)
    # Two separate transactions, not a loop in one — `now()` is fixed for the
    # whole transaction in Postgres, so inserts sharing one would tie on
    # `occurred_at` and make "newest first" unverifiable.
    for kind, severity in (("schema_change", 0.3), ("statistical_shift", 0.6)):
        async with service_session() as db:
            await db.execute(
                text(
                    "insert into public.drift_events (org_id, source_id, kind, severity, details) "
                    "values (:org, :source, :kind, :severity, cast(:details as jsonb))"
                ),
                {
                    "org": identity.org_id,
                    "source": source_id,
                    "kind": kind,
                    "severity": severity,
                    "details": json.dumps({}),
                },
            )

    result = await list_drift_events(source_id, identity, limit=20)
    assert [event["kind"] for event in result["events"]] == ["statistical_shift", "schema_change"]
    assert all(event["status"] == "detected" for event in result["events"])
