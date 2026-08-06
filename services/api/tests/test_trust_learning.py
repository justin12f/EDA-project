"""decide_proposal -> pattern_trust_scores (ADR-0011), against the live
project: the posterior update, the streak reset on rejection, and the
role gate (only admin/owner decisions build trust) all read wrong from a
mock session — this exercises the real accept/reject path.
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid

import httpx
import pytest
import pytest_asyncio
from sqlalchemy import text

from lumen.agents.master_factory import AgentMasterFactory
from lumen_api.auth.dependencies import Identity
from lumen_api.datasets.store import HandleStore, SupabaseStorage
from lumen_api.db.session import service_session, user_session
from lumen_api.proposals import DecisionRequest, decide_proposal
from lumen_api.settings import get_settings

pytestmark = pytest.mark.integration

CSV = b"id,customer_id\n1,c-1\n2,c-2\n"
PATTERN = "cleaning_pipeline:imputation"


def _admin_headers() -> dict[str, str]:
    key = get_settings().supabase_service_role_key.get_secret_value()
    return {"apikey": key, "Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _create_user(display_name: str = "Trust Tester", org_name: str = "Trust Org") -> uuid.UUID:
    settings = get_settings()
    response = httpx.post(
        f"{settings.supabase_url}/auth/v1/admin/users",
        headers=_admin_headers(),
        json={
            "email": f"lumen-trust-{uuid.uuid4().hex[:12]}@example.com",
            "password": uuid.uuid4().hex + "Aa1!",
            "email_confirm": True,
            "user_metadata": {"display_name": display_name, "org_name": org_name},
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


async def _add_member(org_id: uuid.UUID, user_id: uuid.UUID, role: str) -> None:
    """Adding someone else into an org they didn't create is exactly the
    cross-tenant maintenance service_session()'s docstring reserves itself
    for — test setup, not a path the app itself takes."""
    async with service_session() as db:
        await db.execute(
            text(
                "insert into public.memberships (org_id, user_id, role) "
                "values (:org, :user, cast(:role as public.org_role)) "
                "on conflict (org_id, user_id) do update set role = excluded.role"
            ),
            {"org": org_id, "user": user_id, "role": role},
        )


@pytest_asyncio.fixture
async def identity():
    user_id = _create_user()
    try:
        yield await _identity_of(user_id)
    finally:
        _delete_user(user_id)


async def _seed_source(identity: Identity) -> uuid.UUID:
    path = f"org/{identity.org_id}/uploads/{uuid.uuid4().hex}.csv"
    await SupabaseStorage().upload(path, CSV, "text/csv")
    source_id = uuid.uuid4()
    async with user_session(identity.user_id) as db:
        await db.execute(
            text(
                "insert into public.data_sources (id, org_id, name, kind, status, object_path, table_name) "
                "values (:id, :org, 'trust.csv', 'csv', 'idle', :path, 'trust')"
            ),
            {"id": source_id, "org": identity.org_id, "path": path},
        )
    payload = await SupabaseStorage().download(path)
    directory = tempfile.mkdtemp(prefix="lumen-trust-test-")
    local = os.path.join(directory, "source.csv")
    with open(local, "wb") as file:
        file.write(payload)
    frame = AgentMasterFactory("polars").readers().read(local)
    await HandleStore(identity.org_id, identity.user_id).put(frame, label="trust", source_id=source_id)
    return source_id


async def _propose(identity: Identity, source_id: uuid.UUID) -> uuid.UUID:
    """A fresh cleaning_pipeline proposal, same structural pattern (a single
    impute_categorical step) every time — a *new* rid per call, since a
    proposal's rid must point at a live handle, but the same shape, since
    that's what pattern_trust_scores keys on."""
    async with user_session(identity.user_id) as db:
        rid = (
            await db.execute(
                text("select rid from public.dataset_handles where source_id = :source limit 1"),
                {"source": source_id},
            )
        ).scalar_one()
        run_id = (
            await db.execute(
                text(
                    "insert into public.runs (org_id, source_id, thread_id, kind, status, backend) "
                    "values (:org, :source, :thread, 'chat', 'succeeded', 'polars') returning id"
                ),
                {"org": identity.org_id, "source": source_id, "thread": uuid.uuid4()},
            )
        ).scalar_one()
        spec = {"rid": rid, "steps": [{"impute_categorical": {"columns": ["customer_id"], "strategy": "fixed"}}]}
        proposal_id = (
            await db.execute(
                text(
                    "insert into public.proposals "
                    "(org_id, run_id, thread_id, author_agent, kind, spec, rationale) "
                    "select org_id, :run, thread_id, 'analyst', 'cleaning_pipeline', "
                    "       cast(:spec as jsonb), 'test' "
                    "from public.runs where id = :run returning id"
                ),
                {"run": run_id, "spec": json.dumps(spec)},
            )
        ).scalar_one()
    return proposal_id


async def _score(identity: Identity) -> dict:
    async with user_session(identity.user_id) as db:
        row = (
            await db.execute(
                text(
                    "select approvals, rejections, consecutive_approvals, score "
                    "from public.pattern_trust_scores where org_id = :org and pattern_signature = :pattern"
                ),
                {"org": identity.org_id, "pattern": PATTERN},
            )
        ).mappings().first()
    return dict(row) if row else {}


async def test_consecutive_approvals_accumulate_and_a_rejection_resets_the_streak(identity):
    source_id = await _seed_source(identity)

    first = await _propose(identity, source_id)
    await decide_proposal(first, DecisionRequest(decision="accept"), identity)
    assert await _score(identity) == pytest.approx(
        {"approvals": 1, "rejections": 0, "consecutive_approvals": 1, "score": 0.2065}, abs=1e-3
    )

    second = await _propose(identity, source_id)
    await decide_proposal(second, DecisionRequest(decision="accept"), identity)
    after_two = await _score(identity)
    assert after_two["approvals"] == 2
    assert after_two["consecutive_approvals"] == 2

    third = await _propose(identity, source_id)
    await decide_proposal(third, DecisionRequest(decision="reject"), identity)
    after_reject = await _score(identity)
    # Lifetime counts both move; the streak alone resets to 0 — the ADR §4
    # behavior ("a fresh run of approvals... reconsidered"), not a
    # gradually-decaying number.
    assert after_reject["approvals"] == 2
    assert after_reject["rejections"] == 1
    assert after_reject["consecutive_approvals"] == 0

    fourth = await _propose(identity, source_id)
    await decide_proposal(fourth, DecisionRequest(decision="accept"), identity)
    after_recovery = await _score(identity)
    assert after_recovery["consecutive_approvals"] == 1  # counting up again from the reset, not from 2


async def test_a_members_decision_does_not_build_trust(identity):
    source_id = await _seed_source(identity)
    other_id = _create_user(display_name="Second Member", org_name="Unused")
    try:
        await _add_member(identity.org_id, other_id, "member")
        member_identity = Identity(
            user_id=other_id,
            email="member@example.com",
            display_name="Second Member",
            avatar_url=None,
            org_id=identity.org_id,
            org_name=identity.org_name,
            org_slug=identity.org_slug,
            plan_code=identity.plan_code,
            role="member",
        )

        proposal_id = await _propose(identity, source_id)
        result = await decide_proposal(proposal_id, DecisionRequest(decision="accept"), member_identity)
        # The proposal itself still applies normally — a member's authority
        # to accept a cleaning_pipeline is unchanged by this ADR.
        assert result["status"] == "applied"
        # But it must not have created or touched a trust row.
        assert await _score(identity) == {}
    finally:
        _delete_user(other_id)


async def test_an_admins_decision_does_build_trust(identity):
    source_id = await _seed_source(identity)
    other_id = _create_user(display_name="Second Admin", org_name="Unused")
    try:
        await _add_member(identity.org_id, other_id, "admin")
        admin_identity = Identity(
            user_id=other_id,
            email="admin@example.com",
            display_name="Second Admin",
            avatar_url=None,
            org_id=identity.org_id,
            org_name=identity.org_name,
            org_slug=identity.org_slug,
            plan_code=identity.plan_code,
            role="admin",
        )

        proposal_id = await _propose(identity, source_id)
        await decide_proposal(proposal_id, DecisionRequest(decision="accept"), admin_identity)
        assert (await _score(identity))["approvals"] == 1
    finally:
        _delete_user(other_id)
