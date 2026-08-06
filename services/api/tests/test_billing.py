"""ADR-0004 against the live project: QuotaGate's admission decisions, the
account tools, and the two proposal kinds that only an owner may decide.

Each test creates its own user (and therefore its own org and its own free
subscription, via the same triggers a real signup exercises), so quota state
from one test can never leak into another.
"""

from __future__ import annotations

import uuid

import httpx
import pytest
import pytest_asyncio
from sqlalchemy import text

from lumen_api.agents.registry import build_tool_registry
from lumen_api.auth.dependencies import Identity
from lumen_api.billing.quota import Decision, QuotaGate
from lumen_api.db.session import service_session, user_session
from lumen_api.errors import Forbidden
from lumen_api.proposals import DecisionRequest, decide_proposal
from lumen_api.settings import get_settings

pytestmark = pytest.mark.integration


def _admin_headers() -> dict[str, str]:
    key = get_settings().supabase_service_role_key.get_secret_value()
    return {"apikey": key, "Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _create_user(display_name: str = "Billing Tester", org_name: str = "Billing Org") -> uuid.UUID:
    settings = get_settings()
    response = httpx.post(
        f"{settings.supabase_url}/auth/v1/admin/users",
        headers=_admin_headers(),
        json={
            "email": f"lumen-billing-{uuid.uuid4().hex[:12]}@example.com",
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
        f"{settings.supabase_url}/auth/v1/admin/users/{user_id}",
        headers=_admin_headers(),
        timeout=30,
    )


async def _identity_of(user_id: uuid.UUID) -> Identity:
    async with user_session(user_id) as db:
        row = (
            await db.execute(text("select * from public.current_identity()"))
        ).mappings().first()
    return Identity(
        user_id=row["user_id"], email=row["email"], display_name=row["display_name"],
        avatar_url=row["avatar_url"], org_id=row["org_id"], org_name=row["org_name"],
        org_slug=row["org_slug"], plan_code=row["plan_code"], role=str(row["role"]),
    )


async def _seed_run(identity: Identity) -> tuple[uuid.UUID, uuid.UUID]:
    """A minimal `runs` row — enough for propose_* tools to attach a proposal
    to, without exercising the full agent loop for tools that never call one."""
    thread_id = uuid.uuid4()
    async with user_session(identity.user_id) as db:
        run_id = (
            await db.execute(
                text(
                    "insert into public.runs (org_id, thread_id, kind, status, backend, created_by) "
                    "values (:org, :thread, 'chat', 'succeeded', 'polars', :user) returning id"
                ),
                {"org": identity.org_id, "thread": thread_id, "user": identity.user_id},
            )
        ).scalar_one()
    return run_id, thread_id


async def _add_member(org_id: uuid.UUID, user_id: uuid.UUID, role: str) -> None:
    """Service session: adding someone else into an org they didn't create is
    exactly the cross-tenant maintenance `service_session()`'s docstring
    reserves itself for — test setup, not a path the app itself takes."""
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


async def test_quota_gate_allows_a_fresh_org(identity):
    gate = QuotaGate(identity.org_id, identity.user_id)
    status = await gate.status("agent_run")
    assert status.decision == Decision.ALLOW
    assert status.used == 0
    assert status.limit == 50  # the free plan's seeded agent_runs_limit


async def test_quota_gate_denies_once_the_free_plan_limit_is_reached(identity):
    gate = QuotaGate(identity.org_id, identity.user_id)
    async with user_session(identity.user_id) as db:
        for _ in range(50):
            await gate.record(db, metric="agent_run", quantity=1, agent="test")

    status = await gate.status("agent_run")
    assert status.decision == Decision.DENY
    assert status.ratio >= 1.0


async def test_quota_gate_warns_at_eighty_percent(identity):
    gate = QuotaGate(identity.org_id, identity.user_id)
    async with user_session(identity.user_id) as db:
        for _ in range(40):  # 40/50 = 80%
            await gate.record(db, metric="agent_run", quantity=1, agent="test")

    status = await gate.status("agent_run")
    assert status.decision == Decision.WARN


async def test_get_usage_reports_real_numbers(identity):
    registry = build_tool_registry(identity.org_id, identity.user_id)
    gate = QuotaGate(identity.org_id, identity.user_id)
    async with user_session(identity.user_id) as db:
        await gate.record(db, metric="agent_run", quantity=3, agent="test")

    result = await registry.invoke("get_usage", {})
    assert result["ok"] is True
    by_metric = {row["metric"]: row for row in result["data"]["usage"]}
    assert by_metric["agent_run"]["used"] == 3
    assert by_metric["agent_run"]["limit"] == 50


async def test_list_members_includes_the_owner(identity):
    registry = build_tool_registry(identity.org_id, identity.user_id)
    result = await registry.invoke("list_members", {})
    assert result["ok"] is True
    roles = {m["email"]: m["role"] for m in result["data"]["members"]}
    assert roles[identity.email] == "owner"


async def test_recommend_plan_suggests_an_upgrade_under_pressure(identity):
    registry = build_tool_registry(identity.org_id, identity.user_id)
    gate = QuotaGate(identity.org_id, identity.user_id)
    async with user_session(identity.user_id) as db:
        for _ in range(45):  # 90% of the free plan's 50 agent_runs_limit
            await gate.record(db, metric="agent_run", quantity=1, agent="test")

    result = await registry.invoke("recommend_plan", {})
    assert result["ok"] is True
    assert result["data"]["current_plan"] == "free"
    assert result["data"]["recommendation"] == "pro"


async def test_propose_plan_change_then_accept_updates_both_copies(identity):
    run_id, thread_id = await _seed_run(identity)
    registry = build_tool_registry(
        identity.org_id, identity.user_id, run_id=run_id, thread_id=thread_id
    )

    proposed = await registry.invoke(
        "propose_plan_change", {"plan_code": "pro", "rationale": "usage is climbing"}
    )
    assert proposed["ok"] is True
    proposal_id = uuid.UUID(proposed["data"]["proposal_id"])

    result = await decide_proposal(proposal_id, DecisionRequest(decision="accept"), identity)
    assert result["status"] == "applied"
    assert result["plan_code"] == "pro"
    # No STRIPE_SECRET_KEY in this environment — the plan still changes in
    # the app, and the response says plainly that no payment was collected,
    # rather than silently doing nothing or pretending to redirect somewhere.
    assert result["checkout_url"] is None
    assert "not configured" in (result["note"] or "").lower()

    async with user_session(identity.user_id) as db:
        org_plan = (
            await db.execute(
                text("select plan_code from public.organizations where id = :org"),
                {"org": identity.org_id},
            )
        ).scalar_one()
        sub_plan = (
            await db.execute(
                text("select plan_code from public.subscriptions where org_id = :org"),
                {"org": identity.org_id},
            )
        ).scalar_one()
    assert org_plan == sub_plan == "pro"


async def test_a_member_cannot_decide_a_plan_change(identity):
    other_id = _create_user(display_name="Second Member", org_name="Unused")
    try:
        await _add_member(identity.org_id, other_id, "member")
        member_identity = Identity(
            user_id=other_id, email="member@example.com", display_name="Second Member",
            avatar_url=None, org_id=identity.org_id, org_name=identity.org_name,
            org_slug=identity.org_slug, plan_code=identity.plan_code, role="member",
        )

        run_id, thread_id = await _seed_run(identity)
        registry = build_tool_registry(
            identity.org_id, identity.user_id, run_id=run_id, thread_id=thread_id
        )
        proposed = await registry.invoke(
            "propose_plan_change", {"plan_code": "pro", "rationale": "test"}
        )
        proposal_id = uuid.UUID(proposed["data"]["proposal_id"])

        with pytest.raises(Forbidden):
            await decide_proposal(proposal_id, DecisionRequest(decision="accept"), member_identity)
    finally:
        _delete_user(other_id)


async def test_propose_member_role_change_then_accept_updates_the_membership(identity):
    other_id = _create_user(display_name="Promotable Member", org_name="Unused")
    try:
        await _add_member(identity.org_id, other_id, "viewer")
        run_id, thread_id = await _seed_run(identity)
        registry = build_tool_registry(
            identity.org_id, identity.user_id, run_id=run_id, thread_id=thread_id
        )

        proposed = await registry.invoke(
            "propose_member_role_change", {"target_user_id": str(other_id), "role": "admin"}
        )
        assert proposed["ok"] is True
        proposal_id = uuid.UUID(proposed["data"]["proposal_id"])

        result = await decide_proposal(proposal_id, DecisionRequest(decision="accept"), identity)
        assert result["status"] == "applied"
        assert result["role"] == "admin"

        async with user_session(identity.user_id) as db:
            role = (
                await db.execute(
                    text(
                        "select role from public.memberships where org_id = :org and user_id = :user"
                    ),
                    {"org": identity.org_id, "user": other_id},
                )
            ).scalar_one()
        assert str(role) == "admin"
    finally:
        _delete_user(other_id)
