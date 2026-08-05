"""Per-user agent memory, against the live project.

The claim being tested is not "rows go in and come out" — it is that two people
in the *same organization* see the same facts about the data and none of each
other's judgments. That distinction only exists in the RLS policy, so a unit
test with a fake store would assert the fake.
"""

from __future__ import annotations

import uuid

import httpx
import pytest
from sqlalchemy import text

from lumen_api.context.store import ContextEntry, ContextStore, Kind, Scope
from lumen_api.db.session import user_session
from lumen_api.settings import get_settings

pytestmark = pytest.mark.integration


def _admin_headers() -> dict[str, str]:
    key = get_settings().supabase_service_role_key.get_secret_value()
    return {"apikey": key, "Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _create_user(name: str = "Context Tester") -> uuid.UUID:
    settings = get_settings()
    response = httpx.post(
        f"{settings.supabase_url}/auth/v1/admin/users",
        headers=_admin_headers(),
        json={
            "email": f"lumen-ctx-{uuid.uuid4().hex[:12]}@example.com",
            "password": uuid.uuid4().hex + "Aa1!",
            "email_confirm": True,
            "user_metadata": {"display_name": name},
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


async def _org_of(user_id: uuid.UUID) -> uuid.UUID:
    async with user_session(user_id) as db:
        return (
            await db.execute(
                text("select org_id from public.memberships where user_id = :id"),
                {"id": user_id},
            )
        ).scalar_one()


@pytest.fixture
def person():
    user_id = _create_user()
    yield user_id
    _delete_user(user_id)


@pytest.fixture
async def teammates():
    """Two people who share one organization — the case the design turns on."""
    ana = _create_user("Ana")
    bo = _create_user("Bo")
    org_id = await _org_of(ana)

    # Put Bo in Ana's organization.
    async with user_session(ana) as db:
        await db.execute(
            text(
                "insert into public.memberships (org_id, user_id, role) "
                "values (:org, :user, 'member')"
            ),
            {"org": org_id, "user": bo},
        )

    yield ana, bo, org_id
    _delete_user(ana)
    _delete_user(bo)


# ── scope defaults ──────────────────────────────────────────────────────────


def test_the_default_scope_follows_what_the_fact_is_about():
    """Facts about data are shared; judgments are private. Encoded once, not at
    twenty call sites."""
    from lumen_api.context.store import DEFAULT_SCOPE

    assert DEFAULT_SCOPE[Kind.PROFILE] is Scope.ORG
    assert DEFAULT_SCOPE[Kind.SCHEMA] is Scope.ORG
    assert DEFAULT_SCOPE[Kind.DECISION] is Scope.USER
    assert DEFAULT_SCOPE[Kind.NOTE] is Scope.USER
    assert set(DEFAULT_SCOPE) == set(Kind), "every kind needs a decided default"


def test_an_entry_can_override_its_default_scope():
    entry = ContextEntry(kind=Kind.PROFILE, title="t", content="c", scope=Scope.USER)
    assert entry.resolved_scope() is Scope.USER
    assert ContextEntry(kind=Kind.PROFILE, title="t", content="c").resolved_scope() is Scope.ORG


# ── round trip ──────────────────────────────────────────────────────────────


async def test_a_written_context_comes_back_from_search(person):
    store = ContextStore(await _org_of(person), person)
    await store.remember(
        ContextEntry(
            kind=Kind.PROFILE,
            title="Profile of users.csv",
            content="12.4M rows. Null rates above 0.5%: country_code 3.2%.",
        )
    )

    matches = await store.search("which column has missing values", limit=5)

    assert matches, "embedding search returned nothing — is EMBEDDING_PROVIDER set?"
    assert any("country_code" in m.content for m in matches)
    assert matches[0].scope is Scope.ORG


async def test_search_reports_whether_a_match_is_mine(person):
    store = ContextStore(await _org_of(person), person)
    await store.remember(
        ContextEntry(
            kind=Kind.DECISION,
            title="Rejected dropping nulls",
            content="Do not drop rows for missing country_code; impute instead.",
        )
    )

    matches = await store.search("what did I decide about nulls", limit=5)
    mine = [m for m in matches if m.scope is Scope.USER]
    assert mine and all(m.is_mine for m in mine)


# ── the isolation the whole design exists for ───────────────────────────────


async def test_teammates_share_facts_about_the_data(teammates):
    """A profile is measured once. Nobody should pay to rediscover it."""
    ana, bo, org_id = teammates

    await ContextStore(org_id, ana).remember(
        ContextEntry(
            kind=Kind.PROFILE,
            title="Profile of orders.csv",
            content="1.8M rows. Null rates above 0.5%: currency 4.1%.",
        )
    )

    seen_by_bo = await ContextStore(org_id, bo).search("currency missing values", limit=5)
    assert any("currency 4.1%" in m.content for m in seen_by_bo)
    assert all(m.is_mine is False for m in seen_by_bo if m.scope is Scope.ORG)


async def test_a_teammate_never_sees_your_private_judgment(teammates):
    """Ana's rejected proposal is not evidence about Bo's intent."""
    ana, bo, org_id = teammates

    await ContextStore(org_id, ana).remember(
        ContextEntry(
            kind=Kind.DECISION,
            title="Rejected the dedupe step",
            content="Never deduplicate orders by customer_id — repeat orders are legitimate.",
        )
    )

    bo_sees = await ContextStore(org_id, bo).search("deduplicate orders", limit=10)
    assert not any("repeat orders are legitimate" in m.content for m in bo_sees)

    ana_sees = await ContextStore(org_id, ana).search("deduplicate orders", limit=10)
    assert any("repeat orders are legitimate" in m.content for m in ana_sees)


async def test_the_database_refuses_a_user_scoped_row_with_no_author(person):
    """A user-scoped row nobody owns would be invisible to everyone — silent
    data loss rather than a safe default."""
    org_id = await _org_of(person)

    with pytest.raises(Exception) as caught:
        async with user_session(person) as db:
            await db.execute(
                text(
                    "insert into public.data_contexts (org_id, user_id, kind, scope, content) "
                    "values (:org, null, 'decision', 'user', 'orphan')"
                ),
                {"org": org_id},
            )
    # Two layers refuse it: the RLS policy (user_id must be auth.uid()) and the
    # CHECK constraint. Whichever fires first, the row does not exist.
    message = str(caught.value).lower()
    assert "row-level security" in message or "data_contexts_user_scope_needs_author" in message


async def test_you_cannot_write_private_context_as_someone_else(teammates):
    ana, bo, org_id = teammates

    with pytest.raises(Exception) as caught:
        async with user_session(bo) as db:
            await db.execute(
                text(
                    "insert into public.data_contexts (org_id, user_id, kind, scope, content) "
                    "values (:org, :ana, 'decision', 'user', 'forged')"
                ),
                {"org": org_id, "ana": ana},
            )
    assert "row-level security" in str(caught.value).lower()


# ── the briefing ────────────────────────────────────────────────────────────


async def test_the_briefing_separates_shared_facts_from_personal_history(teammates):
    ana, bo, org_id = teammates
    source_id = uuid.uuid4()

    async with user_session(ana) as db:
        await db.execute(
            text(
                "insert into public.data_sources (id, org_id, name, kind, status) "
                "values (:id, :org, 'orders.csv', 'csv', 'idle')"
            ),
            {"id": source_id, "org": org_id},
        )

    ana_store = ContextStore(org_id, ana)
    await ana_store.remember(
        ContextEntry(
            kind=Kind.PROFILE,
            title="Profile",
            content="1.8M rows, 5 columns.",
            source_id=source_id,
        )
    )
    await ana_store.remember(
        ContextEntry(
            kind=Kind.DECISION,
            title="No dedupe",
            content="Repeat orders are legitimate.",
            source_id=source_id,
        )
    )

    ana_briefing = await ana_store.briefing(source_id)
    assert "already known about this source" in ana_briefing
    assert "1.8M rows" in ana_briefing
    assert "you previously decided" in ana_briefing
    assert "Repeat orders are legitimate" in ana_briefing

    # Bo gets the shared half and none of Ana's judgment.
    bo_briefing = await ContextStore(org_id, bo).briefing(source_id)
    assert "1.8M rows" in bo_briefing
    assert "Repeat orders are legitimate" not in bo_briefing


async def test_the_briefing_is_empty_for_an_unseen_source(person):
    store = ContextStore(await _org_of(person), person)
    assert await store.briefing(uuid.uuid4()) == ""


# ── graceful degradation ────────────────────────────────────────────────────


async def test_context_is_still_recorded_when_embeddings_are_disabled(person):
    """Losing semantic search is bad; losing what the agent learned is worse."""
    from lumen.embeddings import get_embedding_provider

    org_id = await _org_of(person)
    store = ContextStore(org_id, person, embedder=get_embedding_provider("none"))

    entry_id = await store.remember(
        ContextEntry(kind=Kind.NOTE, title="No embeddings", content="written anyway")
    )

    async with user_session(person) as db:
        row = (
            await db.execute(
                text("select content, embedding from public.data_contexts where id = :id"),
                {"id": entry_id},
            )
        ).mappings().one()

    assert row["content"] == "written anyway"
    assert row["embedding"] is None, "a null embedding is backfillable; a lost row is not"
