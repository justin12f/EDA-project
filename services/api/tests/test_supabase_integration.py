"""End-to-end checks against a live Supabase project.

Marked `integration` and excluded from the default run, because they need real
credentials and they create and delete real rows. Run them with:

    uv run --directory services/api pytest tests -m integration -v

What these cover that a unit test cannot: that `handle_new_user` actually fires,
that `SET LOCAL ROLE authenticated` plus JWT claims actually makes `auth.uid()`
resolve, and that row-level security actually stops one organization reading
another's rows. Those are the three assumptions the whole tenancy model rests
on, and all three live in the database rather than in Python.

Every test cleans up the auth user it creates; deleting the user cascades to the
profile, membership and organization.
"""

from __future__ import annotations

import uuid

import httpx
import pytest
from sqlalchemy import text

from lumen_api.db.session import service_session, user_session
from lumen_api.settings import get_settings

pytestmark = pytest.mark.integration


def _admin_headers() -> dict[str, str]:
    key = get_settings().supabase_service_role_key.get_secret_value()
    return {"apikey": key, "Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _create_user(display_name: str) -> tuple[uuid.UUID, str]:
    """Create a confirmed auth user through the admin API. Returns (id, email)."""
    settings = get_settings()
    email = f"lumen-test-{uuid.uuid4().hex[:12]}@example.com"
    response = httpx.post(
        f"{settings.supabase_url}/auth/v1/admin/users",
        headers=_admin_headers(),
        json={
            "email": email,
            "password": uuid.uuid4().hex + "Aa1!",
            "email_confirm": True,
            "user_metadata": {"display_name": display_name},
        },
        timeout=30,
    )
    response.raise_for_status()
    return uuid.UUID(response.json()["id"]), email


def _delete_user(user_id: uuid.UUID) -> None:
    settings = get_settings()
    httpx.delete(
        f"{settings.supabase_url}/auth/v1/admin/users/{user_id}",
        headers=_admin_headers(),
        timeout=30,
    )


@pytest.fixture
def person():
    user_id, email = _create_user("Ana Kovač")
    yield user_id, email
    _delete_user(user_id)


@pytest.fixture
def two_people():
    a, _ = _create_user("Ana")
    b, _ = _create_user("Bo")
    yield a, b
    _delete_user(a)
    _delete_user(b)


# ── connectivity ────────────────────────────────────────────────────────────


async def test_the_database_is_reachable():
    async with service_session() as db:
        assert (await db.execute(text("select 1"))).scalar_one() == 1


async def test_pgvector_and_the_search_function_exist():
    async with service_session() as db:
        assert (
            await db.execute(text("select count(*) from pg_extension where extname='vector'"))
        ).scalar_one() == 1
        assert (
            await db.execute(
                text("select count(*) from pg_proc where proname='match_data_contexts'")
            )
        ).scalar_one() == 1


# ── the signup trigger ──────────────────────────────────────────────────────


async def test_a_new_auth_user_gets_a_profile_org_and_owner_membership(person):
    user_id, email = person

    async with service_session() as db:
        profile = (
            await db.execute(
                text("select email, display_name from public.profiles where id = :id"),
                {"id": user_id},
            )
        ).mappings().first()
        assert profile is not None, "handle_new_user did not create a profile"
        assert profile["email"] == email
        assert profile["display_name"] == "Ana Kovač"

        membership = (
            await db.execute(
                text(
                    "select m.role, o.name, o.plan_code from public.memberships m "
                    "join public.organizations o on o.id = m.org_id where m.user_id = :id"
                ),
                {"id": user_id},
            )
        ).mappings().first()
        assert membership is not None, "handle_new_user did not create an organization"
        assert membership["role"] == "owner"
        assert membership["plan_code"] == "free"


async def test_current_identity_resolves_inside_the_users_own_session(person):
    user_id, email = person

    async with user_session(user_id) as db:
        row = (await db.execute(text("select * from public.current_identity()"))).mappings().first()

    assert row is not None
    assert row["user_id"] == user_id
    assert row["email"] == email
    assert row["role"] == "owner"


async def test_auth_uid_resolves_from_the_session_claims(person):
    user_id, _ = person
    async with user_session(user_id) as db:
        assert (await db.execute(text("select auth.uid()"))).scalar_one() == user_id


# ── row-level security ──────────────────────────────────────────────────────


async def test_one_org_cannot_read_anothers_data_sources(two_people):
    ana, bo = two_people

    async with user_session(ana) as db:
        ana_org = (
            await db.execute(text("select org_id from public.memberships where user_id = :id"), {"id": ana})
        ).scalar_one()
        await db.execute(
            text(
                "insert into public.data_sources (org_id, name, kind, status) "
                "values (:org, 'ana-secret.csv', 'csv', 'idle')"
            ),
            {"org": ana_org},
        )

    async with user_session(ana) as db:
        assert (await db.execute(text("select count(*) from public.data_sources"))).scalar_one() == 1

    async with user_session(bo) as db:
        assert (await db.execute(text("select count(*) from public.data_sources"))).scalar_one() == 0


async def test_a_write_into_another_org_is_refused(two_people):
    ana, bo = two_people

    async with user_session(ana) as db:
        ana_org = (
            await db.execute(text("select org_id from public.memberships where user_id = :id"), {"id": ana})
        ).scalar_one()

    with pytest.raises(Exception) as caught:
        async with user_session(bo) as db:
            await db.execute(
                text(
                    "insert into public.data_sources (org_id, name, kind, status) "
                    "values (:org, 'bo-intrusion.csv', 'csv', 'idle')"
                ),
                {"org": ana_org},
            )
    assert "row-level security" in str(caught.value).lower()


async def test_every_org_scoped_table_has_rls_enabled():
    """The invariant the whole tenancy model rests on, asserted against the catalogue."""
    async with service_session() as db:
        offenders = [
            row[0]
            for row in await db.execute(
                text(
                    """
                    select c.relname
                    from pg_class c
                    join pg_namespace n on n.oid = c.relnamespace
                    join information_schema.columns col
                      on col.table_name = c.relname and col.table_schema = n.nspname
                    where n.nspname = 'public'
                      and c.relkind = 'r'
                      and col.column_name = 'org_id'
                      and c.relrowsecurity = false
                    """
                )
            )
        ]
    assert offenders == [], f"tables carrying org_id but no RLS: {offenders}"


# ── the vector context store ────────────────────────────────────────────────


async def test_a_context_row_round_trips_through_similarity_search(person):
    user_id, _ = person

    async with user_session(user_id) as db:
        org_id = (
            await db.execute(text("select org_id from public.memberships where user_id = :id"), {"id": user_id})
        ).scalar_one()

        embedding = "[" + ",".join(["0.1"] * 384) + "]"
        await db.execute(
            text(
                "insert into public.data_contexts (org_id, kind, title, content, embedding) "
                "values (:org, 'profile', 'users profile', "
                "'3.2% nulls in country_code across 12.4M rows', :embedding)"
            ),
            {"org": org_id, "embedding": embedding},
        )

    async with user_session(user_id) as db:
        rows = (
            await db.execute(
                text(
                    "select title, similarity from public.match_data_contexts("
                    "  cast(:embedding as extensions.vector), :org, 5, 0.0)"
                ),
                {"embedding": "[" + ",".join(["0.1"] * 384) + "]", "org": org_id},
            )
        ).mappings().all()

    assert len(rows) == 1
    assert rows[0]["title"] == "users profile"
    assert rows[0]["similarity"] > 0.99, "an identical vector must score ~1.0"
