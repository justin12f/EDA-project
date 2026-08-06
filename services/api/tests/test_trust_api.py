"""API-key issuance and the public trust API (ADR-0009 §4), against the live
project. This is the first surface in the product an unauthenticated-by-login
caller reaches — the RLS bridge (`api_keys.authenticate` resolving a live
org member to run the actual read as) is exactly the kind of thing a mock
session could get subtly wrong without ever failing.
"""

from __future__ import annotations

import uuid

import httpx
import pytest
import pytest_asyncio
from sqlalchemy import text

from lumen_api.api_keys import ApiKeyCreate, create_api_key, list_api_keys, revoke_api_key
from lumen_api.auth.dependencies import Identity
from lumen_api.db.session import service_session, user_session
from lumen_api.errors import NotFound, TooManyRequests, Unauthorized
from lumen_api.public import get_glossary_entry, get_public_certification
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
            "email": f"lumen-trust-{uuid.uuid4().hex[:12]}@example.com",
            "password": uuid.uuid4().hex + "Aa1!",
            "email_confirm": True,
            "user_metadata": {"display_name": "Trust Tester", "org_name": "Trust Org"},
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


async def _seed_entity(identity: Identity) -> tuple[uuid.UUID, str]:
    source_id = uuid.uuid4()
    async with user_session(identity.user_id) as db:
        await db.execute(
            text(
                "insert into public.data_sources (id, org_id, name, kind, status) "
                "values (:id, :org, 'watched.csv', 'csv', 'idle')"
            ),
            {"id": source_id, "org": identity.org_id},
        )
        entity_id = (
            await db.execute(
                text(
                    "insert into public.canonical_entities "
                    "(org_id, name, entity_type, reconciliation_rule, status, created_by) "
                    "values (:org, 'customer', 'identifier', '{}'::jsonb, 'approved', :user) "
                    "returning id"
                ),
                {"org": identity.org_id, "user": identity.user_id},
            )
        ).scalar_one()
        await db.execute(
            text(
                "insert into public.canonical_entity_members (entity_id, source_id, column_name) "
                "values (:entity, :source, 'customer_id')"
            ),
            {"entity": entity_id, "source": source_id},
        )
    return source_id, "customer"


async def test_a_freshly_issued_key_authenticates_and_reads_the_glossary(identity):
    source_id, entity_name = await _seed_entity(identity)
    created = await create_api_key(
        ApiKeyCreate(name="BI tool", scope="read:glossary"), identity
    )
    assert created["key"].startswith("lum_")
    assert created["key_prefix"] == created["key"][:12]

    result = await get_glossary_entry(entity_name, authorization=f"Bearer {created['key']}")
    assert result["name"] == "customer"
    assert result["members"] == [
        {"source_id": str(source_id), "source_name": "watched.csv", "column": "customer_id"}
    ]


async def test_a_key_scoped_to_glossary_cannot_read_certification(identity):
    source_id, _ = await _seed_entity(identity)
    created = await create_api_key(
        ApiKeyCreate(name="Glossary-only tool", scope="read:glossary"), identity
    )
    with pytest.raises(Unauthorized):
        await get_public_certification(source_id, authorization=f"Bearer {created['key']}")


async def test_a_revoked_key_is_rejected(identity):
    source_id, entity_name = await _seed_entity(identity)
    created = await create_api_key(
        ApiKeyCreate(name="Short-lived", scope="read:glossary"), identity
    )
    await revoke_api_key(uuid.UUID(created["id"]), identity)

    with pytest.raises(Unauthorized):
        await get_glossary_entry(entity_name, authorization=f"Bearer {created['key']}")


async def test_revoking_an_already_revoked_key_is_not_found(identity):
    created = await create_api_key(
        ApiKeyCreate(name="Once", scope="read:glossary"), identity
    )
    await revoke_api_key(uuid.UUID(created["id"]), identity)
    with pytest.raises(NotFound):
        await revoke_api_key(uuid.UUID(created["id"]), identity)


async def test_listed_keys_never_expose_the_hash_or_raw_key(identity):
    await create_api_key(ApiKeyCreate(name="Listed", scope="read:certification"), identity)
    listing = await list_api_keys(identity)
    assert listing["keys"]
    for key in listing["keys"]:
        assert "key" not in key
        assert "key_hash" not in key


async def test_certification_endpoint_reflects_a_never_checked_source(identity):
    source_id = uuid.uuid4()
    async with user_session(identity.user_id) as db:
        await db.execute(
            text(
                "insert into public.data_sources (id, org_id, name, kind, status) "
                "values (:id, :org, 'unchecked.csv', 'csv', 'idle')"
            ),
            {"id": source_id, "org": identity.org_id},
        )
    created = await create_api_key(
        ApiKeyCreate(name="Certifier", scope="read:certification"), identity
    )

    result = await get_public_certification(source_id, authorization=f"Bearer {created['key']}")
    # Never profiled or scheduled — "no evidence against it" must not read as
    # "certified" (certification.py's own guard on checked_by is None).
    assert result["certified"] is False
    assert result["last_checked_at"] is None


async def test_the_rate_limit_trips_after_the_configured_count(identity, monkeypatch: pytest.MonkeyPatch):
    import lumen_api.public as public_module

    monkeypatch.setattr(public_module, "_RATE_LIMIT_PER_MINUTE", 2)

    source_id, entity_name = await _seed_entity(identity)
    created = await create_api_key(ApiKeyCreate(name="Bursty", scope="read:glossary"), identity)
    auth_header = f"Bearer {created['key']}"

    await get_glossary_entry(entity_name, authorization=auth_header)
    await get_glossary_entry(entity_name, authorization=auth_header)
    with pytest.raises(TooManyRequests):
        await get_glossary_entry(entity_name, authorization=auth_header)


async def test_every_authenticated_request_is_audit_logged(identity):
    source_id, entity_name = await _seed_entity(identity)
    created = await create_api_key(ApiKeyCreate(name="Logged", scope="read:glossary"), identity)
    key_id = uuid.UUID(created["id"])

    await get_glossary_entry(entity_name, authorization=f"Bearer {created['key']}")
    with pytest.raises(NotFound):
        await get_glossary_entry("no-such-entity", authorization=f"Bearer {created['key']}")

    async with service_session() as db:
        rows = (
            await db.execute(
                text(
                    "select endpoint, status_code from public.api_key_audit_log "
                    "where api_key_id = :key order by requested_at"
                ),
                {"key": key_id},
            )
        ).mappings().all()
    assert [r["status_code"] for r in rows] == [200, 404]
