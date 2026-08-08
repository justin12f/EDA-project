"""Tenant isolation against the live instance.

The two tests that matter here are (b) and (c). Everything else in this
plan is a feature; these are the reason the feature is safe to ship.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import text

from lumen_api.settings import get_settings
from lumen_api.tenant_db import (
    ensure_tenant_schema,
    get_tenant_engine,
    tenant_schema_name,
    tenant_session,
)

pytestmark = pytest.mark.integration

# Skip rather than error when no instance is configured — a developer
# without one should see a clear skip, not a confusing connection failure.
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not get_settings().has_tenant_db,
        reason="TENANT_DATABASE_URL is not configured",
    ),
]


@pytest.fixture
def org_a() -> uuid.UUID:
    return uuid.uuid4()


@pytest.fixture
def org_b() -> uuid.UUID:
    return uuid.uuid4()


async def _drop(org_id: uuid.UUID) -> None:
    engine = get_tenant_engine()
    async with engine.begin() as conn:
        await conn.execute(text(f'DROP SCHEMA IF EXISTS "{tenant_schema_name(org_id)}" CASCADE'))
        await conn.execute(text(f'DROP SCHEMA IF EXISTS "{tenant_schema_name(org_id)}_raw" CASCADE'))
        await conn.execute(text(f'DROP ROLE IF EXISTS "{tenant_schema_name(org_id)}_role"'))


async def test_provisioning_is_idempotent(org_a):
    try:
        await ensure_tenant_schema(org_a)
        await ensure_tenant_schema(org_a)  # must not raise

        engine = get_tenant_engine()
        async with engine.connect() as conn:
            count = (
                await conn.execute(
                    text(
                        "select count(*) from information_schema.schemata "
                        "where schema_name = :name"
                    ),
                    {"name": tenant_schema_name(org_a)},
                )
            ).scalar_one()
        assert count == 1
    finally:
        await _drop(org_a)


async def test_one_org_cannot_read_another_orgs_schema(org_a, org_b):
    """The critical test. Asserting on the RAISE matters: a version that
    merely returned zero rows would pass against a broken implementation
    that granted access to an empty schema."""
    try:
        await ensure_tenant_schema(org_a)
        await ensure_tenant_schema(org_b)

        engine = get_tenant_engine()
        async with engine.begin() as conn:
            await conn.execute(
                text(f'CREATE TABLE "{tenant_schema_name(org_b)}".secrets (v text)')
            )
            await conn.execute(
                text(f"INSERT INTO \"{tenant_schema_name(org_b)}\".secrets VALUES ('classified')")
            )

        with pytest.raises(Exception) as caught:
            async with tenant_session(org_a) as db:
                await db.execute(text(f'SELECT * FROM "{tenant_schema_name(org_b)}".secrets'))
        assert "permission denied" in str(caught.value).lower()
    finally:
        await _drop(org_a)
        await _drop(org_b)


async def test_the_tenant_instance_holds_no_control_plane_table():
    """Verifies the separation as a fact about the instance rather than
    trusting that it was configured correctly."""
    engine = get_tenant_engine()
    async with engine.connect() as conn:
        rows = (
            await conn.execute(
                text(
                    "select table_name from information_schema.tables "
                    "where table_name in "
                    "('organizations', 'api_keys', 'subscriptions', 'memberships', 'proposals')"
                )
            )
        ).scalars().all()
    assert rows == [], f"control-plane tables found on the tenant instance: {rows}"
