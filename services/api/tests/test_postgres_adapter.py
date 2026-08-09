"""PostgresAdapter against a real database.

Uses the tenant instance as a stand-in for a customer's database — the
adapter cannot tell the difference, which is the point.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import text

from lumen.architect.adapters.postgres import PostgresAdapter
from lumen.architect.spec import SqlType
from lumen_api.settings import get_settings
from lumen_api.tenant_db import get_tenant_engine

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not get_settings().has_tenant_db,
        reason="TENANT_DATABASE_URL is not configured",
    ),
]


@pytest.fixture
async def sample_schema():
    name = f"adapter_test_{uuid.uuid4().hex[:8]}"
    engine = get_tenant_engine()
    async with engine.begin() as conn:
        await conn.execute(text(f'CREATE SCHEMA "{name}"'))
        await conn.execute(
            text(
                f'CREATE TABLE "{name}".customers ('
                "  id text PRIMARY KEY,"
                "  name varchar(120) NOT NULL,"
                "  balance numeric(12,2)"
                ")"
            )
        )
        await conn.execute(
            text(
                f'CREATE TABLE "{name}".orders ('
                "  id bigint PRIMARY KEY,"
                f'  customer_id text REFERENCES "{name}".customers (id)'
                ")"
            )
        )
        await conn.execute(text(f"INSERT INTO \"{name}\".customers VALUES ('c1', 'Acme', 10.00)"))
    try:
        yield name
    finally:
        async with engine.begin() as conn:
            await conn.execute(text(f'DROP SCHEMA "{name}" CASCADE'))


def _dsn() -> str:
    return get_settings().tenant_database_url.get_secret_value()


async def test_discovery_is_marked_declared(sample_schema):
    structure = await PostgresAdapter(_dsn(), sample_schema).discover()
    assert structure.declared is True


async def test_every_table_is_found(sample_schema):
    structure = await PostgresAdapter(_dsn(), sample_schema).discover()
    assert {t.name for t in structure.tables} == {"customers", "orders"}


async def test_the_real_primary_key_is_read_not_inferred(sample_schema):
    structure = await PostgresAdapter(_dsn(), sample_schema).discover()
    customers = next(t for t in structure.tables if t.name == "customers")
    assert customers.primary_key == ("id",)


async def test_the_real_foreign_key_is_read(sample_schema):
    """This is the whole argument for D10: when the source is a database,
    the relationship is read, not guessed at 95% containment."""
    structure = await PostgresAdapter(_dsn(), sample_schema).discover()
    orders = next(t for t in structure.tables if t.name == "orders")
    assert orders.foreign_keys == (("customer_id", "customers", "id"),)


async def test_types_map_onto_the_closed_enum(sample_schema):
    structure = await PostgresAdapter(_dsn(), sample_schema).discover()
    customers = next(t for t in structure.tables if t.name == "customers")
    types = {c.name: (c.sql_type, c.type_arg) for c in customers.columns}
    assert types["id"] == (SqlType.TEXT, None)
    assert types["name"] == (SqlType.VARCHAR, "120")
    assert types["balance"] == (SqlType.NUMERIC, "12,2")


async def test_nullability_is_read(sample_schema):
    structure = await PostgresAdapter(_dsn(), sample_schema).discover()
    customers = next(t for t in structure.tables if t.name == "customers")
    assert next(c for c in customers.columns if c.name == "name").nullable is False


async def test_read_returns_rows(sample_schema):
    frame = await PostgresAdapter(_dsn(), sample_schema).read("customers")
    assert frame.height == 1


async def test_reading_an_undiscovered_table_is_refused(sample_schema):
    """A table name cannot be parameterised, so it is interpolated — which
    makes validating it against the discovered list the only thing standing
    between this and injection."""
    adapter = PostgresAdapter(_dsn(), sample_schema)
    await adapter.discover()
    with pytest.raises(ValueError, match="not discovered"):
        await adapter.read('customers"; DROP TABLE customers; --')
