"""The read tier. The parity test is the one that matters: a tiering that
changes results is a correctness bug, not a performance knob."""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import text

from lumen.datasets.sql_read import read_table
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
async def sample_table():
    schema = f"read_test_{uuid.uuid4().hex[:8]}"
    engine = get_tenant_engine()
    async with engine.begin() as conn:
        await conn.execute(text(f'CREATE SCHEMA "{schema}"'))
        await conn.execute(
            text(f'CREATE TABLE "{schema}".rows (id bigint, label text, amount numeric(12,2))')
        )
        for i in range(50):
            await conn.execute(
                text(f'INSERT INTO "{schema}".rows VALUES (:i, :label, :amount)'),
                {"i": i, "label": f"row-{i}", "amount": i * 1.5},
            )
    try:
        yield schema
    finally:
        async with engine.begin() as conn:
            await conn.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))


def _dsn() -> str:
    return get_settings().tenant_database_url.get_secret_value()


def test_the_polars_path_reads_every_row(sample_table):
    frame = read_table(_dsn(), sample_table, "rows", row_count=50, threshold=1_000_000)
    assert frame.height == 50


def test_the_duckdb_path_reads_every_row(sample_table):
    frame = read_table(_dsn(), sample_table, "rows", row_count=50, threshold=1)
    assert frame.height == 50


def test_both_paths_produce_identical_results(sample_table):
    """Forced by passing an explicit threshold rather than by generating
    five million rows."""
    low = read_table(_dsn(), sample_table, "rows", row_count=50, threshold=1_000_000)
    high = read_table(_dsn(), sample_table, "rows", row_count=50, threshold=1)

    assert low.columns == high.columns
    assert low.height == high.height
    assert low.sort("id").to_dicts() == high.sort("id").to_dicts()


def test_an_unknown_row_count_takes_the_polars_path(sample_table):
    """Unknown size is the common case on a first read. Defaulting to the
    simpler path means the accelerator is opt-in on evidence."""
    frame = read_table(_dsn(), sample_table, "rows", row_count=None)
    assert frame.height == 50
