"""Enrichment must never be load-bearing.

The deterministic spec is the product; the model makes it readable. If the
provider is absent, denied or broken, the spec ships unchanged — which is
what keeps the keyless end-to-end path working.
"""

from __future__ import annotations

import uuid

import pytest

from lumen.architect.spec import ColumnSpec, SchemaSpec, SqlType, TableSpec
from lumen_api.architect import enrich_spec

_SRC = uuid.uuid4()


def _spec() -> SchemaSpec:
    return SchemaSpec(
        tables=(
            TableSpec(
                name="orders",
                source_id=_SRC,
                columns=(
                    ColumnSpec(name="id", source_column="id", sql_type=SqlType.TEXT),
                ),
                primary_key=("id",),
                pk_rationale="unique and complete",
            ),
        )
    )


async def test_a_failing_provider_returns_the_spec_unchanged(monkeypatch):
    def explode(*args, **kwargs):
        raise RuntimeError("provider is down")

    monkeypatch.setattr("lumen_api.architect.provider", explode)
    assert await enrich_spec(_spec()) == _spec()


async def test_a_provider_returning_nonsense_returns_the_spec_unchanged(monkeypatch):
    class _Garbage:
        async def complete(self, *args, **kwargs):
            class _R:
                text = "not json at all"
            return _R()

    monkeypatch.setattr("lumen_api.architect.provider", lambda *a, **k: _Garbage())
    assert await enrich_spec(_spec()) == _spec()


async def test_enrichment_never_changes_structure(monkeypatch):
    """A model may improve prose. It may not invent a column, drop one, or
    change a type — those are the deterministic layer's decisions."""
    class _Meddling:
        async def complete(self, *args, **kwargs):
            class _R:
                text = '{"tables": {"orders": {"pk_rationale": "the order identifier"}}}'
            return _R()

    monkeypatch.setattr("lumen_api.architect.provider", lambda *a, **k: _Meddling())
    enriched = await enrich_spec(_spec())

    assert enriched.tables[0].pk_rationale == "the order identifier"
    assert enriched.tables[0].columns == _spec().tables[0].columns
    assert enriched.tables[0].name == "orders"
    assert enriched.tables[0].primary_key == ("id",)
