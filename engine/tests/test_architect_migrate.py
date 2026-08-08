"""Schema evolution — what the agent may do alone, and what it may never do."""

from __future__ import annotations

import uuid

import pytest

from lumen.architect.migrate import classify_migration, render_migration
from lumen.architect.spec import ColumnSpec, SchemaSpec, SqlType, TableSpec

_SRC = uuid.uuid4()


def _spec(*columns: tuple[str, SqlType]) -> SchemaSpec:
    return SchemaSpec(
        tables=(
            TableSpec(
                name="orders",
                source_id=_SRC,
                columns=tuple(
                    ColumnSpec(name=n, source_column=n, sql_type=t) for n, t in columns
                ),
                primary_key=None,
                pk_rationale="",
            ),
        )
    )


def test_a_new_column_is_additive_and_reversible():
    plan = classify_migration(_spec(("id", SqlType.TEXT)),
                              _spec(("id", SqlType.TEXT), ("email", SqlType.TEXT)))
    assert [s.kind for s in plan.steps] == ["add_column"]
    assert plan.reversible is True


def test_a_widening_type_change_is_reversible():
    plan = classify_migration(_spec(("n", SqlType.INTEGER)), _spec(("n", SqlType.BIGINT)))
    assert [s.kind for s in plan.steps] == ["widen_type"]
    assert plan.reversible is True


@pytest.mark.parametrize(
    "old,new",
    [
        (SqlType.INTEGER, SqlType.BIGINT),
        (SqlType.INTEGER, SqlType.NUMERIC),
        (SqlType.VARCHAR, SqlType.TEXT),
        (SqlType.DOUBLE, SqlType.TEXT),
    ],
)
def test_widening_pairs(old, new):
    plan = classify_migration(_spec(("n", old)), _spec(("n", new)))
    assert plan.reversible is True


def test_a_narrowing_type_change_is_not_reversible():
    plan = classify_migration(_spec(("n", SqlType.TEXT)), _spec(("n", SqlType.INTEGER)))
    assert [s.kind for s in plan.steps] == ["narrow_type"]
    assert plan.reversible is False


def test_a_column_absent_at_origin_is_deprecated_not_dropped():
    plan = classify_migration(_spec(("id", SqlType.TEXT), ("legacy", SqlType.TEXT)),
                              _spec(("id", SqlType.TEXT)))
    assert [s.kind for s in plan.steps] == ["deprecate_column"]
    assert plan.reversible is True


def test_an_unchanged_schema_produces_an_empty_plan():
    plan = classify_migration(_spec(("id", SqlType.TEXT)), _spec(("id", SqlType.TEXT)))
    assert plan.steps == ()
    assert plan.reversible is True


def test_render_migration_never_emits_drop_column():
    """D7's hard rule. The test exists because this is the single line that,
    if it regressed, would silently make the product destructive."""
    plan = classify_migration(_spec(("id", SqlType.TEXT), ("legacy", SqlType.TEXT)),
                              _spec(("id", SqlType.TEXT)))
    for statement in render_migration(plan, "tenant_abc"):
        assert "DROP" not in statement.upper()


def test_render_migration_adds_a_column():
    plan = classify_migration(_spec(("id", SqlType.TEXT)),
                              _spec(("id", SqlType.TEXT), ("email", SqlType.TEXT)))
    assert render_migration(plan, "tenant_abc") == [
        'ALTER TABLE "tenant_abc"."orders" ADD COLUMN IF NOT EXISTS "email" text'
    ]


def test_render_migration_widens_a_type():
    plan = classify_migration(_spec(("n", SqlType.INTEGER)), _spec(("n", SqlType.BIGINT)))
    assert render_migration(plan, "tenant_abc") == [
        'ALTER TABLE "tenant_abc"."orders" ALTER COLUMN "n" TYPE bigint'
    ]
