"""Identifier sanitisation — the only place in the system where untrusted
text becomes SQL, so the only place that needs injection review."""

from __future__ import annotations

import pytest

from lumen.architect.ddl import sanitize_identifier
from lumen.architect.spec import SpecError


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("orders", "orders"),
        ("Orders", "orders"),
        ("Customer ID", "customer_id"),
        ("customer-id", "customer_id"),
        ("customer..id", "customer_id"),
        ("  spaced  ", "spaced"),
        ("2024_revenue", "col_2024_revenue"),
        ("café", "caf"),
    ],
)
def test_sanitisation_cases(raw, expected):
    assert sanitize_identifier(raw) == expected


def test_a_reserved_word_is_suffixed_not_quoted_away():
    """Quoting would work, but a column a customer has to write as "select"
    in every query is a papercut we can spend one underscore to avoid."""
    assert sanitize_identifier("select") == "select_col"
    assert sanitize_identifier("ORDER") == "order_col"


def test_collisions_get_a_numeric_suffix():
    taken: set[str] = set()
    first = sanitize_identifier("Customer ID", taken=taken)
    taken.add(first)
    second = sanitize_identifier("customer.id", taken=taken)
    taken.add(second)
    third = sanitize_identifier("CUSTOMER-ID", taken=taken)
    assert [first, second, third] == ["customer_id", "customer_id_2", "customer_id_3"]


def test_a_long_name_is_truncated_to_the_postgres_limit():
    result = sanitize_identifier("a" * 200)
    assert len(result.encode("utf-8")) == 63


def test_a_long_name_built_from_a_multibyte_source_still_fits_the_limit():
    """Sanitisation always collapses non-ASCII content to underscores before
    truncation ever runs, so `collapsed` is pure ASCII by the time
    `_truncate_bytes` sees it and byte length equals character length there.
    The real risk this guards is upstream: a naive `raw[:63]` on the
    *original* multibyte string could split a character in half. Truncating
    the already-ASCII `collapsed` string sidesteps that entirely, which this
    proves by using a long, realistic multibyte-sourced name and asserting
    the result both fits and is still a valid identifier."""
    import re

    raw = ("a" + "ñ") * 50  # alternating valid/invalid -> a long collapsed name
    result = sanitize_identifier(raw)
    assert len(result.encode("utf-8")) <= 63
    assert re.fullmatch(r"[a-z_][a-z0-9_]*", result)


def test_a_name_with_nothing_usable_raises():
    with pytest.raises(SpecError, match="cannot be sanitised"):
        sanitize_identifier("!!!")


def test_an_empty_name_raises():
    with pytest.raises(SpecError, match="cannot be sanitised"):
        sanitize_identifier("")


def test_injection_attempts_are_neutralised():
    assert sanitize_identifier('x"; drop table users; --') == "x_drop_table_users"


def test_every_result_matches_the_validation_pattern():
    import re

    for raw in ["orders", "Customer ID", "2024", "select", "a" * 200, "ñoño"]:
        assert re.fullmatch(r"[a-z_][a-z0-9_]*", sanitize_identifier(raw))


# ── DDL rendering ───────────────────────────────────────────────────────

import uuid  # noqa: E402

from lumen.architect.ddl import render_ddl, render_replace  # noqa: E402
from lumen.architect.spec import (  # noqa: E402
    ColumnSpec,
    Evidence,
    ForeignKeySpec,
    SchemaSpec,
    SqlType,
    TableSpec,
)

_SRC = uuid.uuid4()


def _customers() -> TableSpec:
    return TableSpec(
        name="customers",
        source_id=_SRC,
        columns=(
            ColumnSpec(name="id", source_column="id", sql_type=SqlType.TEXT, nullable=False),
            ColumnSpec(name="amount", source_column="amount", sql_type=SqlType.NUMERIC, type_arg="12,2"),
        ),
        primary_key=("id",),
        pk_rationale="",
    )


def _orders() -> TableSpec:
    return TableSpec(
        name="orders",
        source_id=_SRC,
        columns=(
            ColumnSpec(name="id", source_column="id", sql_type=SqlType.TEXT, nullable=False),
            ColumnSpec(name="customer_id", source_column="customer_id", sql_type=SqlType.TEXT),
        ),
        primary_key=("id",),
        pk_rationale="",
    )


def _fk(enforced: bool, containment: float) -> ForeignKeySpec:
    return ForeignKeySpec(
        from_table="orders",
        from_column="customer_id",
        to_table="customers",
        to_column="id",
        containment=containment,
        enforced=enforced,
        evidence=(Evidence.STRUCTURAL,),
        rationale="",
    )


def test_create_schema_comes_first():
    statements = render_ddl(SchemaSpec(tables=(_customers(),)), "tenant_abc")
    assert statements[0] == 'CREATE SCHEMA IF NOT EXISTS "tenant_abc"'


def test_a_table_renders_with_quoted_identifiers_and_real_types():
    statements = render_ddl(SchemaSpec(tables=(_customers(),)), "tenant_abc")
    assert statements[1] == (
        'CREATE TABLE IF NOT EXISTS "tenant_abc"."customers" (\n'
        '  "id" text NOT NULL,\n'
        '  "amount" numeric(12,2)\n'
        ')'
    )


def test_a_primary_key_renders_as_its_own_constraint():
    statements = render_ddl(SchemaSpec(tables=(_customers(),)), "tenant_abc")
    assert any(
        s == 'ALTER TABLE "tenant_abc"."customers" '
             'ADD CONSTRAINT "customers_pkey" PRIMARY KEY ("id")'
        for s in statements
    )


def test_an_enforced_key_is_deferrable():
    """SET CONSTRAINTS ALL DEFERRED only works on constraints declared
    DEFERRABLE, and D6's replace-in-one-transaction depends on it."""
    spec = SchemaSpec(tables=(_customers(), _orders()), foreign_keys=(_fk(True, 1.0),))
    statements = render_ddl(spec, "tenant_abc")
    fk = [s for s in statements if "FOREIGN KEY" in s]
    assert len(fk) == 1
    assert fk[0] == (
        'ALTER TABLE "tenant_abc"."orders" '
        'ADD CONSTRAINT "orders_customer_id_fkey" '
        'FOREIGN KEY ("customer_id") REFERENCES "tenant_abc"."customers" ("id") '
        'DEFERRABLE INITIALLY IMMEDIATE'
    )


def test_an_observed_key_emits_no_ddl_at_all():
    spec = SchemaSpec(tables=(_customers(), _orders()), foreign_keys=(_fk(False, 0.97),))
    statements = render_ddl(spec, "tenant_abc")
    assert not any("FOREIGN KEY" in s for s in statements)


def test_a_deprecated_column_is_still_created():
    """D7 keeps deprecated columns; dropping them is what it forbids."""
    table = TableSpec(
        name="orders",
        source_id=_SRC,
        columns=(
            ColumnSpec(name="id", source_column="id", sql_type=SqlType.TEXT, nullable=False),
            ColumnSpec(name="legacy", source_column="legacy", sql_type=SqlType.TEXT, deprecated=True),
        ),
        primary_key=("id",),
        pk_rationale="",
    )
    statements = render_ddl(SchemaSpec(tables=(table,)), "tenant_abc")
    assert '"legacy" text' in statements[1]


def test_render_replace_defers_constraints_before_deleting():
    spec = SchemaSpec(tables=(_customers(), _orders()), foreign_keys=(_fk(True, 1.0),))
    statements = render_replace(spec, "tenant_abc", "customers")
    assert statements == [
        "SET CONSTRAINTS ALL DEFERRED",
        'DELETE FROM "tenant_abc"."customers"',
    ]


def test_no_rendered_statement_ever_contains_drop():
    spec = SchemaSpec(tables=(_customers(), _orders()), foreign_keys=(_fk(True, 1.0),))
    for statement in render_ddl(spec, "tenant_abc") + render_replace(spec, "tenant_abc", "customers"):
        assert "DROP" not in statement.upper()
