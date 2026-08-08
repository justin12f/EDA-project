"""SchemaSpec validation — the invariants every later stage assumes hold."""

from __future__ import annotations

import uuid

import pytest

from lumen.architect.spec import (
    ColumnSpec,
    Evidence,
    ForeignKeySpec,
    SchemaSpec,
    SpecError,
    SqlType,
    TableSpec,
)

SOURCE = uuid.uuid4()


def _table(name: str, *columns: str, pk: tuple[str, ...] | None = None) -> TableSpec:
    return TableSpec(
        name=name,
        source_id=SOURCE,
        columns=tuple(
            ColumnSpec(name=c, source_column=c, sql_type=SqlType.TEXT) for c in columns
        ),
        primary_key=pk,
        pk_rationale="test fixture",
    )


def test_a_minimal_valid_spec_passes():
    SchemaSpec(tables=(_table("orders", "id", "customer_id", pk=("id",)),)).validate()


def test_duplicate_table_names_are_rejected():
    spec = SchemaSpec(tables=(_table("orders", "id"), _table("orders", "id")))
    with pytest.raises(SpecError, match="duplicate table"):
        spec.validate()


def test_duplicate_column_names_within_a_table_are_rejected():
    spec = SchemaSpec(tables=(_table("orders", "id", "id"),))
    with pytest.raises(SpecError, match="duplicate column"):
        spec.validate()


def test_a_primary_key_naming_an_unknown_column_is_rejected():
    spec = SchemaSpec(tables=(_table("orders", "id", pk=("missing",)),))
    with pytest.raises(SpecError, match="primary key"):
        spec.validate()


def test_a_foreign_key_referencing_an_unknown_table_is_rejected():
    spec = SchemaSpec(
        tables=(_table("orders", "id", "customer_id", pk=("id",)),),
        foreign_keys=(
            ForeignKeySpec(
                from_table="orders",
                from_column="customer_id",
                to_table="customers",
                to_column="id",
                containment=1.0,
                enforced=True,
                evidence=(Evidence.STRUCTURAL,),
                rationale="test",
            ),
        ),
    )
    with pytest.raises(SpecError, match="unknown table"):
        spec.validate()


def test_a_foreign_key_referencing_an_unknown_column_is_rejected():
    spec = SchemaSpec(
        tables=(
            _table("orders", "id", "customer_id", pk=("id",)),
            _table("customers", "id", pk=("id",)),
        ),
        foreign_keys=(
            ForeignKeySpec(
                from_table="orders",
                from_column="nope",
                to_table="customers",
                to_column="id",
                containment=1.0,
                enforced=True,
                evidence=(Evidence.STRUCTURAL,),
                rationale="test",
            ),
        ),
    )
    with pytest.raises(SpecError, match="unknown column"):
        spec.validate()


@pytest.mark.parametrize("containment", [-0.1, 1.1])
def test_containment_outside_zero_to_one_is_rejected(containment):
    spec = SchemaSpec(
        tables=(
            _table("orders", "id", "customer_id", pk=("id",)),
            _table("customers", "id", pk=("id",)),
        ),
        foreign_keys=(
            ForeignKeySpec(
                from_table="orders",
                from_column="customer_id",
                to_table="customers",
                to_column="id",
                containment=containment,
                enforced=False,
                evidence=(Evidence.STRUCTURAL,),
                rationale="test",
            ),
        ),
    )
    with pytest.raises(SpecError, match="containment"):
        spec.validate()


def test_an_enforced_key_with_partial_containment_is_rejected():
    """Decision D5's core invariant: Postgres cannot enforce a relationship
    the data does not fully satisfy, so the spec must not be able to claim it."""
    spec = SchemaSpec(
        tables=(
            _table("orders", "id", "customer_id", pk=("id",)),
            _table("customers", "id", pk=("id",)),
        ),
        foreign_keys=(
            ForeignKeySpec(
                from_table="orders",
                from_column="customer_id",
                to_table="customers",
                to_column="id",
                containment=0.97,
                enforced=True,
                evidence=(Evidence.STRUCTURAL,),
                rationale="test",
            ),
        ),
    )
    with pytest.raises(SpecError, match="enforced"):
        spec.validate()


def test_partial_containment_is_fine_when_not_enforced():
    SchemaSpec(
        tables=(
            _table("orders", "id", "customer_id", pk=("id",)),
            _table("customers", "id", pk=("id",)),
        ),
        foreign_keys=(
            ForeignKeySpec(
                from_table="orders",
                from_column="customer_id",
                to_table="customers",
                to_column="id",
                containment=0.97,
                enforced=False,
                evidence=(Evidence.STRUCTURAL,),
                rationale="test",
            ),
        ),
    ).validate()
