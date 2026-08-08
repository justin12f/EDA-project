"""The typed schema contract.

The agent never emits SQL. It emits one of these, and `ddl.render_ddl`
turns it into statements. That indirection is the whole safety argument:
a model can propose a shape, but it cannot propose syntax.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal


class SpecError(ValueError):
    """A spec that would produce invalid or dishonest DDL."""


class SqlType(StrEnum):
    """A closed vocabulary. Types are never interpolated from model output —
    a value not in this enum cannot reach `render_ddl`."""

    TEXT = "text"
    VARCHAR = "varchar"
    INTEGER = "integer"
    BIGINT = "bigint"
    NUMERIC = "numeric"
    DOUBLE = "double precision"
    BOOLEAN = "boolean"
    DATE = "date"
    TIMESTAMP = "timestamp"
    TIMESTAMPTZ = "timestamptz"
    UUID = "uuid"
    JSONB = "jsonb"


class Evidence(StrEnum):
    """Why we believe a relationship exists, strongest first.

    DECLARED outranks everything: it means the customer's own database
    already declares the constraint, so there is nothing to infer.
    """

    DECLARED = "declared"
    STRUCTURAL = "structural"
    SEMANTIC = "semantic"
    NAMING = "naming"


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    source_column: str
    sql_type: SqlType
    # "255" for varchar(255); "12,2" for numeric(12,2). None for types that
    # take no argument, which is most of them.
    type_arg: str | None = None
    nullable: bool = True
    # Absent at origin but retained. D7 forbids DROP COLUMN outright, which
    # is what keeps almost every migration reversible.
    deprecated: bool = False


@dataclass(frozen=True)
class TableSpec:
    name: str
    source_id: uuid.UUID
    columns: tuple[ColumnSpec, ...]
    primary_key: tuple[str, ...] | None = None
    pk_rationale: str = ""
    source_table: str | None = None


@dataclass(frozen=True)
class ForeignKeySpec:
    from_table: str
    from_column: str
    to_table: str
    to_column: str
    # Fraction of child values present in the parent. 1.0 is the only value
    # that may be enforced — see SchemaSpec.validate.
    containment: float
    enforced: bool
    evidence: tuple[Evidence, ...]
    rationale: str


@dataclass(frozen=True)
class SchemaSpec:
    tables: tuple[TableSpec, ...]
    foreign_keys: tuple[ForeignKeySpec, ...] = ()
    layout: Literal["merged", "namespaced"] = "merged"

    def validate(self) -> None:
        by_name: dict[str, TableSpec] = {}
        for table in self.tables:
            if table.name in by_name:
                raise SpecError(f"duplicate table name '{table.name}'")
            by_name[table.name] = table

            seen: set[str] = set()
            for column in table.columns:
                if column.name in seen:
                    raise SpecError(
                        f"duplicate column '{column.name}' in table '{table.name}'"
                    )
                seen.add(column.name)

            for key_column in table.primary_key or ():
                if key_column not in seen:
                    raise SpecError(
                        f"primary key of '{table.name}' names unknown column '{key_column}'"
                    )

        for fk in self.foreign_keys:
            for side, table_name, column_name in (
                ("from", fk.from_table, fk.from_column),
                ("to", fk.to_table, fk.to_column),
            ):
                table = by_name.get(table_name)
                if table is None:
                    raise SpecError(
                        f"foreign key {side} side references unknown table '{table_name}'"
                    )
                if column_name not in {c.name for c in table.columns}:
                    raise SpecError(
                        f"foreign key {side} side references unknown column "
                        f"'{table_name}.{column_name}'"
                    )

            if not 0.0 <= fk.containment <= 1.0:
                raise SpecError(
                    f"containment {fk.containment} for "
                    f"{fk.from_table}.{fk.from_column} is outside [0, 1]"
                )

            # The invariant D5 exists to protect. Postgres will reject the
            # constraint at CREATE time anyway, but failing here means the
            # spec is honest before it ever reaches a database.
            if fk.enforced and fk.containment < 1.0:
                raise SpecError(
                    f"{fk.from_table}.{fk.from_column} is marked enforced but only "
                    f"{fk.containment:.1%} of its values exist in "
                    f"{fk.to_table}.{fk.to_column}"
                )
