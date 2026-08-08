"""Schema evolution — the "administers" half of the Architect.

D7 gates evolution on reversibility rather than on a list of allowed
operations, and the reason is ADR-0017 §3: irreversibility is a hard
ceiling that no amount of earned trust may raise. If dropping a column were
possible, a computer-authored migration could never be auto-applied at all,
because every schema change would carry an irreversible branch. Refusing to
drop is what keeps the whole mechanism eligible for autonomy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from lumen.architect.ddl import _quote, _render_type
from lumen.architect.spec import SchemaSpec, SqlType

# A change is reversible when the old values still fit the new type. These
# are the pairs where that holds without inspecting a single row.
_WIDENING: frozenset[tuple[SqlType, SqlType]] = frozenset(
    {
        (SqlType.INTEGER, SqlType.BIGINT),
        (SqlType.INTEGER, SqlType.NUMERIC),
        (SqlType.INTEGER, SqlType.DOUBLE),
        (SqlType.BIGINT, SqlType.NUMERIC),
        (SqlType.BIGINT, SqlType.DOUBLE),
        (SqlType.NUMERIC, SqlType.DOUBLE),
        (SqlType.VARCHAR, SqlType.TEXT),
        (SqlType.DATE, SqlType.TIMESTAMP),
        (SqlType.TIMESTAMP, SqlType.TIMESTAMPTZ),
    }
)

MigrationKind = Literal[
    "add_column", "widen_type", "deprecate_column", "narrow_type", "add_fk", "drop_fk"
]


@dataclass(frozen=True)
class MigrationStep:
    kind: MigrationKind
    table: str
    column: str | None
    detail: str
    reversible: bool


@dataclass(frozen=True)
class MigrationPlan:
    steps: tuple[MigrationStep, ...] = ()

    @property
    def reversible(self) -> bool:
        return all(step.reversible for step in self.steps)


def _is_widening(old: SqlType, new: SqlType) -> bool:
    # Anything becoming text is lossless — text holds every representation.
    return new is SqlType.TEXT or (old, new) in _WIDENING


def classify_migration(old: SchemaSpec, new: SchemaSpec) -> MigrationPlan:
    """Diff two specs into steps, each labelled reversible or not."""
    steps: list[MigrationStep] = []
    old_tables = {t.name: t for t in old.tables}

    for table in new.tables:
        previous = old_tables.get(table.name)
        if previous is None:
            continue  # a brand-new table is a create, not a migration

        old_columns = {c.name: c for c in previous.columns}
        new_columns = {c.name: c for c in table.columns}

        for name, column in new_columns.items():
            before = old_columns.get(name)
            if before is None:
                steps.append(
                    MigrationStep(
                        kind="add_column",
                        table=table.name,
                        column=name,
                        detail=f"new column '{name}' ({column.sql_type.value})",
                        reversible=True,
                    )
                )
            elif before.sql_type is not column.sql_type:
                widening = _is_widening(before.sql_type, column.sql_type)
                steps.append(
                    MigrationStep(
                        kind="widen_type" if widening else "narrow_type",
                        table=table.name,
                        column=name,
                        detail=(
                            f"'{name}' changes from {before.sql_type.value} "
                            f"to {column.sql_type.value}"
                        ),
                        reversible=widening,
                    )
                )

        for name in old_columns:
            if name not in new_columns:
                steps.append(
                    MigrationStep(
                        kind="deprecate_column",
                        table=table.name,
                        column=name,
                        detail=f"'{name}' is no longer present at origin",
                        reversible=True,
                    )
                )

    return MigrationPlan(steps=tuple(steps))


def render_migration(plan: MigrationPlan, schema: str) -> list[str]:
    """Statements for a plan.

    `deprecate_column` renders nothing. That is the point: the column stays,
    the flag lives in the spec, and no path through this function can emit
    DROP COLUMN.
    """
    statements: list[str] = []
    for step in plan.steps:
        if step.kind == "add_column":
            column_type = step.detail.rsplit("(", 1)[-1].rstrip(")")
            statements.append(
                f"ALTER TABLE {_quote(schema)}.{_quote(step.table)} "
                f"ADD COLUMN IF NOT EXISTS {_quote(step.column)} {column_type}"
            )
        elif step.kind in ("widen_type", "narrow_type"):
            new_type = step.detail.rsplit(" to ", 1)[-1]
            statements.append(
                f"ALTER TABLE {_quote(schema)}.{_quote(step.table)} "
                f"ALTER COLUMN {_quote(step.column)} TYPE {new_type}"
            )
    return statements
