"""Deterministic schema inference.

Nothing here calls a model. That is the same discipline `lumen.sentinel.
drift` states for detection and `lumen.sentinel.baseline` for calibration:
the expensive, non-reproducible tier is reserved for judgment, and a type
mapping is not judgment. The model's only job downstream is to make the
names readable and write the rationale prose.
"""

from __future__ import annotations

import re
from typing import Any

from lumen.architect.spec import Evidence, ForeignKeySpec, SqlType, TableSpec
from lumen.datasets.materialize import column_values, duplicate_counts, null_rates, row_count

# Exact dtype strings, checked before the prefix rules below. Ordering
# matters: "int64" must not be matched by a naive "int" prefix rule that
# would call it INTEGER.
_EXACT: dict[str, SqlType] = {
    "int8": SqlType.INTEGER,
    "int16": SqlType.INTEGER,
    "int32": SqlType.INTEGER,
    "int64": SqlType.BIGINT,
    "uint8": SqlType.INTEGER,
    "uint16": SqlType.INTEGER,
    "uint32": SqlType.BIGINT,
    "uint64": SqlType.BIGINT,
    "float32": SqlType.DOUBLE,
    "float64": SqlType.DOUBLE,
    "boolean": SqlType.BOOLEAN,
    "bool": SqlType.BOOLEAN,
    "utf8": SqlType.TEXT,
    "string": SqlType.TEXT,
    "str": SqlType.TEXT,
    "object": SqlType.TEXT,
    "category": SqlType.TEXT,
    "date": SqlType.DATE,
    "time": SqlType.TEXT,
    "binary": SqlType.TEXT,
    "null": SqlType.TEXT,
}

_DECIMAL = re.compile(r"decimal\(\s*(\d+)\s*,\s*(\d+)\s*\)")


def infer_sql_type(dtype: str) -> tuple[SqlType, str | None]:
    """Map a polars or pandas dtype string to a Postgres type.

    Returns `(type, type_arg)` where `type_arg` carries the parenthesised
    part for the two types that take one — `numeric(p,s)` and `varchar(n)`.
    An unrecognised dtype falls back to TEXT rather than raising: a column
    we cannot type is still a column the customer wants, and text loses
    nothing the file contained.
    """
    normalised = dtype.strip().lower()

    decimal = _DECIMAL.search(normalised)
    if decimal:
        return SqlType.NUMERIC, f"{decimal.group(1)},{decimal.group(2)}"

    if normalised.startswith("datetime"):
        # Both dialects spell the zone differently — polars as
        # time_zone='UTC', pandas as datetime64[ns, UTC] — but the question
        # is the same: is there a zone at all. Dropping one would silently
        # reinterpret every value in the column.
        aware = "time_zone='" in normalised or re.search(r",\s*[a-z/_+\-0-9]+\]", normalised)
        return (SqlType.TIMESTAMPTZ if aware else SqlType.TIMESTAMP), None

    if normalised.startswith(("list(", "array(", "struct(")):
        return SqlType.JSONB, None

    if normalised.startswith("duration"):
        return SqlType.TEXT, None

    exact = _EXACT.get(normalised)
    if exact is not None:
        return exact, None

    if normalised.startswith("uuid"):
        return SqlType.UUID, None

    return SqlType.TEXT, None


def select_primary_key(
    frame: Any, backend: str, columns: list[str]
) -> tuple[tuple[str, ...] | None, str]:
    """Choose a single-column primary key, or explain why there isn't one.

    Single-column only in v1. A composite key is a modelling judgment — which
    combination is *the* identity, rather than merely unique together — and
    D2's conservative posture says a human makes that call, not an inference
    rule. The returned rationale is shown to that human, so it is written as
    an explanation rather than a trace.
    """
    if not columns:
        return None, "The table has no columns."

    if row_count(frame, backend) == 0:
        return None, "The table is empty, so no column can be shown to be unique."

    rates = null_rates(frame, backend)
    duplicates = duplicate_counts(frame, backend, list(columns))

    candidates = [
        column
        for column in columns
        if duplicates.get(column, 1) == 0 and rates.get(column, 1.0) == 0.0
    ]
    if not candidates:
        return None, (
            "No column is both unique and complete, so no primary key was declared. "
            "Every column either repeats a value or has gaps."
        )

    for preferred, why in (
        (lambda c: c == "id", "it is named 'id'"),
        (lambda c: c.endswith("_id"), "its name identifies it as a key"),
    ):
        for column in candidates:
            if preferred(column):
                return (column,), (
                    f"'{column}' is unique and complete across every row, and {why}."
                )

    column = candidates[0]
    return (column,), (
        f"'{column}' is unique and complete across every row, and is the first "
        f"such column in the table."
    )


# Below this share of child values present in the parent, a match is
# coincidence rather than a relationship. Chosen conservative on purpose:
# a missed key costs a line on a diagram, a false one costs a human's trust
# in every other line.
CONTAINMENT_FLOOR = 0.95


def _names_suggest_a_key(child_column: str, parent_table: str, parent_column: str) -> bool:
    singular = parent_table[:-1] if parent_table.endswith("s") else parent_table
    return child_column in {
        f"{singular}_{parent_column}",
        f"{parent_table}_{parent_column}",
        parent_column if child_column != parent_column else "",
    } or child_column == f"{singular}_id"


def detect_foreign_keys(
    frames: dict[str, Any],
    backend: str,
    tables: tuple[TableSpec, ...],
    semantic_pairs: list[tuple[str, str, str, str]] | None = None,
) -> list[ForeignKeySpec]:
    """Find relationships between tables by value containment.

    The classic algorithm: a child column references a parent when every one
    of its values exists in the parent's primary key. `column_values` and the
    set arithmetic are what `impact.py::_match_rate` already does for
    ADR-0010's cross-source match rate, applied to a different question.

    `semantic_pairs` carries ADR-0009's canonical entities in as
    `(child_table, child_column, parent_table, parent_column)` tuples. The
    engine never learns what a canonical entity is; it just weighs the hint.
    """
    semantic = set(semantic_pairs or ())
    parents = [
        (table.name, table.primary_key[0])
        for table in tables
        if table.primary_key and len(table.primary_key) == 1
    ]

    found: list[ForeignKeySpec] = []
    for child in tables:
        child_frame = frames.get(child.name)
        if child_frame is None:
            continue

        for parent_table, parent_column in parents:
            # A self-reference is a real pattern (org charts, threaded
            # comments) but it needs its own handling on replace and its own
            # UI treatment, so v1 declines rather than half-supporting it.
            if parent_table == child.name:
                continue
            parent_frame = frames.get(parent_table)
            if parent_frame is None:
                continue

            parent_values = column_values(parent_frame, backend, parent_column)
            if not parent_values:
                continue

            for column in child.columns:
                if column.name == parent_column and child.name == parent_table:
                    continue
                if child.primary_key and column.name in child.primary_key:
                    continue

                child_values = column_values(child_frame, backend, column.name)
                if not child_values:
                    continue

                containment = len(child_values & parent_values) / len(child_values)
                if containment < CONTAINMENT_FLOOR:
                    continue

                evidence = [Evidence.STRUCTURAL]
                if _names_suggest_a_key(column.name, parent_table, parent_column):
                    evidence.append(Evidence.NAMING)
                if (child.name, column.name, parent_table, parent_column) in semantic:
                    evidence.append(Evidence.SEMANTIC)

                found.append(
                    ForeignKeySpec(
                        from_table=child.name,
                        from_column=column.name,
                        to_table=parent_table,
                        to_column=parent_column,
                        containment=containment,
                        enforced=containment == 1.0,
                        evidence=tuple(evidence),
                        rationale=(
                            f"{containment:.1%} of values in "
                            f"{child.name}.{column.name} exist in "
                            f"{parent_table}.{parent_column}, which is that "
                            f"table's primary key."
                        ),
                    )
                )
    return found
