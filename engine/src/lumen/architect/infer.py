"""Deterministic schema inference.

Nothing here calls a model. That is the same discipline `lumen.sentinel.
drift` states for detection and `lumen.sentinel.baseline` for calibration:
the expensive, non-reproducible tier is reserved for judgment, and a type
mapping is not judgment. The model's only job downstream is to make the
names readable and write the rationale prose.
"""

from __future__ import annotations

import re

from lumen.architect.spec import SqlType

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
