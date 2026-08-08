"""Deterministic inference — no model call reaches any of this."""

from __future__ import annotations

import pytest

from lumen.architect.infer import infer_sql_type
from lumen.architect.spec import SqlType


@pytest.mark.parametrize(
    "dtype,expected",
    [
        # polars
        ("Int8", SqlType.INTEGER),
        ("Int32", SqlType.INTEGER),
        ("Int64", SqlType.BIGINT),
        ("UInt64", SqlType.BIGINT),
        ("Float32", SqlType.DOUBLE),
        ("Float64", SqlType.DOUBLE),
        ("Boolean", SqlType.BOOLEAN),
        ("Utf8", SqlType.TEXT),
        ("String", SqlType.TEXT),
        ("Date", SqlType.DATE),
        ("Decimal(12, 2)", SqlType.NUMERIC),
        # pandas
        ("int64", SqlType.BIGINT),
        ("int32", SqlType.INTEGER),
        ("float64", SqlType.DOUBLE),
        ("bool", SqlType.BOOLEAN),
        ("object", SqlType.TEXT),
        ("category", SqlType.TEXT),
    ],
)
def test_scalar_dtypes(dtype, expected):
    assert infer_sql_type(dtype)[0] is expected


@pytest.mark.parametrize(
    "dtype",
    [
        "Datetime(time_unit='us', time_zone=None)",
        "datetime64[ns]",
    ],
)
def test_naive_datetimes_map_to_timestamp(dtype):
    assert infer_sql_type(dtype)[0] is SqlType.TIMESTAMP


@pytest.mark.parametrize(
    "dtype",
    [
        "Datetime(time_unit='us', time_zone='UTC')",
        "datetime64[ns, UTC]",
    ],
)
def test_aware_datetimes_map_to_timestamptz(dtype):
    """Dropping the zone would silently reinterpret every timestamp, which is
    the kind of corruption nobody notices until a report is wrong."""
    assert infer_sql_type(dtype)[0] is SqlType.TIMESTAMPTZ


@pytest.mark.parametrize("dtype", ["List(Int64)", "Struct({'a': Int64})"])
def test_nested_types_map_to_jsonb(dtype):
    assert infer_sql_type(dtype)[0] is SqlType.JSONB


def test_decimal_carries_its_precision_and_scale():
    assert infer_sql_type("Decimal(12, 2)") == (SqlType.NUMERIC, "12,2")


def test_a_type_without_an_argument_returns_none():
    assert infer_sql_type("Int64") == (SqlType.BIGINT, None)


def test_an_unknown_dtype_falls_back_to_text():
    """Falling back is right: a column we cannot type is still a column the
    customer wants to see, and text loses nothing that was in the file."""
    assert infer_sql_type("SomeFutureType") == (SqlType.TEXT, None)


# ── primary key selection ───────────────────────────────────────────────

import polars as pl  # noqa: E402

from lumen.architect.infer import select_primary_key  # noqa: E402


def test_a_column_named_id_wins_over_other_candidates():
    frame = pl.DataFrame({"code": ["a", "b"], "id": [1, 2]})
    key, rationale = select_primary_key(frame, "polars", ["code", "id"])
    assert key == ("id",)
    assert "id" in rationale


def test_an_id_suffixed_column_wins_when_there_is_no_bare_id():
    frame = pl.DataFrame({"name": ["a", "b"], "order_id": [1, 2]})
    key, _ = select_primary_key(frame, "polars", ["name", "order_id"])
    assert key == ("order_id",)


def test_the_leftmost_unique_column_wins_when_no_name_hints():
    frame = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    key, _ = select_primary_key(frame, "polars", ["a", "b"])
    assert key == ("a",)


def test_a_column_with_duplicates_is_not_a_candidate():
    frame = pl.DataFrame({"id": [1, 1], "code": ["x", "y"]})
    key, _ = select_primary_key(frame, "polars", ["id", "code"])
    assert key == ("code",)


def test_a_column_with_nulls_is_not_a_candidate():
    frame = pl.DataFrame({"id": [1, None], "code": ["x", "y"]})
    key, _ = select_primary_key(frame, "polars", ["id", "code"])
    assert key == ("code",)


def test_no_viable_key_returns_none_with_an_explanation():
    """The rationale is shown to a human deciding whether to accept the
    schema, so it has to read as a reason, not a stack trace."""
    frame = pl.DataFrame({"a": [1, 1], "b": [2, 2]})
    key, rationale = select_primary_key(frame, "polars", ["a", "b"])
    assert key is None
    assert "no column" in rationale.lower()


def test_an_empty_frame_returns_none():
    frame = pl.DataFrame({"a": []})
    key, _ = select_primary_key(frame, "polars", ["a"])
    assert key is None
