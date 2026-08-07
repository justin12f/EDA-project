"""Dataset materialisation and the profiling primitives a proposal is built from."""

from __future__ import annotations

import datetime as dt

import pandas as pd
import polars as pl
import pytest

from lumen.datasets.materialize import (
    duplicate_counts,
    frame_schema,
    latest_datetime,
    null_rates,
    numeric_summary,
    read_parquet,
    string_length_summary,
    value_counts,
    write_parquet,
)

DATA = {"id": [1, 2, 3], "name": ["a", "b", "c"], "score": [1.5, 2.5, 3.5]}

# Two nulls out of five in country_code (40%), and 'a1' repeated in email_hash —
# the same shape as the end-to-end fixture, so the numbers here and the numbers
# the agent proposes on are derived the same way.
DIRTY = {
    "id": [1, 2, 3, 4, 5],
    "country_code": ["DE", None, "US", "FR", None],
    "email_hash": ["a1", "b2", "a1", "c3", "d4"],
}


# ── round trips ─────────────────────────────────────────────────────────────


def test_pandas_round_trip(tmp_path):
    path = str(tmp_path / "p.parquet")
    meta = write_parquet(pd.DataFrame(DATA), "pandas", path)

    assert meta.row_count == 3
    assert set(meta.schema) == {"id", "name", "score"}
    assert meta.byte_size > 0

    restored = read_parquet(path, "pandas")
    assert list(restored.columns) == ["id", "name", "score"]
    assert len(restored) == 3


def test_polars_round_trip_stays_lazy(tmp_path):
    path = str(tmp_path / "q.parquet")
    meta = write_parquet(pl.DataFrame(DATA), "polars", path)
    assert meta.row_count == 3

    restored = read_parquet(path, "polars")
    assert isinstance(restored, pl.LazyFrame), "a downstream filter must push into the scan"
    assert restored.collect().height == 3


def test_a_polars_lazyframe_is_collected_before_writing(tmp_path):
    meta = write_parquet(pl.LazyFrame(DATA), "polars", str(tmp_path / "r.parquet"))
    assert meta.row_count == 3


def test_a_round_trip_preserves_values_across_backends(tmp_path):
    pandas_path = str(tmp_path / "a.parquet")
    polars_path = str(tmp_path / "b.parquet")
    write_parquet(pd.DataFrame(DATA), "pandas", pandas_path)
    write_parquet(pl.DataFrame(DATA), "polars", polars_path)

    from_pandas = read_parquet(pandas_path, "pandas")["score"].tolist()
    from_polars = read_parquet(polars_path, "polars").collect()["score"].to_list()
    assert from_pandas == from_polars == [1.5, 2.5, 3.5]


def test_unknown_backend_raises(tmp_path):
    with pytest.raises(ValueError, match="Unsupported backend"):
        write_parquet(pd.DataFrame(DATA), "duckdb", str(tmp_path / "x.parquet"))


# ── schema ──────────────────────────────────────────────────────────────────


def test_schema_reports_dtypes_as_strings():
    schema = frame_schema(pd.DataFrame(DATA), "pandas")
    assert schema["id"].startswith("int")
    assert schema["score"].startswith("float")
    assert all(isinstance(value, str) for value in schema.values())


def test_schema_of_a_lazyframe_does_not_require_collecting():
    schema = frame_schema(pl.LazyFrame(DATA), "polars")
    assert set(schema) == {"id", "name", "score"}


# ── profiling ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_null_rates_are_computed_not_sampled(backend):
    frame = pd.DataFrame(DIRTY) if backend == "pandas" else pl.DataFrame(DIRTY)
    rates = null_rates(frame, backend)

    assert rates["country_code"] == pytest.approx(0.4)
    assert rates["id"] == 0.0
    assert rates["email_hash"] == 0.0


def test_both_backends_report_the_same_null_rates():
    from_pandas = null_rates(pd.DataFrame(DIRTY), "pandas")
    from_polars = null_rates(pl.DataFrame(DIRTY), "polars")
    assert from_pandas == pytest.approx(from_polars)


def test_null_rates_of_an_empty_frame_are_zero_not_a_division_error():
    empty = pl.DataFrame({"a": [], "b": []})
    assert null_rates(empty, "polars") == {"a": 0.0, "b": 0.0}


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_duplicate_counts_ignore_nulls(backend):
    frame = pd.DataFrame(DIRTY) if backend == "pandas" else pl.DataFrame(DIRTY)
    counts = duplicate_counts(frame, backend, ["email_hash", "country_code"])

    # 'a1' appears twice → one duplicate beyond the first.
    assert counts["email_hash"] == 1
    # Two nulls are not duplicates of each other; DE/US/FR are all distinct.
    assert counts["country_code"] == 0


def test_a_column_with_no_duplicates_reports_zero():
    counts = duplicate_counts(pl.DataFrame(DATA), "polars", ["id", "name"])
    assert counts == {"id": 0, "name": 0}


def test_null_rates_survive_a_parquet_round_trip(tmp_path):
    """The worker profiles what it read back, not what the API held in memory."""
    path = str(tmp_path / "dirty.parquet")
    write_parquet(pl.DataFrame(DIRTY), "polars", path)

    rates = null_rates(read_parquet(path, "polars"), "polars")
    assert rates["country_code"] == pytest.approx(0.4)


# ── baselines (ADR-0013) ────────────────────────────────────────────────────

NUMERIC_DATA = {"amount": [10.0, 20.0, 30.0, 40.0, 50.0]}
CATEGORY_DATA = {"country": ["US", "US", "US", "DE", "FR", None]}
STRING_DATA = {"note": ["hi", "hello", "hey there", None, "yo"]}
DATE_DATA = {"seen_at": [dt.date(2024, 1, 1), dt.date(2024, 1, 3), dt.date(2024, 1, 2)]}


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_numeric_summary_computes_mean_and_stdev(backend):
    frame = pd.DataFrame(NUMERIC_DATA) if backend == "pandas" else pl.DataFrame(NUMERIC_DATA)
    summary = numeric_summary(frame, backend, ["amount"])
    assert summary["amount"]["mean"] == pytest.approx(30.0)
    assert summary["amount"]["stdev"] == pytest.approx(15.811, abs=1e-2)
    assert summary["amount"]["count"] == 5


def test_numeric_summary_of_an_empty_column_list_is_empty():
    assert numeric_summary(pl.DataFrame(NUMERIC_DATA), "polars", []) == {}


def test_both_backends_agree_on_numeric_summary():
    from_pandas = numeric_summary(pd.DataFrame(NUMERIC_DATA), "pandas", ["amount"])
    from_polars = numeric_summary(pl.DataFrame(NUMERIC_DATA), "polars", ["amount"])
    assert from_pandas["amount"]["mean"] == pytest.approx(from_polars["amount"]["mean"])
    assert from_pandas["amount"]["stdev"] == pytest.approx(from_polars["amount"]["stdev"])


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_value_counts_ignores_nulls(backend):
    frame = pd.DataFrame(CATEGORY_DATA) if backend == "pandas" else pl.DataFrame(CATEGORY_DATA)
    counts = value_counts(frame, backend, ["country"])
    assert counts["country"] == {"US": 3, "DE": 1, "FR": 1}


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_string_length_summary_ignores_nulls(backend):
    frame = pd.DataFrame(STRING_DATA) if backend == "pandas" else pl.DataFrame(STRING_DATA)
    summary = string_length_summary(frame, backend, ["note"])
    # 'hi'(2) 'hello'(5) 'hey there'(9) 'yo'(2) -> mean 4.5, n=4 (the null skipped)
    assert summary["note"]["count"] == 4
    assert summary["note"]["mean"] == pytest.approx(4.5)


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_latest_datetime_reports_the_max_value(backend):
    frame = pd.DataFrame(DATE_DATA) if backend == "pandas" else pl.DataFrame(DATE_DATA)
    latest = latest_datetime(frame, backend, ["seen_at"])
    assert latest["seen_at"] is not None
    assert "2024-01-03" in latest["seen_at"]


def test_latest_datetime_of_an_all_null_column_is_none():
    frame = pl.DataFrame({"seen_at": [None, None]}, schema={"seen_at": pl.Date})
    assert latest_datetime(frame, "polars", ["seen_at"]) == {"seen_at": None}
