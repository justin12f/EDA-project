"""Dataset materialisation and the profiling primitives a proposal is built from."""

from __future__ import annotations

import pandas as pd
import polars as pl
import pytest

from lumen.datasets.materialize import (
    duplicate_counts,
    frame_schema,
    null_rates,
    read_parquet,
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
