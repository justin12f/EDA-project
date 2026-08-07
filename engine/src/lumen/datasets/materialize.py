"""Move dataframes between processes as Parquet.

The API process that reads a source and the worker process that cleans it are
different processes, so a dataframe cannot be handed over in memory — the
module-level dict this replaces was not merely untidy, it was incorrect.

Parquet because it is the one format all three backends read and write natively,
preserves dtypes, and is cheap to re-open lazily: polars gets a `LazyFrame` back,
so a downstream filter is pushed into the scan rather than materialising 48M rows.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from lumen.core.backend import Backend, validate_backend


@dataclass(frozen=True)
class DatasetMeta:
    schema: dict[str, str]
    row_count: int
    byte_size: int


def frame_schema(frame: Any, backend: Backend | str) -> dict[str, str]:
    """Column name → dtype, as strings an agent can reason about."""
    backend = validate_backend(str(backend))

    if backend == "pandas":
        return {str(column): str(dtype) for column, dtype in frame.dtypes.items()}

    if backend == "polars":
        # collect_schema() avoids materialising a LazyFrame just to read its types.
        schema = frame.collect_schema() if hasattr(frame, "collect_schema") else frame.schema
        return {str(name): str(dtype) for name, dtype in schema.items()}

    return {field.name: field.dataType.simpleString() for field in frame.schema.fields}


def write_parquet(frame: Any, backend: Backend | str, path: str) -> DatasetMeta:
    backend = validate_backend(str(backend))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    if backend == "pandas":
        frame.to_parquet(path, index=False)
        row_count = len(frame)
        schema = frame_schema(frame, backend)

    elif backend == "polars":
        materialised = frame.collect() if hasattr(frame, "collect") else frame
        materialised.write_parquet(path)
        row_count = materialised.height
        schema = frame_schema(materialised, backend)

    else:  # spark writes a directory of part files
        frame.write.mode("overwrite").parquet(path)
        row_count = frame.count()
        schema = frame_schema(frame, backend)

    return DatasetMeta(schema=schema, row_count=row_count, byte_size=_tree_size(path))


def read_parquet(path: str, backend: Backend | str) -> Any:
    backend = validate_backend(str(backend))

    if backend == "pandas":
        import pandas as pd

        return pd.read_parquet(path)

    if backend == "polars":
        import polars as pl

        # Lazy on purpose: the caller may only want a sample or a few columns.
        return pl.scan_parquet(path)

    from pyspark.sql import SparkSession

    return SparkSession.builder.getOrCreate().read.parquet(path)


def row_count(frame: Any, backend: Backend | str) -> int:
    """How many rows, without writing anything — a shadow run (ADR-0008) reads
    this before and after a candidate pipeline to size its effect, and must
    not materialise a handle just to answer "how many rows now"."""
    backend = validate_backend(str(backend))

    if backend == "pandas":
        return len(frame)

    if backend == "polars":
        materialised = frame.collect() if hasattr(frame, "collect") else frame
        return materialised.height

    return frame.count()


def null_rates(frame: Any, backend: Backend | str) -> dict[str, float]:
    """Fraction of nulls per column, in one pass.

    This is what the cleaning agent's proposal is derived from, so it is computed
    from the data rather than sampled — a proposal that drops rows on the basis
    of an estimate is not something a human should be asked to approve.
    """
    backend = validate_backend(str(backend))

    if backend == "pandas":
        total = len(frame)
        if total == 0:
            return {str(c): 0.0 for c in frame.columns}
        return {str(c): float(frame[c].isna().sum()) / total for c in frame.columns}

    if backend == "polars":
        import polars as pl

        lazy = frame.lazy() if hasattr(frame, "lazy") else frame
        columns = list(frame_schema(lazy, "polars"))
        if not columns:
            return {}
        counts = lazy.select(
            [pl.len().alias("__rows")]
            + [pl.col(c).null_count().alias(c) for c in columns]
        ).collect()
        total = int(counts["__rows"][0])
        if total == 0:
            return {c: 0.0 for c in columns}
        return {c: float(counts[c][0]) / total for c in columns}

    from pyspark.sql import functions as F

    total = frame.count()
    if total == 0:
        return {c: 0.0 for c in frame.columns}
    row = frame.select(
        [F.sum(F.col(c).isNull().cast("int")).alias(c) for c in frame.columns]
    ).collect()[0]
    return {c: float(row[c] or 0) / total for c in frame.columns}


def column_values(frame: Any, backend: Backend | str, column: str) -> set[str]:
    """Distinct non-null values of one column, as strings — the raw material
    for a cross-source match-rate comparison (ADR-0010's impact simulation).

    Read wholesale rather than sampled: this product's dataset sizes make an
    exact set affordable today, and a match rate computed from a sample
    would swing on which rows happened to be drawn — a worse kind of wrong
    than "slow." Revisit with a sampling strategy once a source is large
    enough that this is measurably the bottleneck, not before.
    """
    backend = validate_backend(str(backend))

    if backend == "pandas":
        return {str(v) for v in frame[column].dropna().unique()}

    if backend == "polars":
        materialised = frame.collect() if hasattr(frame, "collect") else frame
        return {str(v) for v in materialised[column].drop_nulls().unique().to_list()}

    rows = frame.select(column).dropna().distinct().collect()
    return {str(row[column]) for row in rows}


def duplicate_counts(frame: Any, backend: Backend | str, columns: list[str]) -> dict[str, int]:
    """Rows beyond the first for each repeated value, per column.

    Reported per column rather than per row-combination because that is the shape
    a `drop_duplicates` step takes, and because a duplicate `email_hash` is a
    different problem from a duplicate row.
    """
    backend = validate_backend(str(backend))
    out: dict[str, int] = {}

    if backend == "pandas":
        for column in columns:
            series = frame[column].dropna()
            out[column] = int(len(series) - series.nunique())
        return out

    if backend == "polars":
        import polars as pl

        lazy = frame.lazy() if hasattr(frame, "lazy") else frame
        stats = lazy.select(
            [
                expression
                for column in columns
                for expression in (
                    pl.col(column).drop_nulls().len().alias(f"{column}__n"),
                    pl.col(column).drop_nulls().n_unique().alias(f"{column}__u"),
                )
            ]
        ).collect()
        for column in columns:
            out[column] = int(stats[f"{column}__n"][0]) - int(stats[f"{column}__u"][0])
        return out

    for column in columns:
        non_null = frame.filter(frame[column].isNotNull())
        out[column] = int(non_null.count() - non_null.select(column).distinct().count())
    return out


def numeric_summary(frame: Any, backend: Backend | str, columns: list[str]) -> dict[str, dict[str, float]]:
    """Mean, stdev, and non-null count per numeric column, in one pass — the
    single-scan snapshot a `NumericBaseline` (ADR-0013) is updated with one
    number (the mean) at a time. Stdev and count travel alongside for
    display; the cross-scan baseline itself only ever consumes the mean.
    """
    backend = validate_backend(str(backend))
    if not columns:
        return {}

    if backend == "pandas":
        out: dict[str, dict[str, float]] = {}
        for column in columns:
            series = frame[column].dropna()
            n = len(series)
            out[column] = {
                "mean": float(series.mean()) if n else 0.0,
                "stdev": float(series.std(ddof=1)) if n > 1 else 0.0,
                "count": n,
            }
        return out

    if backend == "polars":
        import polars as pl

        lazy = frame.lazy() if hasattr(frame, "lazy") else frame
        stats = lazy.select(
            [
                expression
                for column in columns
                for expression in (
                    pl.col(column).mean().alias(f"{column}__mean"),
                    pl.col(column).std().alias(f"{column}__std"),
                    pl.col(column).drop_nulls().len().alias(f"{column}__n"),
                )
            ]
        ).collect()
        return {
            column: {
                "mean": float(stats[f"{column}__mean"][0] or 0.0),
                "stdev": float(stats[f"{column}__std"][0] or 0.0),
                "count": int(stats[f"{column}__n"][0]),
            }
            for column in columns
        }

    from pyspark.sql import functions as F

    row = frame.select(
        [
            expression
            for column in columns
            for expression in (
                F.avg(F.col(column)).alias(f"{column}__mean"),
                F.stddev_samp(F.col(column)).alias(f"{column}__std"),
                F.count(F.col(column)).alias(f"{column}__n"),
            )
        ]
    ).collect()[0]
    return {
        column: {
            "mean": float(row[f"{column}__mean"] or 0.0),
            "stdev": float(row[f"{column}__std"] or 0.0),
            "count": int(row[f"{column}__n"] or 0),
        }
        for column in columns
    }


def value_counts(frame: Any, backend: Backend | str, columns: list[str]) -> dict[str, dict[str, int]]:
    """Frequency of each observed value, per categorical column — the raw
    material a `CategoricalBaseline` (ADR-0013) tracks a rolling share of.
    Read wholesale, the same "this product's dataset sizes make an exact
    count affordable today" reasoning `column_values` already documents.
    """
    backend = validate_backend(str(backend))
    out: dict[str, dict[str, int]] = {}

    if backend == "pandas":
        for column in columns:
            out[column] = {str(k): int(v) for k, v in frame[column].dropna().value_counts().items()}
        return out

    if backend == "polars":
        materialised = frame.collect() if hasattr(frame, "collect") else frame
        for column in columns:
            counted = materialised[column].drop_nulls().value_counts()
            out[column] = {str(row[0]): int(row[1]) for row in counted.iter_rows()}
        return out

    for column in columns:
        rows = frame.groupBy(column).count().collect()
        out[column] = {str(row[column]): int(row["count"]) for row in rows if row[column] is not None}
    return out


def string_length_summary(frame: Any, backend: Backend | str, columns: list[str]) -> dict[str, dict[str, float]]:
    """Mean/stdev/count of string length per column — half of a
    `freeform_string` baseline (ADR-0013 §1). The other half (a
    character-class signature) is computed in
    `services/api/src/lumen_api/baselines.py` from a sample of the actual
    values, not here, since it needs per-value inspection this per-column
    aggregate does not.
    """
    backend = validate_backend(str(backend))
    if not columns:
        return {}

    if backend == "pandas":
        out: dict[str, dict[str, float]] = {}
        for column in columns:
            lengths = frame[column].dropna().astype(str).str.len()
            n = len(lengths)
            out[column] = {
                "mean": float(lengths.mean()) if n else 0.0,
                "stdev": float(lengths.std(ddof=1)) if n > 1 else 0.0,
                "count": n,
            }
        return out

    if backend == "polars":
        import polars as pl

        lazy = frame.lazy() if hasattr(frame, "lazy") else frame
        stats = lazy.select(
            [
                expression
                for column in columns
                for expression in (
                    pl.col(column).str.len_chars().mean().alias(f"{column}__mean"),
                    pl.col(column).str.len_chars().std().alias(f"{column}__std"),
                    pl.col(column).drop_nulls().len().alias(f"{column}__n"),
                )
            ]
        ).collect()
        return {
            column: {
                "mean": float(stats[f"{column}__mean"][0] or 0.0),
                "stdev": float(stats[f"{column}__std"][0] or 0.0),
                "count": int(stats[f"{column}__n"][0]),
            }
            for column in columns
        }

    from pyspark.sql import functions as F

    row = frame.select(
        [
            expression
            for column in columns
            for expression in (
                F.avg(F.length(F.col(column))).alias(f"{column}__mean"),
                F.stddev_samp(F.length(F.col(column))).alias(f"{column}__std"),
                F.count(F.col(column)).alias(f"{column}__n"),
            )
        ]
    ).collect()[0]
    return {
        column: {
            "mean": float(row[f"{column}__mean"] or 0.0),
            "stdev": float(row[f"{column}__std"] or 0.0),
            "count": int(row[f"{column}__n"] or 0),
        }
        for column in columns
    }


def latest_datetime(frame: Any, backend: Backend | str, columns: list[str]) -> dict[str, str | None]:
    """The most recent value in each datetime column, as an ISO-ish string
    — a `datetime` baseline (ADR-0013 §1) tracks how the *lag* between this
    and "now" moves scan to scan, computed by the caller; this only reads
    the raw max."""
    backend = validate_backend(str(backend))
    if not columns:
        return {}

    if backend == "pandas":
        import pandas as pd

        out: dict[str, str | None] = {}
        for column in columns:
            value = frame[column].max()
            out[column] = None if pd.isna(value) else str(value)
        return out

    if backend == "polars":
        import polars as pl

        lazy = frame.lazy() if hasattr(frame, "lazy") else frame
        stats = lazy.select([pl.col(c).max().alias(c) for c in columns]).collect()
        return {column: (str(stats[column][0]) if stats[column][0] is not None else None) for column in columns}

    from pyspark.sql import functions as F

    row = frame.select([F.max(F.col(c)).alias(c) for c in columns]).collect()[0]
    return {column: (str(row[column]) if row[column] is not None else None) for column in columns}


def _tree_size(path: str) -> int:
    if os.path.isfile(path):
        return os.path.getsize(path)
    total = 0
    for root, _, files in os.walk(path):
        total += sum(os.path.getsize(os.path.join(root, name)) for name in files)
    return total
