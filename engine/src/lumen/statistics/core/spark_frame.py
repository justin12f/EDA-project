"""Spark frame helpers for statistics backends."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyspark.sql import DataFrame as SparkDataFrame


def numeric_column(df: "SparkDataFrame", column: str) -> "SparkDataFrame":
    # PySpark is an optional dependency (the ``spark`` extra) — imported
    # lazily here so importing this module never requires it.
    try:
        from pyspark.sql import functions as F
    except ImportError as exc:
        raise ImportError(
            "Backend 'spark' requires PySpark. "
            "Install it with: uv sync --extra spark"
        ) from exc

    if column not in df.columns:
        raise KeyError(f"Column '{column}' not in frame")
    return df.select(F.col(column).cast("double").alias(column)).na.drop()
