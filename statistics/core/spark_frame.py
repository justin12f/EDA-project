"""Spark frame helpers for statistics backends."""

from __future__ import annotations

from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F


def numeric_column(df: SparkDataFrame, column: str) -> SparkDataFrame:
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not in frame")
    return df.select(F.col(column).cast("double").alias(column)).na.drop()
