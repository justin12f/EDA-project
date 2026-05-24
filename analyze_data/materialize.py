"""Explicit sklearn/scipy materialization boundary (Group C analyzers)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from statistics.core.frame_extract import column_to_numpy


def to_pandas(data: Any) -> pd.DataFrame:
    """Materialize backend frame to pandas for sklearn/scipy-only code paths."""
    if isinstance(data, pd.DataFrame):
        return data
    module = type(data).__module__
    if "polars" in module:
        import polars as pl

        frame = data.collect() if isinstance(data, pl.LazyFrame) else data
        return frame.to_pandas()
    if "pyspark" in module:
        return data.toPandas()
    raise TypeError(f"Cannot materialize type {type(data)!r} to pandas")


def column_as_numpy(data: Any, column: str):
    """Single-column materialization for statistics adapters."""
    return column_to_numpy(data, column)
