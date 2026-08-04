"""Polars frame helpers for statistics backends."""

from __future__ import annotations

from typing import Any

import polars as pl


def eager(data: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    if isinstance(data, pl.LazyFrame):
        return data.collect()
    return data


def numeric_series(frame: pl.DataFrame, column: str) -> pl.Series:
    if column not in frame.columns:
        raise KeyError(f"Column '{column}' not in frame")
    return frame[column].drop_nulls().cast(pl.Float64, strict=False)


def require_min_samples(series: pl.Series, minimum: int) -> pl.Series:
    if series.len() < minimum:
        raise ValueError(f"Need at least {minimum} samples, got {series.len()}")
    return series
