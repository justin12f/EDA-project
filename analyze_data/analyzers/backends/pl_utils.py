"""Native Polars helpers for lightweight analyzers (no pandas)."""

from __future__ import annotations

from typing import Any

import polars as pl


def ensure_frame(data: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    return data.collect() if isinstance(data, pl.LazyFrame) else data


def dtypes_dict(frame: pl.DataFrame) -> dict[str, str]:
    return {name: str(dtype) for name, dtype in frame.schema.items()}


def describe_dict(frame: pl.DataFrame) -> dict[str, Any]:
    return frame.describe().to_dict(as_series=False)
