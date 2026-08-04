"""Extract 1-D numeric series from backend frames (explicit materialization boundary)."""

from __future__ import annotations

from typing import Any

import numpy as np


def column_to_numpy(data: Any, column: str) -> np.ndarray:
    """Materialize a numeric column to numpy (documented adapter boundary)."""
    module = type(data).__module__
    if "pandas" in module:
        return data[column].to_numpy()
    if "polars" in module:
        import polars as pl

        frame = data.collect() if isinstance(data, pl.LazyFrame) else data
        return frame[column].to_numpy()
    if "pyspark" in module:
        rows = data.select(column).dropna().collect()
        return np.array([r[0] for r in rows], dtype=float)
    raise TypeError(f"Unsupported frame type for column extraction: {type(data)!r}")
