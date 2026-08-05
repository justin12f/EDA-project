"""Frame → arrays. This is where the backend difference actually lives.

Predictors take numpy; pandas, polars and Spark disagree about how to hand it
over. Rather than three copies of every algorithm, there is one copy of every
algorithm and three copies of this.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from lumen.core.backend import Backend, validate_backend


def _to_pandas(frame: Any, backend: Backend) -> Any:
    """Collect to pandas once, at the boundary.

    Spark gets a `limit` guard: a `toPandas()` on an unbounded frame pulls the
    whole dataset into the driver, which is how a well-meaning fit takes a
    cluster down.
    """
    if backend == "pandas":
        return frame
    if backend == "polars":
        collected = frame.collect() if hasattr(frame, "collect") else frame
        return collected.to_pandas()

    raise TypeError(
        "Spark frames must be sampled before fitting — call "
        "`frame.limit(n).toPandas()` and pass the result as pandas. Fitting on an "
        "unbounded Spark frame pulls the whole dataset into the driver."
    )


def to_supervised(
    frame: Any,
    backend: Backend | str,
    target: str,
    features: list[str] | None = None,
    *,
    order_by: str | None = None,
    dropna_features: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Split a frame into (X, y, feature_names).

    `order_by` sorts first, which is what makes a time-series fit meaningful:
    every forecaster here treats row order as time order, and a frame that
    arrived from a parquet scan has whatever order the scan produced.

    Non-numeric feature columns are dropped rather than silently encoded. One-hot
    or ordinal is a modelling decision with consequences a person should approve,
    and guessing it here would bury that choice inside a utility function.
    """
    backend = validate_backend(str(backend))
    pandas_frame = _to_pandas(frame, backend)

    if target not in pandas_frame.columns:
        raise ValueError(
            f"target '{target}' is not a column. Available: {list(pandas_frame.columns)}"
        )

    if order_by is not None:
        if order_by not in pandas_frame.columns:
            raise ValueError(f"order_by '{order_by}' is not a column")
        pandas_frame = pandas_frame.sort_values(order_by)

    if features is None:
        candidates = [c for c in pandas_frame.columns if c != target and c != order_by]
    else:
        missing = [c for c in features if c not in pandas_frame.columns]
        if missing:
            raise ValueError(f"features not in the frame: {missing}")
        candidates = list(features)

    import pandas as pd

    numeric = [c for c in candidates if pd.api.types.is_numeric_dtype(pandas_frame[c])]
    dropped = [c for c in candidates if c not in numeric]
    if features is not None and dropped:
        raise ValueError(
            f"these requested features are not numeric: {dropped}. Encode them in a "
            "cleaning step first — how to encode a category is a modelling decision, "
            "not something this should guess."
        )
    if not numeric:
        raise ValueError(
            "no numeric feature columns available. Encode categoricals in a cleaning "
            "step, or pass `features` explicitly."
        )

    y = pandas_frame[target].to_numpy(dtype=float, na_value=np.nan)
    X = pandas_frame[numeric].to_numpy(dtype=float, na_value=np.nan)

    if dropna_features and X.size:
        keep = ~np.isnan(X).any(axis=1)
        X, y = X[keep], y[keep]

    return X, y, numeric


def to_series(
    frame: Any,
    backend: Backend | str,
    column: str,
    *,
    order_by: str | None = None,
) -> np.ndarray:
    """Pull one column as a 1-D float array, in time order when asked."""
    backend = validate_backend(str(backend))
    pandas_frame = _to_pandas(frame, backend)

    if column not in pandas_frame.columns:
        raise ValueError(
            f"column '{column}' is not in the frame. Available: {list(pandas_frame.columns)}"
        )
    if order_by is not None:
        if order_by not in pandas_frame.columns:
            raise ValueError(f"order_by '{order_by}' is not a column")
        pandas_frame = pandas_frame.sort_values(order_by)

    return pandas_frame[column].to_numpy(dtype=float, na_value=np.nan)
