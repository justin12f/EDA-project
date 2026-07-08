"""Abstract contract for percentile and outlier detection calculations.

Defines the interface every backend must implement to compute configurable
percentile sets and detect extreme values in a numeric column.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class AbstractPercentilesCalculator(ABC):
    """Contract for computing percentiles and detecting outliers in a numeric column.

    Functionality
    -------------
    Computes a configurable set of percentiles for a numeric column and
    optionally identifies outlier values:

    * **Percentile map** — for each requested percentile ``p`` in
      ``[0, 100]``, returns the corresponding data value under key
      ``"p{p}"``.  Default set: ``[1, 5, 10, 25, 50, 75, 90, 95, 99]``.
    * **Outlier detection** — when ``outlier_bounds`` is provided as a
      ``(lower_pct, upper_pct)`` tuple, counts the number of observations
      falling below the lower-bound percentile value and above the
      upper-bound percentile value, together with the percentage of
      observations identified as outliers.

    Implementations must use the native expression API of their backend and
    must not convert data to NumPy or an alien backend inside this method.

    Parameters
    ----------
    data:
        Backend-native dataframe.
    column:
        Name of the numeric column to analyse.
    percentiles:
        Percentile values in ``[0, 100]`` to compute.  Defaults to
        ``[1, 5, 10, 25, 50, 75, 90, 95, 99]``.
    outlier_bounds:
        ``(lower_pct, upper_pct)`` percentile pair for outlier detection.
        Pass ``None`` to skip outlier detection.

    Returns
    -------
    dict[str, Any]
        Keys:
        * ``percentiles`` — dict mapping ``"p{n}"`` → float value.
        * ``n`` — int (number of non-null observations).
        * ``outlier_detection`` — dict (present only when
          ``outlier_bounds`` is not ``None``) with keys
          ``n_below_lower_bound``, ``n_above_upper_bound``,
          ``outlier_count``, ``outlier_percentage``, ``bounds``.

    Raises
    ------
    KeyError
        If ``column`` is not present in ``data``.
    ValueError
        If the column is empty after dropping nulls, or if any percentile
        value is outside ``[0, 100]``.
    """

    @abstractmethod
    def calculate(
        self,
        data: Any,
        column: str,
        percentiles: list[int] | None = None,
        outlier_bounds: tuple[int, int] | None = (1, 99),
    ) -> dict[str, Any]:
        """Calculate percentiles and optional outlier detection for ``column``."""
