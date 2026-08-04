"""Abstract contract for central tendency calculations.

This module defines the single abstract class that every backend
(Polars, Spark, Pandas) must implement to compute central tendency
measures on a column of a dataframe.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class AbstractCentralTendencyCalculator(ABC):
    """Contract for computing central tendency measures on a dataframe column.

    Functionality
    -------------
    Computes the following measures for a specified numeric column:

    * **Mean** — arithmetic average of all non-null values.
    * **Median** — 50th percentile; robust to outliers.
    * **Mode** — most frequent value and its occurrence count.
    * **Trimmed mean** — mean after removing the extreme proportions on
      both ends of the distribution (reduces the influence of outliers).
    * **Distribution shape hint** — qualitative label derived from the
      mean-median relationship (``symmetric``, ``right_skewed``,
      ``left_skewed``, ``mean_is_zero``).

    Implementations must use the native expression API of their backend
    (Polars lazy expressions, PySpark SQL functions, Pandas vectorised
    operations) and must not convert data to NumPy or an alien backend
    inside this method.

    Parameters
    ----------
    data:
        Backend-native dataframe (``pl.DataFrame | pl.LazyFrame`` for
        Polars; ``pyspark.sql.DataFrame`` for Spark;
        ``pandas.DataFrame`` for Pandas).
    column:
        Name of the numeric column to analyse.
    trim_proportion:
        Fraction of observations to remove from each tail when computing
        the trimmed mean.  Must be in ``[0.0, 0.5)``.

    Returns
    -------
    dict[str, Any]
        Keys: ``mean``, ``median``, ``mode`` (dict with ``value`` and
        ``count``), ``trimmed_mean``, ``trim_proportion``,
        ``distribution_shape_hint``.

    Raises
    ------
    KeyError
        If ``column`` is not present in ``data``.
    ValueError
        If the column is empty after dropping nulls, or if
        ``trim_proportion`` is out of range.
    """

    @abstractmethod
    def calculate(
        self,
        data: Any,
        column: str,
        trim_proportion: float = 0.1,
    ) -> dict[str, Any]:
        """Calculate all central tendency measures for ``column`` in ``data``."""
