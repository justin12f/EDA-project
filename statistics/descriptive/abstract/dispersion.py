"""Abstract contract for dispersion / spread measure calculations.

This module defines the abstract class that every backend must implement
to compute all classical measures of statistical dispersion on a numeric
column of a dataframe.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class AbstractDispersionCalculator(ABC):
    """Contract for computing statistical dispersion measures on a dataframe column.

    Functionality
    -------------
    Computes the following measures for a specified numeric column:

    * **Variance** — average squared deviation from the mean; the ``ddof``
      parameter controls sample (``ddof=1``) vs. population (``ddof=0``)
      variance.
    * **Standard deviation** — square root of variance; same ``ddof`` semantics.
    * **Range** — minimum value, maximum value, and their arithmetic spread.
    * **IQR** — interquartile range (Q3 − Q1), together with Q1 and Q3.
    * **MAD** — median absolute deviation; a robust, outlier-resistant spread
      measure.
    * **Coefficient of variation** — ``std / |mean|``; scale-independent
      spread comparison across columns with different units.

    Implementations must use the native expression API of their backend and
    must not convert data to NumPy or an alien backend inside this method.

    Parameters
    ----------
    data:
        Backend-native dataframe.
    column:
        Name of the numeric column to analyse.
    ddof:
        Delta degrees of freedom for variance and standard deviation.
        ``1`` → sample statistics; ``0`` → population statistics.

    Returns
    -------
    dict[str, Any]
        Keys: ``variance``, ``std``, ``range`` (dict with ``min``, ``max``,
        ``range``), ``iqr`` (dict with ``q1``, ``q3``, ``iqr``), ``mad``,
        ``coefficient_of_variation``, ``ddof``.

    Raises
    ------
    KeyError
        If ``column`` is not present in ``data``.
    ValueError
        If the column is empty after dropping nulls.
    """

    @abstractmethod
    def calculate(
        self,
        data: Any,
        column: str,
        ddof: int = 1,
    ) -> dict[str, Any]:
        """Calculate all dispersion measures for ``column`` in ``data``."""
