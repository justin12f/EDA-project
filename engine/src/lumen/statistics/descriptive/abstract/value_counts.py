"""Abstract contract for value counts / frequency analysis.

Defines the interface every backend must implement to compute absolute
and relative frequency distributions for any column dtype.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class AbstractValueCountsCalculator(ABC):
    """Contract for computing value frequency distributions on any column.

    Functionality
    -------------
    Computes absolute and relative frequency counts for every distinct
    value in a column, regardless of dtype (numeric, string, categorical,
    boolean):

    * **Frequency table** — each distinct value with its absolute count,
      relative frequency (proportion of total), and percentage.
    * **Missing value accounting** — counts NaN/null occurrences separately
      and reports the missing percentage.  When ``include_missing=True``
      the null category appears in the frequency table.
    * **Top-N filtering** — optionally truncates the table to the ``top_n``
      most frequent values.
    * **Summary statistics** — ``n_total``, ``n_unique``, ``n_missing``,
      ``n_valid``.

    Implementations must use the native expression API of their backend and
    must not convert data to NumPy or an alien backend inside this method.

    Parameters
    ----------
    data:
        Backend-native dataframe.
    column:
        Name of the column to analyse.
    top_n:
        If set, returns only the top ``n`` most frequent values.
    include_missing:
        When ``True``, null values are counted as a separate category.

    Returns
    -------
    dict[str, Any]
        Keys: ``table`` (list of dicts with ``value``, ``frequency``,
        ``relative_frequency``, ``percentage``), ``n_total``,
        ``n_unique``, ``n_missing``, ``missing_percentage``,
        ``n_valid``, ``top_n``.

    Raises
    ------
    KeyError
        If ``column`` is not present in ``data``.
    """

    @abstractmethod
    def calculate(
        self,
        data: Any,
        column: str,
        top_n: int | None = None,
        include_missing: bool = True,
    ) -> dict[str, Any]:
        """Compute value frequency distribution for ``column`` in ``data``."""
