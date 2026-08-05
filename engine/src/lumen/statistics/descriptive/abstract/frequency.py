"""Abstract contract for frequency distribution construction.

Defines the interface that every backend must implement to build a
complete frequency distribution table (histogram) from a numeric column.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class AbstractFrequencyDistributionBuilder(ABC):
    """Contract for building a frequency distribution from a numeric column.

    Functionality
    -------------
    Constructs a histogram-based frequency table for a numeric column:

    * **Bin count selection** — automatically selects the number of bins
      using classical rules (Sturges, Scott, Freedman-Diaconis, or
      ``"auto"`` which chooses based on sample size), unless the caller
      supplies ``n_bins`` explicitly.
    * **Frequency table** — returns per-bin statistics: bin boundaries,
      human-readable bin label, absolute frequency, relative frequency
      (proportion), and cumulative relative frequency.

    Implementations must use the native expression API of their backend and
    must not convert data to NumPy or an alien backend inside this method.

    Parameters
    ----------
    data:
        Backend-native dataframe.
    column:
        Name of the numeric column to analyse.
    n_bins:
        Fixed number of bins.  When ``None``, the bin count is selected
        automatically using ``bin_method``.
    bin_method:
        Rule for automatic bin count selection.  One of ``"sturges"``,
        ``"scott"``, ``"fd"`` (Freedman-Diaconis), or ``"auto"``.
        Ignored when ``n_bins`` is provided.

    Returns
    -------
    dict[str, Any]
        Keys:
        * ``table`` — list of dicts, each with ``bin_start``, ``bin_end``,
          ``bin_label``, ``frequency``, ``relative_frequency``,
          ``cumulative_frequency``.
        * ``n_bins`` — int (actual number of bins used).
        * ``total_count`` — int (number of non-null observations).
        * ``bin_method`` — str (``"manual"`` if ``n_bins`` was supplied).

    Raises
    ------
    KeyError
        If ``column`` is not present in ``data``.
    ValueError
        If the column is empty after dropping nulls.
    """

    @abstractmethod
    def build(
        self,
        data: Any,
        column: str,
        n_bins: int | None = None,
        bin_method: str = "auto",
    ) -> dict[str, Any]:
        """Build the frequency distribution for ``column`` in ``data``."""
