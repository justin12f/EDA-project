"""Abstract contract for Population Splits."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class AbstractPopulationSplitsCalculator(ABC):
    """Contract for calculating arbitrary population splits (quantiles, bins).

    Parameters
    ----------
    data:
        Backend-native dataframe.
    column:
        Numeric column to split.
    method:
        'quantiles' or 'equal_width'.
    n_bins:
        Number of groups to create.

    Returns
    -------
    dict[str, Any]
        Dictionary with bin edges and counts per group.
    """

    @abstractmethod
    def calculate(
        self,
        data: Any,
        column: str,
        method: str = "quantiles",
        n_bins: int = 4,
    ) -> dict[str, Any]:
        """Create population splits."""
