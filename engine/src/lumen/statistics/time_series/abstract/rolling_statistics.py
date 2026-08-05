"""Abstract contract for Rolling Statistics."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class AbstractRollingStatisticsCalculator(ABC):
    """Contract for calculating rolling statistics (mean, std, min, max).

    Parameters
    ----------
    data:
        Backend-native dataframe (assumed ordered by time).
    value_column:
        Column containing the time series values.
    window:
        Window size.

    Returns
    -------
    dict[str, Any]
        Dictionary with rolling statistics.
    """

    @abstractmethod
    def calculate(
        self,
        data: Any,
        value_column: str,
        window: int = 14,
    ) -> dict[str, Any]:
        """Calculate rolling statistics."""
