"""Abstract contract for Moving Averages."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class AbstractMovingAveragesCalculator(ABC):
    """Contract for calculating simple and exponential moving averages.

    Parameters
    ----------
    data:
        Backend-native dataframe (assumed ordered by time).
    value_column:
        Column containing the time series values.
    windows:
        List of window sizes (e.g., [7, 14, 30]).

    Returns
    -------
    dict[str, Any]
        Dictionary with moving averages for each window.
    """

    @abstractmethod
    def calculate(
        self,
        data: Any,
        value_column: str,
        windows: list[int],
    ) -> dict[str, Any]:
        """Calculate moving averages."""
