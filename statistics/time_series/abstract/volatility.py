"""Abstract contract for Volatility."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class AbstractVolatilityCalculator(ABC):
    """Contract for calculating volatility (e.g. standard deviation of returns).

    Parameters
    ----------
    data:
        Backend-native dataframe (assumed ordered by time).
    value_column:
        Column containing the time series values (returns).
    window:
        Window size for rolling volatility.

    Returns
    -------
    dict[str, Any]
        Dictionary with volatility metrics.
    """

    @abstractmethod
    def calculate(
        self,
        data: Any,
        value_column: str,
        window: int = 30,
    ) -> dict[str, Any]:
        """Calculate volatility."""
