"""Abstract contract for Momentum."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class AbstractMomentumCalculator(ABC):
    """Contract for calculating momentum indicators (e.g. RSI, MACD).

    Parameters
    ----------
    data:
        Backend-native dataframe (assumed ordered by time).
    value_column:
        Column containing the time series values.

    Returns
    -------
    dict[str, Any]
        Dictionary with momentum indicators.
    """

    @abstractmethod
    def calculate(
        self,
        data: Any,
        value_column: str,
    ) -> dict[str, Any]:
        """Calculate momentum indicators."""
