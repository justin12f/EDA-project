"""Abstract contract for Seasonal."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class AbstractSeasonalCalculator(ABC):
    """Contract for seasonal decomposition (Trend, Seasonality, Residual).

    Parameters
    ----------
    data:
        Backend-native dataframe (assumed ordered by time).
    value_column:
        Column containing the time series values.
    period:
        Seasonal period (e.g. 12 for monthly data with annual seasonality).

    Returns
    -------
    dict[str, Any]
        Dictionary with decomposition components.
    """

    @abstractmethod
    def calculate(
        self,
        data: Any,
        value_column: str,
        period: int = 12,
    ) -> dict[str, Any]:
        """Decompose time series into seasonal components."""
