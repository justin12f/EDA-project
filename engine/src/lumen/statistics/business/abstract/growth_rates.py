"""Abstract contract for Growth Rates."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class AbstractGrowthRatesCalculator(ABC):
    """Contract for calculating growth rates (MoM, YoY, CAGR).

    Parameters
    ----------
    data:
        Backend-native dataframe containing time series data.
    date_column:
        Date or period column.
    value_column:
        Metric to calculate growth on.
    periods:
        Number of periods for sequential growth (e.g., 1 for MoM).

    Returns
    -------
    dict[str, Any]
        Dictionary with period-over-period growth and CAGR.
    """

    @abstractmethod
    def calculate(
        self,
        data: Any,
        date_column: str,
        value_column: str,
        periods: int = 1,
    ) -> dict[str, Any]:
        """Calculate growth rates over time."""
