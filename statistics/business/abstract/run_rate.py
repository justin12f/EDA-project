"""Abstract contract for Run Rate calculation."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class AbstractRunRateCalculator(ABC):
    """Contract for extrapolating current performance to future periods (Run Rate).

    Parameters
    ----------
    data:
        Backend-native dataframe with periodic data.
    date_column:
        Date column.
    value_column:
        Value to extrapolate.
    extrapolation_periods:
        Number of periods to project forward (e.g., 12 for annualizing monthly data).

    Returns
    -------
    dict[str, Any]
        Dictionary with extrapolated run rate.
    """

    @abstractmethod
    def calculate(
        self,
        data: Any,
        date_column: str,
        value_column: str,
        extrapolation_periods: int = 12,
    ) -> dict[str, Any]:
        """Calculate projected run rate."""
