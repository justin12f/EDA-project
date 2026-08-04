"""Abstract contract for Change Points."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class AbstractChangePointsCalculator(ABC):
    """Contract for detecting structural breaks or change points in a time series.

    Parameters
    ----------
    data:
        Backend-native dataframe (assumed ordered by time).
    value_column:
        Column containing the time series values.
    penalty:
        Penalty value for the changepoint detection algorithm.

    Returns
    -------
    dict[str, Any]
        Dictionary with detected changepoints.
    """

    @abstractmethod
    def calculate(
        self,
        data: Any,
        value_column: str,
        penalty: float = 1.0,
    ) -> dict[str, Any]:
        """Detect change points."""
