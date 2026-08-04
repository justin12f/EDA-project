"""Abstract contract for Cross Correlation."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class AbstractCrossCorrelationCalculator(ABC):
    """Contract for calculating cross-correlation (time-lagged correlation).

    Parameters
    ----------
    data:
        Backend-native dataframe (assumed ordered by time).
    col1:
        First numeric column.
    col2:
        Second numeric column.
    max_lag:
        Maximum lag to evaluate.

    Returns
    -------
    dict[str, Any]
        Dictionary with lags and their correlation values.
    """

    @abstractmethod
    def calculate(
        self,
        data: Any,
        col1: str,
        col2: str,
        max_lag: int = 10,
    ) -> dict[str, Any]:
        """Calculate cross correlation across lags."""
