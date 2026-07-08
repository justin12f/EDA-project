"""Abstract contract for Granger Causality."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class AbstractGrangerCausalityCalculator(ABC):
    """Contract for calculating Granger causality between two time series.

    Parameters
    ----------
    data:
        Backend-native dataframe (assumed ordered by time).
    target_column:
        The target column (Y).
    predictor_column:
        The predictor column (X).
    max_lag:
        Maximum lag to evaluate.

    Returns
    -------
    dict[str, Any]
        Dictionary with F-statistics and p-values for each lag.
    """

    @abstractmethod
    def calculate(
        self,
        data: Any,
        target_column: str,
        predictor_column: str,
        max_lag: int = 5,
    ) -> dict[str, Any]:
        """Test Granger Causality."""
