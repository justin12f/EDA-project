"""Abstract contract for Lag Features."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class AbstractLagFeaturesCalculator(ABC):
    """Contract for generating lagged features of a time series.

    Parameters
    ----------
    data:
        Backend-native dataframe.
    value_column:
        Column to lag.
    lags:
        List of integer lag amounts (e.g., [1, 2, 7]).

    Returns
    -------
    dict[str, Any]
        Dictionary with generated lags.
    """

    @abstractmethod
    def calculate(
        self,
        data: Any,
        value_column: str,
        lags: list[int],
    ) -> dict[str, Any]:
        """Generate lag features."""
