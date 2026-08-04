"""Abstract contract for Forecast Accuracy."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class AbstractForecastAccuracyCalculator(ABC):
    """Contract for calculating forecast accuracy metrics (MAE, RMSE, MAPE).

    Parameters
    ----------
    data:
        Backend-native dataframe.
    actual_column:
        Column containing actual values.
    forecast_column:
        Column containing predicted values.

    Returns
    -------
    dict[str, Any]
        Dictionary with MAE, RMSE, MAPE, etc.
    """

    @abstractmethod
    def calculate(
        self,
        data: Any,
        actual_column: str,
        forecast_column: str,
    ) -> dict[str, Any]:
        """Calculate forecast accuracy."""
