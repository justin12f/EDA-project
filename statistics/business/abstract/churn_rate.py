"""Abstract contract for Churn Rate calculation."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class AbstractChurnRateCalculator(ABC):
    """Contract for calculating churn rate.

    Parameters
    ----------
    data:
        Backend-native dataframe containing customer subscription data.
    customer_id_column:
        Name of the customer identifier column.
    start_date_column:
        Name of the subscription start date column.
    end_date_column:
        Name of the subscription end date column (null if active).
    analysis_start:
        Start of the analysis period (string YYYY-MM-DD or datetime).
    analysis_end:
        End of the analysis period (string YYYY-MM-DD or datetime).

    Returns
    -------
    dict[str, Any]
        Dictionary with churn rate and related metrics.
    """

    @abstractmethod
    def calculate(
        self,
        data: Any,
        customer_id_column: str,
        start_date_column: str,
        end_date_column: str,
        analysis_start: str,
        analysis_end: str,
    ) -> dict[str, Any]:
        """Calculate customer churn rate for a specific period."""
