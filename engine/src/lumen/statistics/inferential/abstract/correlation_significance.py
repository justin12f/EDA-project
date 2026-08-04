"""Abstract contract for Correlation Significance."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class AbstractCorrelationSignificanceCalculator(ABC):
    """Contract for calculating correlation and its statistical significance.

    Parameters
    ----------
    data:
        Backend-native dataframe.
    column1:
        Name of the first numeric column.
    column2:
        Name of the second numeric column.
    method:
        'pearson' or 'spearman'.
    significance_level:
        Alpha threshold.

    Returns
    -------
    dict[str, Any]
        Keys: ``correlation``, ``p_value``, ``method``, ``reject_null``,
        ``significance_level``, ``n``.
    """

    @abstractmethod
    def calculate(
        self,
        data: Any,
        column1: str,
        column2: str,
        method: str = "pearson",
        significance_level: float = 0.05,
    ) -> dict[str, Any]:
        """Calculate correlation and significance."""
