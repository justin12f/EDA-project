"""Abstract contract for Correlation Matrix."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class AbstractCorrelationMatrixCalculator(ABC):
    """Contract for calculating correlation matrix.

    Parameters
    ----------
    data:
        Backend-native dataframe.
    columns:
        List of numeric column names to include.
    method:
        'pearson' or 'spearman'.

    Returns
    -------
    dict[str, Any]
        Dictionary with the correlation matrix and p-values matrix.
    """

    @abstractmethod
    def calculate(
        self,
        data: Any,
        columns: list[str],
        method: str = "pearson",
    ) -> dict[str, Any]:
        """Calculate correlation matrix."""
