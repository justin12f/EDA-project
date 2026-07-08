"""Abstract contract for Partial Correlation."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class AbstractPartialCorrelationCalculator(ABC):
    """Contract for calculating partial correlation between two variables controlling for others.

    Parameters
    ----------
    data:
        Backend-native dataframe.
    col1:
        First numeric column.
    col2:
        Second numeric column.
    covariates:
        List of control variables.

    Returns
    -------
    dict[str, Any]
        Dictionary with partial correlation and significance.
    """

    @abstractmethod
    def calculate(
        self,
        data: Any,
        col1: str,
        col2: str,
        covariates: list[str],
    ) -> dict[str, Any]:
        """Calculate partial correlation."""
