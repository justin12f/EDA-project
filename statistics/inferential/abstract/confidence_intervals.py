"""Abstract contract for Confidence Intervals."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class AbstractConfidenceIntervalCalculator(ABC):
    """Contract for calculating confidence intervals of a mean or proportion.

    Parameters
    ----------
    data:
        Backend-native dataframe.
    column:
        Name of the numeric column.
    confidence_level:
        Confidence level for the interval (e.g., 0.95).
    method:
        't' (Student's t-distribution) or 'z' (Normal distribution).

    Returns
    -------
    dict[str, Any]
        Keys: ``mean``, ``margin_of_error``, ``lower_bound``, ``upper_bound``,
        ``confidence_level``, ``method``, ``n``, ``std_error``.
    """

    @abstractmethod
    def calculate(
        self,
        data: Any,
        column: str,
        confidence_level: float = 0.95,
        method: str = "t",
    ) -> dict[str, Any]:
        """Calculate confidence interval."""
