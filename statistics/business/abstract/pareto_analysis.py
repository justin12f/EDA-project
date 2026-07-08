"""Abstract contract for Pareto Analysis."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class AbstractParetoAnalysisCalculator(ABC):
    """Contract for performing Pareto (80/20) analysis.

    Parameters
    ----------
    data:
        Backend-native dataframe.
    entity_column:
        Entity identifier (e.g. customer_id, product_id).
    value_column:
        Metric to aggregate (e.g. revenue, volume).

    Returns
    -------
    dict[str, Any]
        Dictionary with cumulative percentage distributions and segments.
    """

    @abstractmethod
    def calculate(
        self,
        data: Any,
        entity_column: str,
        value_column: str,
    ) -> dict[str, Any]:
        """Perform Pareto analysis on entities by value."""
