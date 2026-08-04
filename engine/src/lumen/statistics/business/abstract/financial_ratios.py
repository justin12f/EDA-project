"""Abstract contract for Financial Ratios."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class AbstractFinancialRatiosCalculator(ABC):
    """Contract for calculating financial ratios (e.g. ROI, Margins).

    Parameters
    ----------
    data:
        Backend-native dataframe containing financial metrics.
    revenue_column:
        Column for revenue.
    cost_column:
        Column for costs.
    equity_column:
        Column for equity (optional).
    assets_column:
        Column for assets (optional).

    Returns
    -------
    dict[str, Any]
        Dictionary with computed financial ratios.
    """

    @abstractmethod
    def calculate(
        self,
        data: Any,
        revenue_column: str,
        cost_column: str,
        equity_column: str | None = None,
        assets_column: str | None = None,
    ) -> dict[str, Any]:
        """Calculate financial ratios from data."""
