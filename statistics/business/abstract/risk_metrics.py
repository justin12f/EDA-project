"""Abstract contract for Risk Metrics."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class AbstractRiskMetricsCalculator(ABC):
    """Contract for calculating business risk metrics (e.g. VaR, Sharpe).

    Parameters
    ----------
    data:
        Backend-native dataframe containing returns/profits.
    returns_column:
        Column containing periodic returns.
    risk_free_rate:
        Risk-free rate for Sharpe/Sortino ratio.
    confidence_level:
        Confidence level for Value at Risk (VaR).

    Returns
    -------
    dict[str, Any]
        Dictionary with risk metrics.
    """

    @abstractmethod
    def calculate(
        self,
        data: Any,
        returns_column: str,
        risk_free_rate: float = 0.0,
        confidence_level: float = 0.95,
    ) -> dict[str, Any]:
        """Calculate financial risk metrics."""
