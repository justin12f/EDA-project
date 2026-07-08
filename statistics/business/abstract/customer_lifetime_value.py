"""Abstract contract for Customer Lifetime Value."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class AbstractCustomerLifetimeValueCalculator(ABC):
    """Contract for calculating Customer Lifetime Value (CLV).

    Parameters
    ----------
    data:
        Backend-native dataframe with transaction data.
    customer_column:
        Customer identifier column.
    order_value_column:
        Transaction amount column.
    date_column:
        Transaction date column.
    discount_rate:
        Annual discount rate as decimal.
    margin_rate:
        Gross margin as decimal.
    periods_per_year:
        Periods per year (12=monthly frequency).

    Returns
    -------
    dict[str, Any]
        Dictionary with per-customer CLV and portfolio summary.
    """

    @abstractmethod
    def calculate(
        self,
        data: Any,
        customer_column: str,
        order_value_column: str,
        date_column: str,
        discount_rate: float = 0.1,
        margin_rate: float = 0.3,
        periods_per_year: int = 12,
    ) -> dict[str, Any]:
        """Calculate CLV from transactional data."""
