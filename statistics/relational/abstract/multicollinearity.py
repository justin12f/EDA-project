"""Abstract contract for Multicollinearity."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class AbstractMulticollinearityCalculator(ABC):
    """Contract for computing Variance Inflation Factor (VIF).

    Parameters
    ----------
    data:
        Backend-native dataframe.
    columns:
        List of numeric columns.

    Returns
    -------
    dict[str, Any]
        Dictionary mapping column names to their VIF scores.
    """

    @abstractmethod
    def calculate(
        self,
        data: Any,
        columns: list[str],
    ) -> dict[str, Any]:
        """Calculate multicollinearity (VIF)."""
