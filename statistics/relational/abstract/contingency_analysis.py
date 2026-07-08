"""Abstract contract for Contingency Analysis."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class AbstractContingencyAnalysisCalculator(ABC):
    """Contract for computing contingency tables and Cramer's V.

    Parameters
    ----------
    data:
        Backend-native dataframe.
    col1:
        First categorical column name.
    col2:
        Second categorical column name.

    Returns
    -------
    dict[str, Any]
        Dictionary with contingency table, Chi2, Cramer's V, etc.
    """

    @abstractmethod
    def calculate(
        self,
        data: Any,
        col1: str,
        col2: str,
    ) -> dict[str, Any]:
        """Calculate contingency metrics."""
