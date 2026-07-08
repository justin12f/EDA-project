"""Abstract contract for Cyclical Patterns."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class AbstractCyclicalPatternsCalculator(ABC):
    """Contract for analyzing cyclical patterns via FFT or similar.

    Parameters
    ----------
    data:
        Backend-native dataframe (assumed ordered by time).
    value_column:
        Column containing the time series values.

    Returns
    -------
    dict[str, Any]
        Dictionary with dominant frequencies and periods.
    """

    @abstractmethod
    def calculate(
        self,
        data: Any,
        value_column: str,
    ) -> dict[str, Any]:
        """Detect cyclical patterns."""
