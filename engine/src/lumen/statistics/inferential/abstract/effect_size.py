"""Abstract contract for Effect Size calculations."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class AbstractEffectSizeCalculator(ABC):
    """Contract for calculating effect sizes (Cohen's d, Hedges' g).

    Parameters
    ----------
    data:
        Backend-native dataframe.
    value_column:
        Name of the numeric column containing values.
    group_column:
        Name of the categorical column indicating group membership (must have 2 groups).

    Returns
    -------
    dict[str, Any]
        Keys: ``cohens_d``, ``hedges_g``, ``interpretation``, ``group_stats``.
    """

    @abstractmethod
    def calculate(
        self,
        data: Any,
        value_column: str,
        group_column: str,
    ) -> dict[str, Any]:
        """Calculate effect sizes between two groups."""
