"""Abstract contract for Interaction Effects."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class AbstractInteractionEffectsCalculator(ABC):
    """Contract for calculating interaction effects between features.

    Parameters
    ----------
    data:
        Backend-native dataframe.
    target_column:
        The target variable (Y).
    features:
        List of predictor variables (X).

    Returns
    -------
    dict[str, Any]
        Dictionary with interaction terms and their significance/strength.
    """

    @abstractmethod
    def calculate(
        self,
        data: Any,
        target_column: str,
        features: list[str],
    ) -> dict[str, Any]:
        """Calculate interaction effects."""
