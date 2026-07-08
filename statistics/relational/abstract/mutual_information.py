"""Abstract contract for Mutual Information."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class AbstractMutualInformationCalculator(ABC):
    """Contract for calculating mutual information between variables.

    Parameters
    ----------
    data:
        Backend-native dataframe.
    target_column:
        The target column.
    feature_columns:
        List of feature columns.
    is_target_discrete:
        Whether the target is categorical (True) or continuous (False).

    Returns
    -------
    dict[str, Any]
        Dictionary with mutual information scores for each feature.
    """

    @abstractmethod
    def calculate(
        self,
        data: Any,
        target_column: str,
        feature_columns: list[str],
        is_target_discrete: bool = True,
    ) -> dict[str, Any]:
        """Calculate mutual information scores."""
