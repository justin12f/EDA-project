"""Abstract contract for Chi-Square tests."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class AbstractChiSquareCalculator(ABC):
    """Contract for Chi-Square test of independence or goodness-of-fit.

    Functionality
    -------------
    Performs a Chi-Square test on categorical data:
    * Goodness-of-fit (1 variable)
    * Independence (2 variables, cross-tabulation)

    Parameters
    ----------
    data:
        Backend-native dataframe.
    column1:
        Name of the primary categorical column.
    column2:
        Name of the secondary categorical column (optional). If None,
        performs Goodness-of-fit on column1.
    significance_level:
        Alpha threshold.

    Returns
    -------
    dict[str, Any]
        Keys: ``test_name``, ``statistic``, ``p_value``, ``dof``,
        ``reject_null``, ``significance_level``, ``expected_frequencies``.
    """

    @abstractmethod
    def calculate(
        self,
        data: Any,
        column1: str,
        column2: str | None = None,
        significance_level: float = 0.05,
    ) -> dict[str, Any]:
        """Run Chi-Square test."""
