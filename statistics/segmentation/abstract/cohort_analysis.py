"""Abstract contract for Cohort Analysis."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class AbstractCohortAnalysisCalculator(ABC):
    """Contract for performing cohort analysis (e.g. retention matrix).

    Parameters
    ----------
    data:
        Backend-native dataframe.
    user_column:
        User/Entity ID.
    date_column:
        Event date.
    period:
        Time period for cohorts ('month', 'week', 'day').

    Returns
    -------
    dict[str, Any]
        Dictionary with cohort matrix and retention rates.
    """

    @abstractmethod
    def calculate(
        self,
        data: Any,
        user_column: str,
        date_column: str,
        period: str = "month",
    ) -> dict[str, Any]:
        """Calculate cohort retention matrix."""
