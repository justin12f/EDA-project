"""Abstract contract for Hypothesis Testing."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class AbstractHypothesisTestSuite(ABC):
    """Contract for executing standard hypothesis tests (t-tests, Mann-Whitney).

    Parameters
    ----------
    data:
        Backend-native dataframe.
    value_column:
        Name of the numeric column containing values.
    group_column:
        Name of the categorical column indicating group membership (must have 2 groups).
    test_type:
        't_test_ind' (independent t-test) or 'mann_whitney' (non-parametric).
    significance_level:
        Alpha threshold.

    Returns
    -------
    dict[str, Any]
        Keys: ``test_name``, ``statistic``, ``p_value``, ``reject_null``,
        ``significance_level``, ``group_stats``.
    """

    @abstractmethod
    def run(
        self,
        data: Any,
        value_column: str,
        group_column: str,
        test_type: str = "t_test_ind",
        significance_level: float = 0.05,
    ) -> dict[str, Any]:
        """Run hypothesis test between two groups."""
