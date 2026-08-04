"""Abstract contract for one-way ANOVA with post-hoc testing."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class AbstractANOVACalculator(ABC):
    """Contract for one-way ANOVA with optional Tukey HSD post-hoc analysis.

    Functionality
    -------------
    Performs a one-way Analysis of Variance (ANOVA) to test whether the means
    of two or more groups differ significantly:

    * **F-statistic** — ratio of between-group variance to within-group variance.
    * **p-value** — probability of observing the result if all group means are equal.
    * **Rejection verdict** — whether H0 (all group means are equal) is rejected.
    * **Group summaries** — per-group mean and sample size.
    * **Tukey HSD post-hoc** — all pairwise mean comparisons with family-wise
      error rate control, performed when ANOVA is significant and requested.

    Parameters
    ----------
    data:
        Backend-native dataframe containing group and value columns.
    value_column:
        Name of the numeric column containing measurement values.
    group_column:
        Name of the categorical column that defines group membership.
    significance_level:
        Alpha threshold for the omnibus test and post-hoc comparisons.
    run_post_hoc:
        When ``True`` and ANOVA is significant, runs Tukey HSD.

    Returns
    -------
    dict[str, Any]
        Keys: ``test_name``, ``f_statistic``, ``p_value``, ``reject_null``,
        ``significance_level``, ``n_groups``, ``group_means``, ``group_sizes``,
        ``post_hoc`` (list of pairwise dicts or ``None``).

    Raises
    ------
    KeyError
        If ``value_column`` or ``group_column`` are not in ``data``.
    ValueError
        If fewer than 2 groups are found, or any group has fewer than 2 observations.
    """

    @abstractmethod
    def calculate(
        self,
        data: Any,
        value_column: str,
        group_column: str,
        significance_level: float = 0.05,
        run_post_hoc: bool = True,
    ) -> dict[str, Any]:
        """Run one-way ANOVA on ``value_column`` grouped by ``group_column``."""
