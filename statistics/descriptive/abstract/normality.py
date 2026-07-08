"""Abstract contract for normality testing.

Defines the interface every backend must implement to run a multi-test
normality assessment suite on a numeric column.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class AbstractNormalityTestSuite(ABC):
    """Contract for running a battery of normality tests on a numeric column.

    Functionality
    -------------
    Executes multiple statistical normality tests and aggregates their
    outcomes via **majority vote**:

    * **Shapiro-Wilk** — most powerful test for small samples (n < 5 000).
      Large samples are automatically truncated to preserve test validity.
    * **Anderson-Darling** — compares the empirical CDF against a normal
      reference; more sensitive to deviations in the distribution tails
      than the Kolmogorov-Smirnov test.
    * **Kolmogorov-Smirnov** — tests against a normal distribution fitted
      from the sample mean and standard deviation.

    The ``overall_is_normal`` flag is ``True`` when the majority of the
    individual tests accept the normality hypothesis.

    Implementations must use the native expression API of their backend and
    must not convert data to NumPy or an alien backend inside this method.

    Parameters
    ----------
    data:
        Backend-native dataframe.
    column:
        Name of the numeric column to test.
    significance_level:
        Alpha threshold for all individual tests.  Must be in ``(0, 1)``.

    Returns
    -------
    dict[str, Any]
        Keys:
        * ``overall_is_normal`` — bool (majority vote).
        * ``votes_normal`` — int.
        * ``total_tests`` — int.
        * ``significance_level`` — float.
        * ``tests`` — list of dicts, each with ``test_name``,
          ``statistic``, ``p_value`` (may be ``None`` for
          Anderson-Darling), ``is_normal``, ``note``.

    Raises
    ------
    KeyError
        If ``column`` is not present in ``data``.
    ValueError
        If the column has fewer than 3 non-null observations.
    """

    @abstractmethod
    def run(
        self,
        data: Any,
        column: str,
        significance_level: float = 0.05,
    ) -> dict[str, Any]:
        """Run all normality tests on ``column`` in ``data``."""
