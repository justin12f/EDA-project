"""Abstract contract for skewness and kurtosis calculations.

Defines the interface every backend must implement to compute skewness
and kurtosis with actionable distributional interpretations.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class AbstractSkewnessKurtosisCalculator(ABC):
    """Contract for computing skewness and kurtosis on a numeric column.

    Functionality
    -------------
    Computes both moment-based shape statistics together with human-readable
    interpretations:

    * **Skewness** — Fisher-Pearson standardised moment coefficient.
      Positive values indicate a right tail; negative values a left tail.
      Accompanied by:
      - *direction*: ``"right (positive)"``, ``"left (negative)"``, or
        ``"none"``.
      - *severity*: ``"approximately symmetric"``, ``"moderately skewed"``,
        or ``"highly skewed"``.
      - *recommended_action*: transformation suggestion.

    * **Excess kurtosis** — Fisher's definition (normal distribution = 0).
      ``pearson_kurtosis`` (= excess + 3) is also returned.
      Accompanied by:
      - *distribution_type*: mesokurtic / leptokurtic / platykurtic.
      - *recommended_action*: model selection advice.

    Implementations must use the native expression API of their backend.
    Requires at least 4 non-null observations.

    Parameters
    ----------
    data:
        Backend-native dataframe.
    column:
        Name of the numeric column to analyse.
    bias:
        When ``True`` (default), uses the biased estimator (Fisher's
        definition, as in SciPy's default).

    Returns
    -------
    dict[str, Any]
        Keys: ``skewness``, ``excess_kurtosis``, ``pearson_kurtosis``,
        ``skewness_interpretation`` (dict), ``kurtosis_interpretation``
        (dict).

    Raises
    ------
    KeyError
        If ``column`` is not present in ``data``.
    ValueError
        If the column has fewer than 4 non-null observations.
    """

    @abstractmethod
    def calculate(
        self,
        data: Any,
        column: str,
        bias: bool = True,
    ) -> dict[str, Any]:
        """Calculate skewness and kurtosis for ``column`` in ``data``."""
