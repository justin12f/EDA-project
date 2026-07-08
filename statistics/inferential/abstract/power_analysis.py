"""Abstract contract for Power Analysis."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class AbstractPowerAnalysisCalculator(ABC):
    """Contract for calculating statistical power or required sample size.

    Parameters
    ----------
    effect_size:
        The expected effect size (e.g., Cohen's d).
    alpha:
        Significance level.
    power:
        Desired statistical power (e.g., 0.8). If None, calculates power given n.
    n:
        Sample size. If None, calculates required n given power.
    test_type:
        't_test_ind' (independent t-test), etc.

    Returns
    -------
    dict[str, Any]
        Keys: ``effect_size``, ``alpha``, ``power``, ``n_per_group``, ``test_type``.
    """

    @abstractmethod
    def calculate(
        self,
        effect_size: float,
        alpha: float = 0.05,
        power: float | None = 0.8,
        n: int | None = None,
        test_type: str = "t_test_ind",
    ) -> dict[str, Any]:
        """Calculate power or required sample size."""
