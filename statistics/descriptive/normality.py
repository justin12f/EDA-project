"""Normality testing suite module."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class NormalityTestResult:
    """Immutable result of a single normality test."""

    test_name: str
    statistic: float
    p_value: Optional[float]
    is_normal: bool
    significance_level: float
    note: Optional[str] = None


class BaseNormalityTest(ABC):
    """Abstract base for all normality tests."""

    @abstractmethod
    def test(self, data: np.ndarray, significance_level: float) -> NormalityTestResult:
        """Run the normality test.

        Args:
            data: 1D numerical array.
            significance_level: Alpha level for the hypothesis test.

        Returns:
            NormalityTestResult with test outcome.
        """


class ShapiroWilkTest(BaseNormalityTest):
    """Shapiro-Wilk test. Most powerful for small samples (n < 5000).

    Truncates to _MAX_SAMPLE_SIZE if data is larger to preserve test validity.
    """

    _MAX_SAMPLE_SIZE: int = 5_000

    def test(self, data: np.ndarray, significance_level: float) -> NormalityTestResult:
        sample = (
            data[: self._MAX_SAMPLE_SIZE]
            if len(data) > self._MAX_SAMPLE_SIZE
            else data
        )
        statistic, p_value = stats.shapiro(sample)
        note = (
            f"Sample truncated to {self._MAX_SAMPLE_SIZE} for test validity."
            if len(data) > self._MAX_SAMPLE_SIZE
            else None
        )

        return NormalityTestResult(
            test_name="shapiro_wilk",
            statistic=float(statistic),
            p_value=float(p_value),
            is_normal=float(p_value) > significance_level,
            significance_level=significance_level,
            note=note,
        )


class AndersonDarlingTest(BaseNormalityTest):
    """Anderson-Darling test. More sensitive to distribution tails than KS.

    Returns critical value comparison instead of a p-value.
    """

    # scipy returns critical values at significance levels: [15%, 10%, 5%, 2.5%, 1%]
    _SIGNIFICANCE_TO_INDEX: dict[float, int] = {
        0.15: 0,
        0.10: 1,
        0.05: 2,
        0.025: 3,
        0.01: 4,
    }
    _DEFAULT_INDEX: int = 2  # 5%

    def test(self, data: np.ndarray, significance_level: float) -> NormalityTestResult:
        result = stats.anderson(data, dist="norm")
        index = self._SIGNIFICANCE_TO_INDEX.get(
            significance_level, self._DEFAULT_INDEX
        )
        critical_value = result.critical_values[index]
        is_normal = bool(result.statistic < critical_value)

        return NormalityTestResult(
            test_name="anderson_darling",
            statistic=float(result.statistic),
            p_value=None,
            is_normal=is_normal,
            significance_level=significance_level,
            note=f"Critical value at α={significance_level}: {critical_value:.4f}",
        )


class KolmogorovSmirnovTest(BaseNormalityTest):
    """Kolmogorov-Smirnov test against a fitted normal distribution.

    Note: KS is less powerful than Shapiro-Wilk for small samples.
    """

    def test(self, data: np.ndarray, significance_level: float) -> NormalityTestResult:
        statistic, p_value = stats.kstest(
            data, "norm", args=(float(data.mean()), float(data.std()))
        )

        return NormalityTestResult(
            test_name="kolmogorov_smirnov",
            statistic=float(statistic),
            p_value=float(p_value),
            is_normal=float(p_value) > significance_level,
            significance_level=significance_level,
        )


class NormalityTestSuite:
    """Runs all normality tests and aggregates results via majority vote.

    Workflow:
        suite = NormalityTestSuite()
        result = suite.run(data, significance_level=0.05)

    Returns a dict with keys:
        - overall_is_normal: bool (majority vote)
        - votes_normal: int
        - total_tests: int
        - significance_level: float
        - tests: list of per-test result dicts
    """

    _MINIMUM_SAMPLE_SIZE: int = 3

    def __init__(self) -> None:
        self._tests: list[BaseNormalityTest] = [
            ShapiroWilkTest(),
            AndersonDarlingTest(),
            KolmogorovSmirnovTest(),
        ]

    def run(self, data: np.ndarray, significance_level: float = 0.05) -> dict:
        """Run all normality tests and return aggregated results.

        Args:
            data: 1D numerical array.
            significance_level: Alpha level for all hypothesis tests.

        Returns:
            Dictionary with overall verdict and per-test details.

        Raises:
            ValueError: If data has fewer than the minimum required samples.
        """
        if len(data) < self._MINIMUM_SAMPLE_SIZE:
            raise ValueError(
                f"Normality tests require at least {self._MINIMUM_SAMPLE_SIZE} "
                f"data points. Got {len(data)}."
            )

        results = [t.test(data, significance_level) for t in self._tests]
        normal_votes = sum(1 for r in results if r.is_normal)

        return {
            "overall_is_normal": normal_votes >= (len(results) // 2 + 1),
            "votes_normal": normal_votes,
            "total_tests": len(results),
            "significance_level": significance_level,
            "tests": [
                {
                    "test_name": r.test_name,
                    "statistic": r.statistic,
                    "p_value": r.p_value,
                    "is_normal": r.is_normal,
                    "note": r.note,
                }
                for r in results
            ],
        }
