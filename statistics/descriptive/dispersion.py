"""Dispersion and spread measures module."""

from __future__ import annotations

import numpy as np
from scipy import stats


class VarianceCalculator:
    """Sample or population variance."""

    def calculate(self, data: np.ndarray, ddof: int) -> float:
        return float(np.var(data, ddof=ddof))


class StandardDeviationCalculator:
    """Sample or population standard deviation."""

    def calculate(self, data: np.ndarray, ddof: int) -> float:
        return float(np.std(data, ddof=ddof))


class RangeCalculator:
    """Statistical range: min, max, and spread."""

    def calculate(self, data: np.ndarray) -> dict[str, float]:
        return {
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "range": float(np.max(data) - np.min(data)),
        }


class IQRCalculator:
    """Interquartile range: Q1, Q3, and IQR."""

    def calculate(self, data: np.ndarray) -> dict[str, float]:
        q1 = float(np.percentile(data, 25))
        q3 = float(np.percentile(data, 75))
        return {"q1": q1, "q3": q3, "iqr": q3 - q1}


class MADCalculator:
    """Median Absolute Deviation — robust spread measure, resistant to outliers."""

    def calculate(self, data: np.ndarray) -> float:
        return float(stats.median_abs_deviation(data))


class CoefficientOfVariationCalculator:
    """CV = std / |mean| — normalized spread, scale-independent comparison."""

    def calculate(self, std: float, mean: float) -> float:
        if mean == 0:
            return float("inf")
        return float(std / abs(mean))


class DispersionCalculator:
    """Calculates all dispersion measures for a numerical array.

    Workflow:
        calculator = DispersionCalculator()
        result = calculator.calculate(data, ddof=1)

    Returns a dict with keys:
        - variance, std, range (min/max/range), iqr (q1/q3/iqr),
          mad, coefficient_of_variation, ddof
    """

    def __init__(self) -> None:
        self._variance_calc = VarianceCalculator()
        self._std_calc = StandardDeviationCalculator()
        self._range_calc = RangeCalculator()
        self._iqr_calc = IQRCalculator()
        self._mad_calc = MADCalculator()
        self._cv_calc = CoefficientOfVariationCalculator()

    def calculate(self, data: np.ndarray, ddof: int = 1) -> dict:
        """Calculate all dispersion measures.

        Args:
            data: 1D numerical array.
            ddof: Delta degrees of freedom for variance and std (1 = sample).

        Returns:
            Dictionary with all dispersion measures.

        Raises:
            ValueError: If data is empty.
        """
        if len(data) == 0:
            raise ValueError("Data array cannot be empty.")

        std = self._std_calc.calculate(data, ddof)
        mean = float(np.mean(data))

        return {
            "variance": self._variance_calc.calculate(data, ddof),
            "std": std,
            "range": self._range_calc.calculate(data),
            "iqr": self._iqr_calc.calculate(data),
            "mad": self._mad_calc.calculate(data),
            "coefficient_of_variation": self._cv_calc.calculate(std, mean),
            "ddof": ddof,
        }
