"""Percentile calculator module."""

from __future__ import annotations

import numpy as np

_DEFAULT_PERCENTILES: list[int] = [1, 5, 10, 25, 50, 75, 90, 95, 99]


class PercentileOutlierDetector:
    """Detects values outside user-defined percentile bounds."""

    def detect(
        self, data: np.ndarray, lower_bound: float, upper_bound: float
    ) -> dict:
        """Count values outside the given bounds.

        Args:
            data: 1D numerical array.
            lower_bound: Numeric lower bound value.
            upper_bound: Numeric upper bound value.

        Returns:
            Dict with outlier counts and percentage.
        """
        n_below = int(np.sum(data < lower_bound))
        n_above = int(np.sum(data > upper_bound))
        n_total = len(data)

        return {
            "n_below_lower_bound": n_below,
            "n_above_upper_bound": n_above,
            "outlier_count": n_below + n_above,
            "outlier_percentage": (n_below + n_above) / n_total * 100,
        }


class PercentilesCalculator:
    """Calculates configurable percentiles with optional outlier detection.

    Workflow:
        calculator = PercentilesCalculator()
        result = calculator.calculate(
            data,
            percentiles=[5, 25, 50, 75, 95],
            outlier_bounds=(1, 99),
        )

    Returns a dict with keys:
        - percentiles (dict), n, outlier_detection (optional)
    """

    def __init__(self) -> None:
        self._outlier_detector = PercentileOutlierDetector()

    def calculate(
        self,
        data: np.ndarray,
        percentiles: list[int] | None = None,
        outlier_bounds: tuple[int, int] | None = (1, 99),
    ) -> dict:
        """Calculate percentiles and optional outlier detection.

        Args:
            data: 1D numerical array.
            percentiles: Percentile values in [0, 100]. Defaults to standard set.
            outlier_bounds: (lower_pct, upper_pct) for outlier detection.
                Set to None to skip.

        Returns:
            Dictionary with percentile values and outlier stats.

        Raises:
            ValueError: If data is empty or any percentile is out of [0, 100].
        """
        if len(data) == 0:
            raise ValueError("Data array cannot be empty.")

        pct_list = percentiles if percentiles is not None else _DEFAULT_PERCENTILES

        if any(p < 0 or p > 100 for p in pct_list):
            raise ValueError("All percentile values must be within [0, 100].")

        pct_values = np.percentile(data, pct_list)
        percentile_map = {f"p{p}": float(v) for p, v in zip(pct_list, pct_values)}

        result: dict = {"percentiles": percentile_map, "n": len(data)}

        if outlier_bounds is not None:
            lower_val = float(np.percentile(data, outlier_bounds[0]))
            upper_val = float(np.percentile(data, outlier_bounds[1]))
            detection = self._outlier_detector.detect(data, lower_val, upper_val)
            detection["bounds"] = {
                f"p{outlier_bounds[0]}": lower_val,
                f"p{outlier_bounds[1]}": upper_val,
            }
            result["outlier_detection"] = detection

        return result
