"""Chi-square independence and goodness-of-fit tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


class ContingencyTableBuilder:
    """Builds and validates a contingency table from two categorical Series."""

    def build(
        self, series_a: pd.Series, series_b: pd.Series
    ) -> pd.DataFrame:
        """Build a crosstab contingency table.

        Args:
            series_a: First categorical Series.
            series_b: Second categorical Series.

        Returns:
            Contingency table as a DataFrame.

        Raises:
            ValueError: If series have different lengths or are empty.
        """
        if len(series_a) != len(series_b):
            raise ValueError(
                f"Both series must have equal length. "
                f"Got {len(series_a)} and {len(series_b)}."
            )
        if len(series_a) == 0:
            raise ValueError("Input series cannot be empty.")

        return pd.crosstab(series_a, series_b)


class CramersVCalculator:
    """Calculates Cramér's V — effect size for chi-square tests.

    V ranges from 0 (no association) to 1 (perfect association).
    Uses bias-corrected formula for small samples.
    """

    def calculate(
        self, chi2: float, n: int, n_rows: int, n_cols: int
    ) -> float:
        """Calculate bias-corrected Cramér's V.

        Args:
            chi2: Chi-square test statistic.
            n: Total number of observations.
            n_rows: Number of rows in contingency table.
            n_cols: Number of columns in contingency table.

        Returns:
            Cramér's V value in [0, 1].
        """
        phi2 = chi2 / n
        r_corrected = n_rows - (n_rows - 1) ** 2 / (n - 1)
        k_corrected = n_cols - (n_cols - 1) ** 2 / (n - 1)
        denominator = min(r_corrected - 1, k_corrected - 1)

        if denominator <= 0:
            return 0.0

        return float(np.sqrt(phi2 / denominator))


class ChiSquareTestCalculator:
    """Chi-square independence test between two categorical variables.

    Workflow:
        calculator = ChiSquareTestCalculator()
        result = calculator.calculate(
            series_a=df["gender"],
            series_b=df["purchase"],
            significance_level=0.05,
        )
    """

    _EXPECTED_FREQUENCY_THRESHOLD: float = 5.0
    _EXPECTED_FREQUENCY_RATIO_WARNING: float = 0.2

    def __init__(self) -> None:
        self._table_builder = ContingencyTableBuilder()
        self._cramers_v = CramersVCalculator()

    def calculate(
        self,
        series_a: pd.Series,
        series_b: pd.Series,
        significance_level: float = 0.05,
    ) -> dict:
        """Run chi-square independence test.

        Args:
            series_a: First categorical variable.
            series_b: Second categorical variable.
            significance_level: Alpha threshold.

        Returns:
            Dictionary with test results, effect size, and reliability warning.

        Raises:
            ValueError: If inputs are invalid.
        """
        contingency_table = self._table_builder.build(series_a, series_b)
        observed = contingency_table.to_numpy()
        chi2, p_value, dof, expected = stats.chi2_contingency(observed)

        n = int(observed.sum())
        n_rows, n_cols = observed.shape
        reject_null = float(p_value) < significance_level

        low_expected_ratio = float(
            np.mean(expected < self._EXPECTED_FREQUENCY_THRESHOLD)
        )
        reliability_warning = (
            low_expected_ratio > self._EXPECTED_FREQUENCY_RATIO_WARNING
        )

        cramers_v = self._cramers_v.calculate(float(chi2), n, n_rows, n_cols)

        return {
            "test_name": "chi_square_independence",
            "chi2_statistic": float(chi2),
            "p_value": float(p_value),
            "degrees_of_freedom": int(dof),
            "reject_null": reject_null,
            "significance_level": significance_level,
            "cramers_v": cramers_v,
            "n_observations": n,
            "contingency_table_shape": (n_rows, n_cols),
            "reliability_warning": reliability_warning,
            "low_expected_frequency_ratio": round(low_expected_ratio, 4),
        }
