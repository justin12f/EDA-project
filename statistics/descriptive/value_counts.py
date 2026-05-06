"""Value counts and frequency analysis module."""

from __future__ import annotations

import pandas as pd


class ValueCountsCalculator:
    """Calculates absolute and relative value frequencies for any Series dtype.

    Workflow:
        calculator = ValueCountsCalculator()
        result = calculator.calculate(series, top_n=20, include_missing=True)

    Returns a dict with keys:
        - table, n_total, n_unique, n_missing,
          missing_percentage, n_valid, top_n
    """

    def calculate(
        self,
        series: pd.Series,
        top_n: int | None = None,
        include_missing: bool = True,
    ) -> dict:
        """Calculate value frequencies with optional top-N filtering.

        Args:
            series: Pandas Series of any dtype.
            top_n: If set, returns only the top N most frequent values.
            include_missing: Whether to count NaN as a separate category.

        Returns:
            Dictionary with frequency table and summary statistics.
        """
        n_total = len(series)
        n_missing = int(series.isna().sum())
        n_valid = n_total - n_missing

        counts = series.value_counts(dropna=not include_missing)
        relative = counts / n_total

        if top_n is not None:
            counts = counts.head(top_n)
            relative = relative.head(top_n)

        table = [
            {
                "value": str(value),
                "frequency": int(freq),
                "relative_frequency": float(rel),
                "percentage": float(rel * 100),
            }
            for (value, freq), (_, rel) in zip(counts.items(), relative.items())
        ]

        return {
            "table": table,
            "n_total": n_total,
            "n_unique": int(series.nunique(dropna=True)),
            "n_missing": n_missing,
            "missing_percentage": round(n_missing / n_total * 100, 4),
            "n_valid": n_valid,
            "top_n": top_n,
        }
