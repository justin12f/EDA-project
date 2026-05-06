"""Cross-correlation analysis with lag detection between two time series."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LagResult:
    """Immutable result for a single lag in a cross-correlation analysis."""

    lag: int
    correlation: float
    abs_correlation: float


class SeriesStandardizer:
    """Zero-mean unit-variance standardization for cross-correlation validity.

    Cross-correlation without standardization conflates scale and correlation,
    producing results that are not bounded in [-1, 1].
    """

    def standardize(self, series: np.ndarray) -> np.ndarray:
        """Standardize a 1D array to zero mean and unit variance.

        Args:
            series: Input 1D numerical array.

        Returns:
            Standardized array.

        Raises:
            ValueError: If the series has zero variance (constant).
        """
        std = float(series.std())
        if std == 0.0:
            raise ValueError(
                "Cannot standardize a constant series (zero variance). "
                "Cross-correlation is undefined for constant inputs."
            )
        return (series - series.mean()) / std


class LaggedCorrelationComputer:
    """Computes Pearson correlation at each specified lag.

    Positive lag: x leads y (x predicts y).
    Negative lag: y leads x (y predicts x).
    """

    def compute(
        self,
        x_std: np.ndarray,
        y_std: np.ndarray,
        max_lag: int,
    ) -> list[LagResult]:
        """Compute correlations across the full lag range [-max_lag, +max_lag].

        Args:
            x_std: Standardized series x.
            y_std: Standardized series y.
            max_lag: Maximum absolute lag to evaluate.

        Returns:
            List of LagResult for each lag in [-max_lag, +max_lag].
        """
        results: list[LagResult] = []

        for lag in range(-max_lag, max_lag + 1):
            if lag >= 0:
                x_slice = x_std[lag:] if lag > 0 else x_std
                y_slice = y_std[:len(x_std) - lag] if lag > 0 else y_std
            else:
                abs_lag = abs(lag)
                x_slice = x_std[:len(x_std) - abs_lag]
                y_slice = y_std[abs_lag:]

            if len(x_slice) < 3:
                continue

            correlation = float(np.corrcoef(x_slice, y_slice)[0, 1])

            if np.isnan(correlation):
                continue

            results.append(
                LagResult(
                    lag=lag,
                    correlation=correlation,
                    abs_correlation=abs(correlation),
                )
            )

        return results


class PeakLagDetector:
    """Identifies the lag with the highest absolute correlation."""

    def detect(self, lag_results: list[LagResult]) -> LagResult | None:
        """Find the lag with maximum |correlation|.

        Args:
            lag_results: List of LagResult objects.

        Returns:
            LagResult at peak correlation, or None if list is empty.
        """
        if not lag_results:
            return None
        return max(lag_results, key=lambda r: r.abs_correlation)


class CrossCorrelationCalculator:
    """Cross-correlation between two time series across a range of lags.

    Workflow:
        calculator = CrossCorrelationCalculator()
        result = calculator.calculate(
            data_frame=df,
            column_x="advertising_spend",
            column_y="sales",
            max_lag=12,   # optional, default 10
        )
    """

    _DEFAULT_MAX_LAG: int = 10
    _MINIMUM_SAMPLE_SIZE: int = 10

    def __init__(self) -> None:
        self._standardizer = SeriesStandardizer()
        self._lag_computer = LaggedCorrelationComputer()
        self._peak_detector = PeakLagDetector()

    def calculate(
        self,
        data_frame: pd.DataFrame,
        column_x: str,
        column_y: str,
        max_lag: int = _DEFAULT_MAX_LAG,
    ) -> dict:
        """Compute cross-correlation across lag range.

        Args:
            data_frame: Source DataFrame.
            column_x: First time series column.
            column_y: Second time series column.
            max_lag: Maximum lag to evaluate in both directions.

        Returns:
            Dict with lag table, peak lag, and direction interpretation.

        Raises:
            KeyError: If columns are not found in DataFrame.
            ValueError: If max_lag is invalid or data is insufficient.
        """
        for col in (column_x, column_y):
            if col not in data_frame.columns:
                raise KeyError(f"Column '{col}' not found in DataFrame.")

        if max_lag < 1:
            raise ValueError(f"max_lag must be >= 1. Got {max_lag}.")

        clean = data_frame[[column_x, column_y]].dropna()

        if len(clean) < self._MINIMUM_SAMPLE_SIZE:
            raise ValueError(
                f"At least {self._MINIMUM_SAMPLE_SIZE} observations required. "
                f"Got {len(clean)}."
            )

        if max_lag >= len(clean):
            raise ValueError(
                f"max_lag ({max_lag}) must be less than n_observations ({len(clean)})."
            )

        x_std = self._standardizer.standardize(clean[column_x].to_numpy())
        y_std = self._standardizer.standardize(clean[column_y].to_numpy())

        lag_results = self._lag_computer.compute(x_std, y_std, max_lag)
        peak = self._peak_detector.detect(lag_results)

        direction = "no significant lag detected"
        if peak is not None and peak.lag != 0:
            direction = (
                f"'{column_x}' leads '{column_y}' by {peak.lag} period(s)"
                if peak.lag > 0
                else f"'{column_y}' leads '{column_x}' by {abs(peak.lag)} period(s)"
            )

        return {
            "column_x": column_x,
            "column_y": column_y,
            "max_lag": max_lag,
            "lags": [
                {
                    "lag": r.lag,
                    "correlation": round(r.correlation, 6),
                    "abs_correlation": round(r.abs_correlation, 6),
                }
                for r in lag_results
            ],
            "peak_lag": {
                "lag": peak.lag,
                "correlation": round(peak.correlation, 6),
            } if peak else None,
            "direction_interpretation": direction,
            "n_observations": len(clean),
        }
