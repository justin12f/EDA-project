"""Granger causality test for time series predictive relationships."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class GrangerLagResult:
    """Immutable result for a single lag in a Granger causality test."""

    lag: int
    f_statistic: float
    p_value: float
    reject_null: bool


class LaggedDesignMatrixBuilder:
    """Builds a lagged regression design matrix for Granger causality.

    Constructs [y_lagged_1, ..., y_lagged_p, x_lagged_1, ..., x_lagged_p]
    as the unrestricted model regressors, and [y_lagged_1, ..., y_lagged_p]
    as the restricted model regressors.
    """

    def build_restricted(
        self, y: np.ndarray, max_lag: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build restricted model matrices (only y lags).

        Args:
            y: Target time series array.
            max_lag: Number of lagged periods to include.

        Returns:
            Tuple (y_target, X_restricted) trimmed to valid observations.
        """
        n = len(y)
        y_target = y[max_lag:]
        x_cols = [y[max_lag - lag: n - lag] for lag in range(1, max_lag + 1)]
        x_matrix = np.column_stack([np.ones(len(y_target))] + x_cols)
        return y_target, x_matrix

    def build_unrestricted(
        self, y: np.ndarray, x: np.ndarray, max_lag: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build unrestricted model matrices (y lags + x lags).

        Args:
            y: Target time series array.
            x: Potential causal series array.
            max_lag: Number of lagged periods to include.

        Returns:
            Tuple (y_target, X_unrestricted) trimmed to valid observations.
        """
        n = len(y)
        y_target = y[max_lag:]
        y_lag_cols = [y[max_lag - lag: n - lag] for lag in range(1, max_lag + 1)]
        x_lag_cols = [x[max_lag - lag: n - lag] for lag in range(1, max_lag + 1)]
        x_matrix = np.column_stack(
            [np.ones(len(y_target))] + y_lag_cols + x_lag_cols
        )
        return y_target, x_matrix


class OLSResidualCalculator:
    """Computes OLS residual sum of squares via the normal equations."""

    def residual_ss(self, y: np.ndarray, x: np.ndarray) -> float:
        """Compute RSS for the OLS regression of y on x.

        Args:
            y: Target array.
            x: Design matrix (includes intercept column).

        Returns:
            Residual sum of squares.
        """
        coefficients, *_ = np.linalg.lstsq(x, y, rcond=None)
        residuals = y - x @ coefficients
        return float(np.sum(residuals ** 2))


class GrangerFStatisticCalculator:
    """Computes F-statistic comparing restricted and unrestricted OLS models."""

    def calculate(
        self,
        rss_restricted: float,
        rss_unrestricted: float,
        n_observations: int,
        lag: int,
    ) -> tuple[float, float]:
        """Compute F-statistic and p-value for a Granger test.

        F = [(RSS_R - RSS_U) / lag] / [RSS_U / (n - 2*lag - 1)]

        Args:
            rss_restricted: RSS of restricted model (y lags only).
            rss_unrestricted: RSS of unrestricted model (y + x lags).
            n_observations: Effective number of observations.
            lag: Number of lags tested.

        Returns:
            Tuple (f_statistic, p_value).
        """
        df_numerator = lag
        df_denominator = n_observations - 2 * lag - 1

        if df_denominator <= 0 or rss_unrestricted == 0.0:
            return 0.0, 1.0

        f_stat = (
            (rss_restricted - rss_unrestricted) / df_numerator
        ) / (rss_unrestricted / df_denominator)

        p_value = float(
            1.0 - stats.f.cdf(f_stat, df_numerator, df_denominator)
        )

        return float(f_stat), p_value


class GrangerCausalityCalculator:
    """Tests whether x Granger-causes y across multiple lags.

    'x Granger-causes y' means that past values of x contain information
    that improves forecasts of y beyond what y's own past provides.
    This is a predictive, not mechanistic, definition of causality.

    Workflow:
        calculator = GrangerCausalityCalculator()
        result = calculator.calculate(
            data_frame=df,
            column_y="sales",
            column_x="advertising",
            max_lag=4,
            significance_level=0.05,
        )
    """

    _MINIMUM_OBSERVATIONS: int = 20

    def __init__(self) -> None:
        self._matrix_builder = LaggedDesignMatrixBuilder()
        self._rss_calculator = OLSResidualCalculator()
        self._f_calculator = GrangerFStatisticCalculator()

    def calculate(
        self,
        data_frame: pd.DataFrame,
        column_y: str,
        column_x: str,
        max_lag: int = 4,
        significance_level: float = 0.05,
    ) -> dict:
        """Run Granger causality test for lags 1 through max_lag.

        Args:
            data_frame: Source DataFrame with time-ordered rows.
            column_y: Dependent (outcome) time series.
            column_x: Potential causal series to test.
            max_lag: Maximum number of lags to test.
            significance_level: Alpha threshold.

        Returns:
            Dict with per-lag results and overall causality verdict.

        Raises:
            KeyError: If columns are not found.
            ValueError: If data is insufficient or max_lag is invalid.
        """
        for col in (column_y, column_x):
            if col not in data_frame.columns:
                raise KeyError(f"Column '{col}' not found in DataFrame.")

        if max_lag < 1:
            raise ValueError(f"max_lag must be >= 1. Got {max_lag}.")

        clean = data_frame[[column_y, column_x]].dropna()

        if len(clean) < self._MINIMUM_OBSERVATIONS:
            raise ValueError(
                f"At least {self._MINIMUM_OBSERVATIONS} observations required. "
                f"Got {len(clean)}."
            )

        y = clean[column_y].to_numpy(dtype=float)
        x = clean[column_x].to_numpy(dtype=float)

        lag_results: list[GrangerLagResult] = []

        for lag in range(1, max_lag + 1):
            y_target_r, x_restricted = self._matrix_builder.build_restricted(y, lag)
            y_target_u, x_unrestricted = self._matrix_builder.build_unrestricted(y, x, lag)

            rss_r = self._rss_calculator.residual_ss(y_target_r, x_restricted)
            rss_u = self._rss_calculator.residual_ss(y_target_u, x_unrestricted)

            f_stat, p_value = self._f_calculator.calculate(
                rss_r, rss_u, len(y_target_u), lag
            )

            lag_results.append(
                GrangerLagResult(
                    lag=lag,
                    f_statistic=round(f_stat, 4),
                    p_value=round(p_value, 6),
                    reject_null=p_value < significance_level,
                )
            )

        any_significant = any(r.reject_null for r in lag_results)

        return {
            "column_y": column_y,
            "column_x": column_x,
            "null_hypothesis": f"'{column_x}' does NOT Granger-cause '{column_y}'",
            "lag_results": [
                {
                    "lag": r.lag,
                    "f_statistic": r.f_statistic,
                    "p_value": r.p_value,
                    "reject_null": r.reject_null,
                }
                for r in lag_results
            ],
            "granger_causality_detected": any_significant,
            "significance_level": significance_level,
            "interpretation": (
                f"'{column_x}' Granger-causes '{column_y}' at one or more lags."
                if any_significant
                else f"No evidence that '{column_x}' Granger-causes '{column_y}'."
            ),
            "n_observations": len(clean),
            "max_lag": max_lag,
        }
