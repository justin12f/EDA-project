"""ACF, PACF computation and lag feature generation."""

from __future__ import annotations

import numpy as np
import pandas as pd


class AutocovarianceCalculator:
    """Computes sample autocovariance at a specific lag k."""

    def calculate(self, series: np.ndarray, lag: int) -> float:
        n = len(series)
        mean = series.mean()
        centered = series - mean
        if lag == 0:
            return float(np.mean(centered ** 2))
        return float(np.mean(centered[:n - lag] * centered[lag:]))


class ACFCalculator:
    """Autocorrelation Function (ACF) up to max_lag."""

    def calculate(self, series: np.ndarray, max_lag: int, significance_level: float) -> dict:
        gamma_0 = AutocovarianceCalculator().calculate(series, 0)
        auto_calc = AutocovarianceCalculator()
        acf_values = [
            auto_calc.calculate(series, k) / gamma_0 if gamma_0 != 0 else 0.0
            for k in range(max_lag + 1)
        ]
        n = len(series)
        z = float(_z_score_for_alpha(significance_level))
        confidence_band = z / np.sqrt(n)
        significant_lags = [k for k, v in enumerate(acf_values) if k > 0 and abs(v) > confidence_band]
        return {
            "values": [round(v, 6) for v in acf_values],
            "lags": list(range(max_lag + 1)),
            "confidence_band": round(float(confidence_band), 6),
            "significant_lags": significant_lags,
        }


class PACFCalculator:
    """Partial Autocorrelation Function (PACF) via Yule-Walker equations."""

    def calculate(self, series: np.ndarray, max_lag: int, significance_level: float) -> dict:
        auto_calc = AutocovarianceCalculator()
        gamma = [auto_calc.calculate(series, k) for k in range(max_lag + 1)]
        pacf_values = [1.0]
        for k in range(1, max_lag + 1):
            r_vec = np.array([gamma[j] for j in range(1, k + 1)])
            R_mat = np.array([[gamma[abs(i - j)] for j in range(k)] for i in range(k)])
            if np.linalg.matrix_rank(R_mat) < k:
                pacf_values.append(0.0)
                continue
            phi = np.linalg.solve(R_mat, r_vec)
            pacf_values.append(float(phi[-1]))
        n = len(series)
        z = float(_z_score_for_alpha(significance_level))
        confidence_band = z / np.sqrt(n)
        significant_lags = [k for k, v in enumerate(pacf_values) if k > 0 and abs(v) > confidence_band]
        return {
            "values": [round(v, 6) for v in pacf_values],
            "lags": list(range(max_lag + 1)),
            "confidence_band": round(float(confidence_band), 6),
            "significant_lags": significant_lags,
        }


class LagFeatureBuilder:
    """Builds a DataFrame of lagged features from a time series."""

    def build(self, series: pd.Series, lags: list[int]) -> pd.DataFrame:
        invalid = [lag for lag in lags if lag < 1]
        if invalid:
            raise ValueError(f"All lag values must be >= 1. Got: {invalid}")
        base_name = series.name or "value"
        lag_dict = {f"{base_name}_lag_{lag}": series.shift(lag) for lag in lags}
        return pd.DataFrame(lag_dict)


def _z_score_for_alpha(significance_level: float) -> float:
    from scipy import stats as scipy_stats
    return float(scipy_stats.norm.ppf(1 - significance_level / 2))


class LagFeaturesCalculator:
    """ACF, PACF analysis and lag feature generation for a time series."""

    _MINIMUM_OBSERVATIONS: int = 15

    def __init__(self) -> None:
        self._acf = ACFCalculator()
        self._pacf = PACFCalculator()
        self._lag_builder = LagFeatureBuilder()

    def calculate(
        self,
        series: pd.Series,
        max_lag: int = 20,
        lags_to_generate: list[int] | None = None,
        significance_level: float = 0.05,
    ) -> dict:
        arr = series.dropna().to_numpy(dtype=float)
        if len(arr) < self._MINIMUM_OBSERVATIONS:
            raise ValueError(f"At least {self._MINIMUM_OBSERVATIONS} observations required. Got {len(arr)}.")
        if max_lag < 1:
            raise ValueError(f"max_lag must be >= 1. Got {max_lag}.")
        if max_lag >= len(arr) // 2:
            raise ValueError(f"max_lag ({max_lag}) must be < half the series length ({len(arr) // 2}).")

        acf_result = self._acf.calculate(arr, max_lag, significance_level)
        pacf_result = self._pacf.calculate(arr, max_lag, significance_level)
        target_lags = (
            lags_to_generate if lags_to_generate is not None
            else acf_result["significant_lags"][:10]
        )
        lag_features_df = self._lag_builder.build(series, target_lags) if target_lags else pd.DataFrame()
        suggested_ma_order = max(acf_result["significant_lags"]) if acf_result["significant_lags"] else 0
        suggested_ar_order = max(pacf_result["significant_lags"]) if pacf_result["significant_lags"] else 0

        return {
            "acf": acf_result,
            "pacf": pacf_result,
            "lag_features": lag_features_df.to_dict(orient="list"),
            "lags_generated": target_lags,
            "model_order_hints": {
                "suggested_ar_order": suggested_ar_order,
                "suggested_ma_order": suggested_ma_order,
                "note": "PACF cutoff → AR(p) order. ACF cutoff → MA(q) order. Both decaying → ARMA model.",
            },
            "n": len(arr),
            "max_lag": max_lag,
        }
