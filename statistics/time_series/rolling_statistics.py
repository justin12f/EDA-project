"""Configurable rolling statistics over a sliding window."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseRollingStatistic(ABC):
    """Abstract base for all rolling statistics."""

    @abstractmethod
    def compute(self, series: np.ndarray, window: int) -> np.ndarray:
        """Compute rolling statistic."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the statistic name."""


class RollingMean(BaseRollingStatistic):
    @property
    def name(self) -> str:
        return "rolling_mean"

    def compute(self, series: np.ndarray, window: int) -> np.ndarray:
        result = np.full(len(series), np.nan)
        for i in range(window - 1, len(series)):
            result[i] = float(np.mean(series[i - window + 1: i + 1]))
        return result


class RollingStd(BaseRollingStatistic):
    @property
    def name(self) -> str:
        return "rolling_std"

    def compute(self, series: np.ndarray, window: int) -> np.ndarray:
        result = np.full(len(series), np.nan)
        for i in range(window - 1, len(series)):
            result[i] = float(np.std(series[i - window + 1: i + 1], ddof=1))
        return result


class RollingMin(BaseRollingStatistic):
    @property
    def name(self) -> str:
        return "rolling_min"

    def compute(self, series: np.ndarray, window: int) -> np.ndarray:
        result = np.full(len(series), np.nan)
        for i in range(window - 1, len(series)):
            result[i] = float(np.min(series[i - window + 1: i + 1]))
        return result


class RollingMax(BaseRollingStatistic):
    @property
    def name(self) -> str:
        return "rolling_max"

    def compute(self, series: np.ndarray, window: int) -> np.ndarray:
        result = np.full(len(series), np.nan)
        for i in range(window - 1, len(series)):
            result[i] = float(np.max(series[i - window + 1: i + 1]))
        return result


class RollingMedian(BaseRollingStatistic):
    @property
    def name(self) -> str:
        return "rolling_median"

    def compute(self, series: np.ndarray, window: int) -> np.ndarray:
        result = np.full(len(series), np.nan)
        for i in range(window - 1, len(series)):
            result[i] = float(np.median(series[i - window + 1: i + 1]))
        return result


class RollingSkewness(BaseRollingStatistic):
    @property
    def name(self) -> str:
        return "rolling_skewness"

    def compute(self, series: np.ndarray, window: int) -> np.ndarray:
        from scipy import stats as scipy_stats
        result = np.full(len(series), np.nan)
        for i in range(window - 1, len(series)):
            result[i] = float(scipy_stats.skew(series[i - window + 1: i + 1]))
        return result


_ROLLING_REGISTRY: dict[str, BaseRollingStatistic] = {
    "mean":     RollingMean(),
    "std":      RollingStd(),
    "min":      RollingMin(),
    "max":      RollingMax(),
    "median":   RollingMedian(),
    "skewness": RollingSkewness(),
}


class RollingStatisticsCalculator:
    """Computes configurable rolling statistics over a sliding window."""

    _MINIMUM_OBSERVATIONS: int = 4

    def calculate(
        self,
        series: np.ndarray,
        window: int = 20,
        statistics: list[str] | None = None,
    ) -> dict:
        if len(series) < self._MINIMUM_OBSERVATIONS:
            raise ValueError(f"At least {self._MINIMUM_OBSERVATIONS} observations required. Got {len(series)}.")
        if window < 2:
            raise ValueError(f"window must be >= 2. Got {window}.")
        if window > len(series):
            raise ValueError(f"window ({window}) cannot exceed series length ({len(series)}).")

        active_stats = statistics if statistics is not None else list(_ROLLING_REGISTRY.keys())
        invalid = [s for s in active_stats if s not in _ROLLING_REGISTRY]
        if invalid:
            raise KeyError(f"Unknown statistic(s): {invalid}. Available: {list(_ROLLING_REGISTRY.keys())}")

        computed = {key: _ROLLING_REGISTRY[key].compute(series, window).tolist() for key in active_stats}
        current_snapshot = {key: values[-1] for key, values in computed.items() if not np.isnan(values[-1])}

        return {
            "statistics": computed,
            "current_window_snapshot": current_snapshot,
            "window": window,
            "statistics_computed": active_stats,
            "n": len(series),
        }
