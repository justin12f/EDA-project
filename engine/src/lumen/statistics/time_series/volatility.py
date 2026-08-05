"""Volatility analysis for time series data."""

# #[AI_CONTEXT_START]
# - CONFIGURACIÓN DE FACTORY: Registrar este calculator en `TimeSeriesStatisticsFactory` (backends pandas | polars | spark) y exponerlo mediante `StatisticsInyeccionDependency`, inyectada por la Factory Maestra de Agentes junto a las demás fábricas de dominio.
# - ABSTRACCIÓN DEL DATO: Mutar constructores y `analyze`/`compute` para recibir el contenedor abstracto del backend (`pd.DataFrame`, `pl.DataFrame`/`pl.LazyFrame`, `pyspark.sql.DataFrame`) inyectado por la factory; eliminar `np.ndarray`/`pd.Series` sueltos en firmas públicas.
# - REFACTOR NATIVO: Resolver métricas con expresiones 100 % nativas del backend activo (Polars: `.select`/`.group_by`/`.agg` sin `.collect()` salvo materialización acordada; PySpark: `pyspark.sql.functions` y ventanas distribuidas; Pandas: operaciones vectorizadas). No convertir a NumPy/Pandas desde backends no-pandas.
# #[AI_CONTEXT_END]
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

@dataclass(frozen=True)
class VolatilityWindow:
    """Immutable volatility metrics for a single rolling window position."""

    index: int
    rolling_std: float
    ewma_volatility: float
    coefficient_of_variation: float

class RollingStdCalculator:
    """Computes rolling standard deviation over a fixed window.

    Rolling std measures local dispersion — how much the series fluctuates
    within each window, making regime changes in volatility visible.
    """

    def calculate(self, series: np.ndarray, window: int) -> np.ndarray:
        """Compute rolling standard deviation.

        Args:
            series: 1D numerical time series array.
            window: Rolling window size in periods.

        Returns:
            Array of rolling std values (NaN for the first window-1 positions).
        """
        result = np.full(len(series), np.nan)
        for i in range(window - 1, len(series)):
            result[i] = float(np.std(series[i - window + 1: i + 1], ddof=1))
        return result

class EWMAVolatilityCalculator:
    """Exponentially Weighted Moving Average volatility (RiskMetrics model).

    More responsive to recent observations than rolling std. The decay
    factor λ (lambda) controls the weight of past observations:
        σ²ₜ = λ·σ²ₜ₋₁ + (1-λ)·rₜ²
    where rₜ = xₜ - xₜ₋₁ is the period return.
    """

    _DEFAULT_LAMBDA: float = 0.94  # RiskMetrics standard for daily data

    def calculate(
        self, series: np.ndarray, decay_factor: float
    ) -> np.ndarray:
        """Compute EWMA volatility.

        Args:
            series: 1D numerical time series array.
            decay_factor: Lambda in (0, 1). Higher = slower decay (longer memory).

        Returns:
            EWMA volatility array (NaN at index 0).

        Raises:
            ValueError: If decay_factor is not in (0, 1).
        """
        if not 0.0 < decay_factor < 1.0:
            raise ValueError(
                f"decay_factor (lambda) must be in (0, 1). Got {decay_factor}."
            )

        returns = np.diff(series).astype(float)
        ewma_var = np.full(len(series), np.nan)

        if len(returns) == 0:
            return ewma_var

        ewma_var[1] = returns[0] ** 2

        for t in range(2, len(series)):
            ewma_var[t] = (
                decay_factor * ewma_var[t - 1]
                + (1 - decay_factor) * returns[t - 1] ** 2
            )

        return np.sqrt(np.where(np.isnan(ewma_var), np.nan, ewma_var))

class CoefficientOfVariationCalculator:
    """Rolling coefficient of variation: CV = rolling_std / |rolling_mean|.

    Scale-independent volatility measure, enabling comparison across
    series with different units or magnitudes.
    """

    def calculate(
        self, series: np.ndarray, window: int
    ) -> np.ndarray:
        """Compute rolling CV.

        Args:
            series: 1D numerical time series array.
            window: Rolling window size in periods.

        Returns:
            Rolling CV array (NaN where mean is zero or window not full).
        """
        result = np.full(len(series), np.nan)
        for i in range(window - 1, len(series)):
            chunk = series[i - window + 1: i + 1]
            mean = float(np.mean(chunk))
            std = float(np.std(chunk, ddof=1))
            result[i] = std / abs(mean) if mean != 0.0 else np.nan
        return result

class VolatilityRegimeDetector:
    """Classifies each period into a volatility regime based on rolling std.

    Thresholds are derived from the distribution of the rolling std itself:
        - Low:    rolling_std < Q33
        - Medium: Q33 <= rolling_std < Q67
        - High:   rolling_std >= Q67
    """

    def detect(self, rolling_std: np.ndarray) -> list[str]:
        """Assign regime labels to each period.

        Args:
            rolling_std: Rolling std array (may contain NaN).

        Returns:
            List of regime strings: 'low', 'medium', 'high', or 'unknown'.
        """
        valid = rolling_std[~np.isnan(rolling_std)]
        if len(valid) == 0:
            return ["unknown"] * len(rolling_std)

        q33 = float(np.percentile(valid, 33))
        q67 = float(np.percentile(valid, 67))

        def classify(v: float) -> str:
            if np.isnan(v):
                return "unknown"
            if v < q33:
                return "low"
            if v < q67:
                return "medium"
            return "high"

        return [classify(v) for v in rolling_std]

class VolatilityCalculator:
    """Orchestrates rolling std, EWMA, CV, and regime detection.

    Workflow:
        calculator = VolatilityCalculator()
        result = calculator.calculate(
            series=df["price"].to_numpy(),
            window=20,
            decay_factor=0.94,   # optional, EWMA lambda
        )
    """

    _MINIMUM_OBSERVATIONS: int = 4

    def __init__(self) -> None:
        self._rolling_std = RollingStdCalculator()
        self._ewma = EWMAVolatilityCalculator()
        self._cv = CoefficientOfVariationCalculator()
        self._regime_detector = VolatilityRegimeDetector()

    def calculate(
        self,
        series: np.ndarray,
        window: int = 20,
        decay_factor: float = 0.94,
    ) -> dict:
        """Run full volatility analysis.

        Args:
            series: 1D numerical time series (clean, no NaN).
            window: Rolling window in periods.
            decay_factor: EWMA lambda decay factor.

        Returns:
            Dict with rolling_std, ewma_volatility, cv, regimes, and summary.

        Raises:
            ValueError: If series is too short or window exceeds series length.
        """
        if len(series) < self._MINIMUM_OBSERVATIONS:
            raise ValueError(
                f"At least {self._MINIMUM_OBSERVATIONS} observations required. "
                f"Got {len(series)}."
            )
        if window < 2:
            raise ValueError(f"window must be >= 2. Got {window}.")
        if window > len(series):
            raise ValueError(
                f"window ({window}) cannot exceed series length ({len(series)})."
            )

        rolling_std = self._rolling_std.calculate(series, window)
        ewma = self._ewma.calculate(series, decay_factor)
        cv = self._cv.calculate(series, window)
        regimes = self._regime_detector.detect(rolling_std)

        valid_std = rolling_std[~np.isnan(rolling_std)]

        return {
            "rolling_std": rolling_std.tolist(),
            "ewma_volatility": ewma.tolist(),
            "coefficient_of_variation": cv.tolist(),
            "volatility_regimes": regimes,
            "summary": {
                "mean_volatility": float(np.mean(valid_std)) if len(valid_std) else None,
                "max_volatility": float(np.max(valid_std)) if len(valid_std) else None,
                "min_volatility": float(np.min(valid_std)) if len(valid_std) else None,
                "current_regime": regimes[-1],
            },
            "window": window,
            "decay_factor": decay_factor,
            "n": len(series),
        }
