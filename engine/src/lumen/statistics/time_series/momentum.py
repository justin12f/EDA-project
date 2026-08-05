"""Momentum and rate-of-change analysis for time series."""

# #[AI_CONTEXT_START]
# - CONFIGURACIÓN DE FACTORY: Registrar este calculator en `TimeSeriesStatisticsFactory` (backends pandas | polars | spark) y exponerlo mediante `StatisticsInyeccionDependency`, inyectada por la Factory Maestra de Agentes junto a las demás fábricas de dominio.
# - ABSTRACCIÓN DEL DATO: Mutar constructores y `analyze`/`compute` para recibir el contenedor abstracto del backend (`pd.DataFrame`, `pl.DataFrame`/`pl.LazyFrame`, `pyspark.sql.DataFrame`) inyectado por la factory; eliminar `np.ndarray`/`pd.Series` sueltos en firmas públicas.
# - REFACTOR NATIVO: Resolver métricas con expresiones 100 % nativas del backend activo (Polars: `.select`/`.group_by`/`.agg` sin `.collect()` salvo materialización acordada; PySpark: `pyspark.sql.functions` y ventanas distribuidas; Pandas: operaciones vectorizadas). No convertir a NumPy/Pandas desde backends no-pandas.
# #[AI_CONTEXT_END]
from __future__ import annotations

import numpy as np

class RateOfChangeCalculator:
    """Rate of Change (ROC): percentage change over n periods.

    ROC(t, n) = (X(t) - X(t-n)) / X(t-n) × 100

    Measures the speed of price/value movement. Positive ROC indicates
    upward momentum; negative indicates downward momentum.
    """

    def calculate(self, series: np.ndarray, period: int) -> np.ndarray:
        """Compute ROC for each period.

        Args:
            series: 1D numerical time series.
            period: Look-back period for the comparison.

        Returns:
            ROC array (NaN for first `period` positions).

        Raises:
            ValueError: If any base value is zero (division undefined).
        """
        result = np.full(len(series), np.nan)
        for i in range(period, len(series)):
            base = series[i - period]
            if base == 0.0:
                result[i] = np.nan
            else:
                result[i] = (series[i] - base) / abs(base) * 100.0
        return result

class AccelerationCalculator:
    """Acceleration: first difference of ROC (momentum of momentum).

    Acceleration(t) = ROC(t) - ROC(t-1)

    Positive acceleration means momentum is strengthening.
    Negative acceleration means momentum is weakening even if still positive.
    """

    def calculate(self, roc: np.ndarray) -> np.ndarray:
        """Compute acceleration from a ROC series.

        Args:
            roc: ROC array (may contain NaN).

        Returns:
            Acceleration array (NaN propagated from ROC NaNs).
        """
        result = np.full(len(roc), np.nan)
        for i in range(1, len(roc)):
            if not np.isnan(roc[i]) and not np.isnan(roc[i - 1]):
                result[i] = roc[i] - roc[i - 1]
        return result

class MomentumSignalClassifier:
    """Classifies momentum and acceleration into actionable signal states.

    Signal logic:
        - 'accelerating_up':   ROC > 0 and Acceleration > 0
        - 'decelerating_up':   ROC > 0 and Acceleration < 0
        - 'accelerating_down': ROC < 0 and Acceleration < 0
        - 'decelerating_down': ROC < 0 and Acceleration > 0
        - 'neutral':           ROC == 0 or values are NaN
    """

    def classify(
        self, roc: np.ndarray, acceleration: np.ndarray
    ) -> list[str]:
        """Classify each period into a momentum signal.

        Args:
            roc: Rate of change array.
            acceleration: Acceleration array.

        Returns:
            List of signal label strings.
        """
        signals: list[str] = []
        for r, a in zip(roc, acceleration):
            if np.isnan(r) or np.isnan(a):
                signals.append("neutral")
            elif r > 0 and a > 0:
                signals.append("accelerating_up")
            elif r > 0 and a < 0:
                signals.append("decelerating_up")
            elif r < 0 and a < 0:
                signals.append("accelerating_down")
            elif r < 0 and a > 0:
                signals.append("decelerating_down")
            else:
                signals.append("neutral")
        return signals

class MomentumCalculator:
    """Computes ROC, acceleration, and momentum signals for a time series.

    Workflow:
        calculator = MomentumCalculator()
        result = calculator.calculate(
            series=df["price"].to_numpy(),
            period=14,   # optional, classic RSI look-back
        )
    """

    _MINIMUM_OBSERVATIONS: int = 4

    def __init__(self) -> None:
        self._roc = RateOfChangeCalculator()
        self._acceleration = AccelerationCalculator()
        self._classifier = MomentumSignalClassifier()

    def calculate(
        self,
        series: np.ndarray,
        period: int = 14,
    ) -> dict:
        """Run full momentum analysis.

        Args:
            series: 1D numerical time series (no NaN).
            period: Look-back period for ROC computation.

        Returns:
            Dict with roc, acceleration, signals, and current state.

        Raises:
            ValueError: If data is insufficient or period is invalid.
        """
        if len(series) < self._MINIMUM_OBSERVATIONS:
            raise ValueError(
                f"At least {self._MINIMUM_OBSERVATIONS} observations required. "
                f"Got {len(series)}."
            )
        if period < 1:
            raise ValueError(f"period must be >= 1. Got {period}.")
        if period >= len(series):
            raise ValueError(
                f"period ({period}) must be less than series length ({len(series)})."
            )

        roc = self._roc.calculate(series, period)
        acceleration = self._acceleration.calculate(roc)
        signals = self._classifier.classify(roc, acceleration)

        valid_roc = roc[~np.isnan(roc)]

        return {
            "rate_of_change": roc.tolist(),
            "acceleration": acceleration.tolist(),
            "signals": signals,
            "summary": {
                "current_roc": float(roc[-1]) if not np.isnan(roc[-1]) else None,
                "current_acceleration": (
                    float(acceleration[-1])
                    if not np.isnan(acceleration[-1]) else None
                ),
                "current_signal": signals[-1],
                "mean_roc": float(np.mean(valid_roc)) if len(valid_roc) else None,
                "max_roc": float(np.max(valid_roc)) if len(valid_roc) else None,
                "min_roc": float(np.min(valid_roc)) if len(valid_roc) else None,
            },
            "period": period,
            "n": len(series),
        }
