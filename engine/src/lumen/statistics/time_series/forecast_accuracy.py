"""Forecast accuracy metrics for time series evaluation."""

# #[AI_CONTEXT_START]
# - CONFIGURACIÓN DE FACTORY: Registrar este calculator en `TimeSeriesStatisticsFactory` (backends pandas | polars | spark) y exponerlo mediante `StatisticsInyeccionDependency`, inyectada por la Factory Maestra de Agentes junto a las demás fábricas de dominio.
# - ABSTRACCIÓN DEL DATO: Mutar constructores y `analyze`/`compute` para recibir el contenedor abstracto del backend (`pd.DataFrame`, `pl.DataFrame`/`pl.LazyFrame`, `pyspark.sql.DataFrame`) inyectado por la factory; eliminar `np.ndarray`/`pd.Series` sueltos en firmas públicas.
# - REFACTOR NATIVO: Resolver métricas con expresiones 100 % nativas del backend activo (Polars: `.select`/`.group_by`/`.agg` sin `.collect()` salvo materialización acordada; PySpark: `pyspark.sql.functions` y ventanas distribuidas; Pandas: operaciones vectorizadas). No convertir a NumPy/Pandas desde backends no-pandas.
# #[AI_CONTEXT_END]
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

@dataclass(frozen=True)
class AccuracyMetric:
    """Immutable result for a single forecast accuracy metric."""

    name: str
    value: float
    unit: str
    interpretation: str

class BaseAccuracyMetric(ABC):
    """Abstract base for all forecast accuracy metrics."""

    @abstractmethod
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> AccuracyMetric:
        """Compute the accuracy metric."""

class MAEMetric(BaseAccuracyMetric):
    """Mean Absolute Error."""

    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> AccuracyMetric:
        value = float(np.mean(np.abs(y_true - y_pred)))
        return AccuracyMetric(name="mae", value=round(value, 6), unit="original units", interpretation=f"Average absolute error of {value:.4f} units.")

class RMSEMetric(BaseAccuracyMetric):
    """Root Mean Squared Error."""

    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> AccuracyMetric:
        value = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        return AccuracyMetric(name="rmse", value=round(value, 6), unit="original units", interpretation=f"RMSE of {value:.4f}. Sensitive to outlier errors.")

class MAPEMetric(BaseAccuracyMetric):
    """Mean Absolute Percentage Error."""

    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> AccuracyMetric:
        non_zero = y_true != 0
        if not np.any(non_zero):
            return AccuracyMetric(name="mape", value=float("nan"), unit="%", interpretation="MAPE undefined: all actual values are zero.")
        value = float(np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100)
        quality = "Excellent." if value < 5 else "Good." if value < 10 else "Acceptable." if value < 20 else "Poor."
        return AccuracyMetric(name="mape", value=round(value, 4), unit="%", interpretation=f"Average percentage error of {value:.2f}%. {quality}")

class MASEMetric(BaseAccuracyMetric):
    """Mean Absolute Scaled Error."""

    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> AccuracyMetric:
        mae = float(np.mean(np.abs(y_true - y_pred)))
        naive_errors = np.abs(np.diff(y_true))
        mae_naive = float(np.mean(naive_errors)) if len(naive_errors) > 0 else 0.0
        if mae_naive == 0.0:
            return AccuracyMetric(name="mase", value=float("nan"), unit="scaled", interpretation="MASE undefined: naive forecast has zero error.")
        value = mae / mae_naive
        verdict = "Model beats naive forecast." if value < 1.0 else "Naive forecast is better."
        return AccuracyMetric(name="mase", value=round(value, 6), unit="scaled", interpretation=f"MASE={value:.4f}. {verdict}")

_METRIC_REGISTRY: dict[str, BaseAccuracyMetric] = {
    "mae": MAEMetric(),
    "rmse": RMSEMetric(),
    "mape": MAPEMetric(),
    "mase": MASEMetric(),
}

class ForecastAccuracyCalculator:
    """Computes a suite of forecast accuracy metrics."""

    _MINIMUM_OBSERVATIONS: int = 2

    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray, metrics: list[str] | None = None) -> dict:
        if len(y_true) != len(y_pred):
            raise ValueError(f"y_true and y_pred must have equal length. Got {len(y_true)} and {len(y_pred)}.")
        if len(y_true) < self._MINIMUM_OBSERVATIONS:
            raise ValueError(f"At least {self._MINIMUM_OBSERVATIONS} observations required. Got {len(y_true)}.")

        active_metrics = metrics if metrics is not None else list(_METRIC_REGISTRY.keys())
        invalid = [m for m in active_metrics if m not in _METRIC_REGISTRY]
        if invalid:
            raise KeyError(f"Unknown metric(s): {invalid}. Available: {list(_METRIC_REGISTRY.keys())}")

        results = {key: _METRIC_REGISTRY[key].calculate(y_true, y_pred) for key in active_metrics}
        return {
            "metrics": {key: {"value": r.value, "unit": r.unit, "interpretation": r.interpretation} for key, r in results.items()},
            "n_observations": len(y_true),
            "metrics_computed": active_metrics,
        }
