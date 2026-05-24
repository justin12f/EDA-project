"""Moving average family: SMA, EMA, WMA with crossover detection."""

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
class CrossoverEvent:
    """Immutable record of a moving average crossover event."""

    index: int
    direction: str  # 'golden_cross' or 'death_cross'
    fast_value: float
    slow_value: float

class BaseMovingAverage(ABC):
    """Abstract base for all moving average strategies."""

    @abstractmethod
    def calculate(self, series: np.ndarray, period: int) -> np.ndarray:
        """Compute moving average values."""

class SimpleMovingAverage(BaseMovingAverage):
    """SMA: arithmetic mean over a fixed rolling window."""

    def calculate(self, series: np.ndarray, period: int) -> np.ndarray:
        result = np.full(len(series), np.nan)
        for i in range(period - 1, len(series)):
            result[i] = float(np.mean(series[i - period + 1: i + 1]))
        return result

class ExponentialMovingAverage(BaseMovingAverage):
    """EMA: exponentially decaying weights, more responsive to recent data."""

    def calculate(self, series: np.ndarray, period: int) -> np.ndarray:
        result = np.full(len(series), np.nan)
        k = 2.0 / (period + 1)
        if len(series) < period:
            return result
        result[period - 1] = float(np.mean(series[:period]))
        for i in range(period, len(series)):
            result[i] = series[i] * k + result[i - 1] * (1.0 - k)
        return result

class WeightedMovingAverage(BaseMovingAverage):
    """WMA: linearly increasing weights, most weight on most recent value."""

    def calculate(self, series: np.ndarray, period: int) -> np.ndarray:
        result = np.full(len(series), np.nan)
        weights = np.arange(1, period + 1, dtype=float)
        weight_sum = weights.sum()
        for i in range(period - 1, len(series)):
            chunk = series[i - period + 1: i + 1]
            result[i] = float(np.dot(chunk, weights) / weight_sum)
        return result

_MA_REGISTRY: dict[str, BaseMovingAverage] = {
    "sma": SimpleMovingAverage(),
    "ema": ExponentialMovingAverage(),
    "wma": WeightedMovingAverage(),
}

class CrossoverDetector:
    """Detects golden crosses and death crosses between two MA lines."""

    def detect(self, fast_ma: np.ndarray, slow_ma: np.ndarray) -> list[CrossoverEvent]:
        events: list[CrossoverEvent] = []
        for i in range(1, len(fast_ma)):
            if any(np.isnan(v) for v in [fast_ma[i], slow_ma[i], fast_ma[i-1], slow_ma[i-1]]):
                continue
            prev_diff = fast_ma[i - 1] - slow_ma[i - 1]
            curr_diff = fast_ma[i] - slow_ma[i]
            if prev_diff <= 0 < curr_diff:
                events.append(CrossoverEvent(i, "golden_cross", float(fast_ma[i]), float(slow_ma[i])))
            elif prev_diff >= 0 > curr_diff:
                events.append(CrossoverEvent(i, "death_cross", float(fast_ma[i]), float(slow_ma[i])))
        return events

class MovingAveragesCalculator:
    """Computes multiple MA types with optional crossover detection."""

    _MINIMUM_OBSERVATIONS: int = 4

    def __init__(self) -> None:
        self._crossover_detector = CrossoverDetector()

    def calculate(
        self,
        series: np.ndarray,
        periods: list[int],
        ma_types: list[str] | None = None,
        detect_crossovers: bool = False,
        fast_period: int | None = None,
        slow_period: int | None = None,
        crossover_ma_type: str = "ema",
    ) -> dict:
        if not periods:
            raise ValueError("periods list cannot be empty.")
        if any(p < 2 for p in periods):
            raise ValueError("All periods must be >= 2.")
        if len(series) < self._MINIMUM_OBSERVATIONS:
            raise ValueError(f"At least {self._MINIMUM_OBSERVATIONS} observations required. Got {len(series)}.")

        active_types = ma_types if ma_types is not None else list(_MA_REGISTRY.keys())
        invalid_types = [t for t in active_types if t not in _MA_REGISTRY]
        if invalid_types:
            raise KeyError(f"Unknown MA type(s): {invalid_types}. Available: {list(_MA_REGISTRY.keys())}")

        moving_averages: dict[str, dict[str, list]] = {}
        for ma_type in active_types:
            calc = _MA_REGISTRY[ma_type]
            moving_averages[ma_type] = {
                f"period_{p}": calc.calculate(series, p).tolist()
                for p in periods if p <= len(series)
            }

        crossovers: list[dict] | None = None
        if detect_crossovers:
            if fast_period is None or slow_period is None:
                raise ValueError("fast_period and slow_period are required when detect_crossovers=True.")
            if fast_period >= slow_period:
                raise ValueError(f"fast_period ({fast_period}) must be < slow_period ({slow_period}).")
            if crossover_ma_type not in _MA_REGISTRY:
                raise KeyError(f"crossover_ma_type '{crossover_ma_type}' not found.")

            ma_calc = _MA_REGISTRY[crossover_ma_type]
            events = self._crossover_detector.detect(
                ma_calc.calculate(series, fast_period),
                ma_calc.calculate(series, slow_period),
            )
            crossovers = [
                {"index": e.index, "direction": e.direction,
                 "fast_value": round(e.fast_value, 6), "slow_value": round(e.slow_value, 6)}
                for e in events
            ]

        return {
            "moving_averages": moving_averages,
            "crossovers": crossovers,
            "periods": periods,
            "ma_types": active_types,
            "n": len(series),
        }
