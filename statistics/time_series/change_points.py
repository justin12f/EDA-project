"""Change point detection: CUSUM and variance shift detection."""

# #[AI_CONTEXT_START]
# - CONFIGURACIÓN DE FACTORY: Registrar este calculator en `TimeSeriesStatisticsFactory` (backends pandas | polars | spark) y exponerlo mediante `StatisticsInyeccionDependency`, inyectada por la Factory Maestra de Agentes junto a las demás fábricas de dominio.
# - ABSTRACCIÓN DEL DATO: Mutar constructores y `analyze`/`compute` para recibir el contenedor abstracto del backend (`pd.DataFrame`, `pl.DataFrame`/`pl.LazyFrame`, `pyspark.sql.DataFrame`) inyectado por la factory; eliminar `np.ndarray`/`pd.Series` sueltos en firmas públicas.
# - REFACTOR NATIVO: Resolver métricas con expresiones 100 % nativas del backend activo (Polars: `.select`/`.group_by`/`.agg` sin `.collect()` salvo materialización acordada; PySpark: `pyspark.sql.functions` y ventanas distribuidas; Pandas: operaciones vectorizadas). No convertir a NumPy/Pandas desde backends no-pandas.
# #[AI_CONTEXT_END]
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

@dataclass(frozen=True)
class ChangePoint:
    """Immutable record of a detected change point."""

    index: int
    method: str
    statistic: float
    direction: str  # 'upward', 'downward', or 'variance_shift'

class CUSUMDetector:
    """CUSUM (Cumulative Sum) change point detector."""

    _DEFAULT_K_MULTIPLIER: float = 0.5
    _DEFAULT_H_MULTIPLIER: float = 4.0

    def detect(self, series: np.ndarray, k_multiplier: float, h_multiplier: float) -> list[ChangePoint]:
        mu = float(series.mean())
        sigma = float(series.std(ddof=1))
        if sigma == 0.0:
            return []
        k = k_multiplier * sigma
        h = h_multiplier * sigma
        cusum_pos = np.zeros(len(series))
        cusum_neg = np.zeros(len(series))
        change_points: list[ChangePoint] = []
        for t in range(1, len(series)):
            cusum_pos[t] = max(0.0, cusum_pos[t - 1] + (series[t] - mu - k))
            cusum_neg[t] = max(0.0, cusum_neg[t - 1] + (mu - series[t] - k))
            if cusum_pos[t] > h:
                change_points.append(ChangePoint(index=t, method="cusum", statistic=round(cusum_pos[t], 4), direction="upward"))
                cusum_pos[t] = 0.0
            if cusum_neg[t] > h:
                change_points.append(ChangePoint(index=t, method="cusum", statistic=round(cusum_neg[t], 4), direction="downward"))
                cusum_neg[t] = 0.0
        return change_points

class VarianceShiftDetector:
    """Detects structural shifts in variance using a rolling F-ratio test."""

    _MINIMUM_SEGMENT_SIZE: int = 5

    def detect(self, series: np.ndarray, variance_ratio_threshold: float) -> list[ChangePoint]:
        change_points: list[ChangePoint] = []
        n = len(series)
        for split in range(self._MINIMUM_SEGMENT_SIZE, n - self._MINIMUM_SEGMENT_SIZE):
            left = series[:split]
            right = series[split:]
            var_left = float(np.var(left, ddof=1))
            var_right = float(np.var(right, ddof=1))
            if var_left == 0.0:
                continue
            ratio = var_right / var_left
            if ratio > variance_ratio_threshold or ratio < 1.0 / variance_ratio_threshold:
                change_points.append(ChangePoint(index=split, method="variance_shift", statistic=round(ratio, 4), direction="variance_shift"))
        return self._deduplicate(change_points, min_gap=5)

    def _deduplicate(self, points: list[ChangePoint], min_gap: int) -> list[ChangePoint]:
        if not points:
            return []
        sorted_points = sorted(points, key=lambda p: abs(p.statistic - 1.0), reverse=True)
        retained: list[ChangePoint] = []
        for candidate in sorted_points:
            too_close = any(abs(candidate.index - kept.index) < min_gap for kept in retained)
            if not too_close:
                retained.append(candidate)
        return sorted(retained, key=lambda p: p.index)

class ChangePointDetector:
    """Orchestrates CUSUM and variance shift detection."""

    _MINIMUM_OBSERVATIONS: int = 20

    def __init__(self) -> None:
        self._cusum = CUSUMDetector()
        self._variance_shift = VarianceShiftDetector()

    def calculate(
        self,
        series: np.ndarray,
        k_multiplier: float = 0.5,
        h_multiplier: float = 4.0,
        variance_ratio_threshold: float = 2.0,
    ) -> dict:
        if len(series) < self._MINIMUM_OBSERVATIONS:
            raise ValueError(f"At least {self._MINIMUM_OBSERVATIONS} observations required. Got {len(series)}.")
        if k_multiplier <= 0:
            raise ValueError(f"k_multiplier must be > 0. Got {k_multiplier}.")
        if h_multiplier <= 0:
            raise ValueError(f"h_multiplier must be > 0. Got {h_multiplier}.")
        if variance_ratio_threshold <= 1.0:
            raise ValueError(f"variance_ratio_threshold must be > 1.0. Got {variance_ratio_threshold}.")

        cusum_points = self._cusum.detect(series, k_multiplier, h_multiplier)
        variance_points = self._variance_shift.detect(series, variance_ratio_threshold)
        all_points = sorted(cusum_points + variance_points, key=lambda p: p.index)

        return {
            "cusum_change_points": [{"index": p.index, "statistic": p.statistic, "direction": p.direction} for p in cusum_points],
            "variance_change_points": [{"index": p.index, "statistic": p.statistic} for p in variance_points],
            "all_change_points": [{"index": p.index, "method": p.method, "statistic": p.statistic, "direction": p.direction} for p in all_points],
            "n_cusum_detected": len(cusum_points),
            "n_variance_detected": len(variance_points),
            "n_total": len(all_points),
            "parameters": {"k_multiplier": k_multiplier, "h_multiplier": h_multiplier, "variance_ratio_threshold": variance_ratio_threshold},
            "n": len(series),
        }
