"""Cyclical pattern detection via FFT and dominant frequency extraction."""

# #[AI_CONTEXT_START]
# - CONFIGURACIÓN DE FACTORY: Registrar este calculator en `TimeSeriesStatisticsFactory` (backends pandas | polars | spark) y exponerlo mediante `StatisticsInyeccionDependency`, inyectada por la Factory Maestra de Agentes junto a las demás fábricas de dominio.
# - ABSTRACCIÓN DEL DATO: Mutar constructores y `analyze`/`compute` para recibir el contenedor abstracto del backend (`pd.DataFrame`, `pl.DataFrame`/`pl.LazyFrame`, `pyspark.sql.DataFrame`) inyectado por la factory; eliminar `np.ndarray`/`pd.Series` sueltos en firmas públicas.
# - REFACTOR NATIVO: Resolver métricas con expresiones 100 % nativas del backend activo (Polars: `.select`/`.group_by`/`.agg` sin `.collect()` salvo materialización acordada; PySpark: `pyspark.sql.functions` y ventanas distribuidas; Pandas: operaciones vectorizadas). No convertir a NumPy/Pandas desde backends no-pandas.
# #[AI_CONTEXT_END]
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

@dataclass(frozen=True)
class DominantCycle:
    """Immutable record of a detected dominant cycle in a time series."""

    rank: int
    frequency: float
    period: float
    amplitude: float
    power: float
    power_fraction: float

class TrendRemover:
    """Removes linear trend from a series before FFT to avoid spectral leakage."""

    def remove(self, series: np.ndarray) -> np.ndarray:
        t = np.arange(len(series), dtype=float)
        x = np.column_stack([np.ones_like(t), t])
        coefficients, *_ = np.linalg.lstsq(x, series, rcond=None)
        return series - x @ coefficients

class HanningWindowApplier:
    """Applies a Hanning window to reduce spectral leakage at series boundaries."""

    def apply(self, series: np.ndarray) -> np.ndarray:
        return series * np.hanning(len(series))

class FFTPowerSpectrumCalculator:
    """Computes the one-sided power spectrum via FFT."""

    def calculate(self, series: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = len(series)
        fft_values = np.fft.rfft(series)
        power = (np.abs(fft_values) ** 2) / (n ** 2)
        power[1:-1] *= 2
        frequencies = np.fft.rfftfreq(n)
        return frequencies, power

class DominantCycleExtractor:
    """Extracts the top N dominant cycles from a power spectrum."""

    def extract(self, frequencies: np.ndarray, power: np.ndarray, top_n: int) -> list[DominantCycle]:
        valid_indices = np.where(frequencies > 0)[0]
        valid_freqs = frequencies[valid_indices]
        valid_power = power[valid_indices]
        total_power = float(valid_power.sum())
        if total_power == 0.0:
            return []
        sorted_indices = np.argsort(valid_power)[::-1]
        top_indices = sorted_indices[:top_n]
        return [
            DominantCycle(
                rank=rank + 1,
                frequency=round(float(valid_freqs[idx]), 6),
                period=round(1.0 / float(valid_freqs[idx]), 4),
                amplitude=round(float(np.sqrt(valid_power[idx])), 6),
                power=round(float(valid_power[idx]), 8),
                power_fraction=round(float(valid_power[idx]) / total_power, 6),
            )
            for rank, idx in enumerate(top_indices)
        ]

class CyclicalPatternsCalculator:
    """Detects dominant cycles in a time series using FFT spectral analysis."""

    _MINIMUM_OBSERVATIONS: int = 16

    def __init__(self) -> None:
        self._trend_remover = TrendRemover()
        self._window_applier = HanningWindowApplier()
        self._power_spectrum = FFTPowerSpectrumCalculator()
        self._cycle_extractor = DominantCycleExtractor()

    def calculate(
        self,
        series: np.ndarray,
        top_n: int = 5,
        remove_trend: bool = True,
        apply_window: bool = True,
    ) -> dict:
        if len(series) < self._MINIMUM_OBSERVATIONS:
            raise ValueError(f"At least {self._MINIMUM_OBSERVATIONS} observations required for FFT. Got {len(series)}.")
        if top_n < 1:
            raise ValueError(f"top_n must be >= 1. Got {top_n}.")

        processed = series.astype(float).copy()
        if remove_trend:
            processed = self._trend_remover.remove(processed)
        if apply_window:
            processed = self._window_applier.apply(processed)

        frequencies, power = self._power_spectrum.calculate(processed)
        cycles = self._cycle_extractor.extract(frequencies, power, top_n)

        return {
            "dominant_cycles": [
                {"rank": c.rank, "frequency": c.frequency, "period_in_samples": c.period,
                 "amplitude": c.amplitude, "power": c.power, "power_fraction": c.power_fraction}
                for c in cycles
            ],
            "top_n": top_n,
            "preprocessing": {"trend_removed": remove_trend, "hanning_window_applied": apply_window},
            "n": len(series),
            "interpretation_note": (
                "period_in_samples = 1/frequency. Multiply by your sampling interval to get "
                "real-world period (e.g., period_in_samples × 1 month = cycle duration in months)."
            ),
        }
