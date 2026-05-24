"""Event density: frequency, inter-event intervals, and temporal clustering."""

# #[AI_CONTEXT_START]
# - CONFIGURACIÓN DE FACTORY: Registrar este calculator en `SurvivalStatisticsFactory` (backends pandas | polars | spark) y exponerlo mediante `StatisticsInyeccionDependency`, inyectada por la Factory Maestra de Agentes junto a las demás fábricas de dominio.
# - ABSTRACCIÓN DEL DATO: Mutar constructores y `analyze`/`compute` para recibir el contenedor abstracto del backend (`pd.DataFrame`, `pl.DataFrame`/`pl.LazyFrame`, `pyspark.sql.DataFrame`) inyectado por la factory; eliminar `np.ndarray`/`pd.Series` sueltos en firmas públicas.
# - REFACTOR NATIVO: Resolver métricas con expresiones 100 % nativas del backend activo (Polars: `.select`/`.group_by`/`.agg` sin `.collect()` salvo materialización acordada; PySpark: `pyspark.sql.functions` y ventanas distribuidas; Pandas: operaciones vectorizadas). No convertir a NumPy/Pandas desde backends no-pandas.
# #[AI_CONTEXT_END]
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

@dataclass(frozen=True)
class IntervalStats:
    """Immutable statistics for inter-event intervals."""

    mean_interval: float
    median_interval: float
    std_interval: float
    min_interval: float
    max_interval: float
    cv_interval: float
    regularity_label: str

class IntervalExtractor:
    """Extracts sorted inter-event intervals from an event time series."""

    def extract(self, event_times: np.ndarray) -> np.ndarray:
        """Compute consecutive inter-event intervals.

        Args:
            event_times: Sorted event time array.

        Returns:
            Inter-event interval array (length = n_events - 1).

        Raises:
            ValueError: If fewer than 2 events.
        """
        if len(event_times) < 2:
            raise ValueError(
                f"At least 2 events required to compute intervals. "
                f"Got {len(event_times)}."
            )
        sorted_times = np.sort(event_times)
        return np.diff(sorted_times)

class RegularityClassifier:
    """Classifies event regularity based on coefficient of variation.

    CV of inter-event intervals:
        CV < 0.3: highly regular (e.g., scheduled maintenance)
        CV 0.3-1.0: moderately irregular
        CV > 1.0: highly irregular / bursty (e.g., failures, crises)

    For a Poisson process, CV ≈ 1.0 (exponential inter-arrivals).
    """

    _THRESHOLDS: list[tuple[float, str]] = [
        (1.0, "highly_irregular_bursty"),
        (0.3, "moderately_irregular"),
        (0.0, "highly_regular"),
    ]

    def classify(self, cv: float) -> str:
        """Classify regularity from CV.

        Args:
            cv: Coefficient of variation of intervals.

        Returns:
            Regularity label string.
        """
        for threshold, label in self._THRESHOLDS:
            if cv >= threshold:
                return label
        return "highly_regular"

class TemporalBurstinessCalculator:
    """Computes Goh-Barabási burstiness parameter B.

    B = (σ_τ - μ_τ) / (σ_τ + μ_τ)

    B ∈ [-1, 1]:
        B = -1: perfectly regular (equal intervals)
        B = 0: Poisson process
        B = 1: maximally bursty

    Reference: Goh & Barabási (2008). Europhys. Lett.
    """

    def calculate(self, intervals: np.ndarray) -> float:
        """Compute burstiness parameter.

        Args:
            intervals: Inter-event interval array.

        Returns:
            Burstiness B in [-1, 1].
        """
        sigma = float(intervals.std(ddof=1))
        mu = float(intervals.mean())
        denominator = sigma + mu
        return float((sigma - mu) / denominator) if denominator > 0 else 0.0

class RollingEventRateCalculator:
    """Computes event rate over rolling time windows."""

    def calculate(
        self,
        event_times: np.ndarray,
        window_size: float,
        n_windows: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute rolling event rate.

        Args:
            event_times: Sorted event time array.
            window_size: Window duration in same units as event_times.
            n_windows: Number of evaluation windows.

        Returns:
            Tuple (window_centers, event_rates) arrays.
        """
        t_min, t_max = float(event_times.min()), float(event_times.max())
        centers = np.linspace(t_min + window_size / 2, t_max - window_size / 2, n_windows)
        rates = np.array([
            float(np.sum(
                (event_times >= c - window_size / 2) &
                (event_times < c + window_size / 2)
            ) / window_size)
            for c in centers
        ])
        return centers, rates

class EventDensityCalculator:
    """Event frequency, inter-event statistics, burstiness, and rolling rate.

    Workflow:
        calculator = EventDensityCalculator()
        result = calculator.calculate(
            data_frame=df,
            event_time_column="failure_time",
            event_indicator_column="is_failure",   # optional, 1=event
            window_size=30.0,                      # optional
            n_rate_windows=20,                     # optional
        )
    """

    _MINIMUM_EVENTS: int = 3

    def __init__(self) -> None:
        self._interval_extractor = IntervalExtractor()
        self._regularity_classifier = RegularityClassifier()
        self._burstiness_calc = TemporalBurstinessCalculator()
        self._rolling_calc = RollingEventRateCalculator()

    def calculate(
        self,
        data_frame: pd.DataFrame,
        event_time_column: str,
        event_indicator_column: str | None = None,
        window_size: float | None = None,
        n_rate_windows: int = 20,
    ) -> dict:
        """Compute event density and temporal clustering metrics.

        Args:
            data_frame: Source DataFrame.
            event_time_column: Numeric time-of-event column.
            event_indicator_column: Binary column to filter only events (1=event).
                If None, all rows are treated as events.
            window_size: Rolling window size for rate computation.
                Auto-set to (max_time - min_time) / 10 if None.
            n_rate_windows: Number of rolling rate evaluation points.

        Returns:
            Dict with interval stats, burstiness, rolling rate, and event summary.

        Raises:
            KeyError: If columns are not found.
            ValueError: If insufficient events.
        """
        if event_time_column not in data_frame.columns:
            raise KeyError(f"Column '{event_time_column}' not found in DataFrame.")
        if event_indicator_column is not None and event_indicator_column not in data_frame.columns:
            raise KeyError(f"Column '{event_indicator_column}' not found in DataFrame.")
        if n_rate_windows < 3:
            raise ValueError(f"n_rate_windows must be >= 3. Got {n_rate_windows}.")

        df = data_frame[[event_time_column] + (
            [event_indicator_column] if event_indicator_column else []
        )].dropna()

        if event_indicator_column is not None:
            df = df[df[event_indicator_column] == 1]

        event_times = np.sort(df[event_time_column].to_numpy(dtype=float))

        if len(event_times) < self._MINIMUM_EVENTS:
            raise ValueError(
                f"At least {self._MINIMUM_EVENTS} events required. "
                f"Got {len(event_times)}."
            )

        intervals = self._interval_extractor.extract(event_times)
        mean_iv = float(intervals.mean())
        std_iv = float(intervals.std(ddof=1))
        cv = std_iv / mean_iv if mean_iv > 0 else float("inf")
        regularity = self._regularity_classifier.classify(cv)
        burstiness = self._burstiness_calc.calculate(intervals)

        t_min, t_max = float(event_times.min()), float(event_times.max())
        observation_window = t_max - t_min
        overall_rate = len(event_times) / observation_window if observation_window > 0 else float("inf")

        effective_window = window_size if window_size is not None else observation_window / 10
        rolling_centers, rolling_rates = self._rolling_calc.calculate(
            event_times, effective_window, n_rate_windows
        )

        return {
            "event_summary": {
                "n_events": len(event_times),
                "observation_window": round(observation_window, 4),
                "overall_rate_per_unit_time": round(overall_rate, 6),
                "first_event_time": round(t_min, 4),
                "last_event_time": round(t_max, 4),
            },
            "interval_statistics": {
                "mean_interval": round(mean_iv, 4),
                "median_interval": round(float(np.median(intervals)), 4),
                "std_interval": round(std_iv, 4),
                "min_interval": round(float(intervals.min()), 4),
                "max_interval": round(float(intervals.max()), 4),
                "cv_interval": round(cv, 4),
                "regularity_label": regularity,
            },
            "burstiness": {
                "burstiness_parameter": round(burstiness, 4),
                "interpretation": (
                    "Bursty pattern — events cluster in time."
                    if burstiness > 0.2
                    else "Poisson-like random pattern."
                    if abs(burstiness) <= 0.2
                    else "Regular / sub-Poisson pattern."
                ),
            },
            "rolling_event_rate": {
                "window_centers": rolling_centers.tolist(),
                "event_rates": rolling_rates.tolist(),
                "window_size": effective_window,
                "peak_rate": round(float(rolling_rates.max()), 6),
                "peak_rate_time": round(float(rolling_centers[int(np.argmax(rolling_rates))]), 4),
            },
        }