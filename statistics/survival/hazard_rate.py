"""Hazard rate estimation: Nelson-Aalen cumulative hazard and smoothed hazard."""

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
class HazardTimePoint:
    """Immutable hazard rate record at a single time point."""

    time: float
    n_at_risk: int
    n_events: int
    hazard_increment: float
    cumulative_hazard: float
    survival_from_hazard: float

class NelsonAalenEstimator:
    """Nelson-Aalen non-parametric cumulative hazard estimator.

    H(t) = Σ_{tᵢ≤t} dᵢ/nᵢ

    where dᵢ = events at tᵢ, nᵢ = at-risk count.
    More stable than KM in small samples near S(t) = 0.
    Survival estimated as: S(t) = exp(-H(t))
    """

    def estimate(
        self,
        times: np.ndarray,
        events: np.ndarray,
    ) -> list[HazardTimePoint]:
        """Compute Nelson-Aalen estimates.

        Args:
            times: Time array.
            events: Binary event indicator.

        Returns:
            List of HazardTimePoint at each unique event time.
        """
        n = len(times)
        unique_event_times = np.sort(np.unique(times[events == 1]))

        cumulative_hazard = 0.0
        at_risk = n
        time_cursor = 0.0
        result: list[HazardTimePoint] = []

        for t in unique_event_times:
            censored_before = int(
                np.sum((times < t) & (events == 0) & (times >= time_cursor))
            )
            events_before = int(
                np.sum((times < t) & (events == 1) & (times >= time_cursor))
            )
            at_risk -= censored_before + events_before

            n_events = int(np.sum((times == t) & (events == 1)))

            hazard_increment = n_events / at_risk if at_risk > 0 else 0.0
            cumulative_hazard += hazard_increment
            survival = float(np.exp(-cumulative_hazard))

            result.append(
                HazardTimePoint(
                    time=float(t),
                    n_at_risk=int(at_risk),
                    n_events=n_events,
                    hazard_increment=round(hazard_increment, 6),
                    cumulative_hazard=round(cumulative_hazard, 6),
                    survival_from_hazard=round(survival, 6),
                )
            )

            time_cursor = float(t)

        return result

class KernelHazardSmoother:
    """Smooths the hazard function using a Gaussian kernel.

    The raw hazard increments at event times are noisy — smoothing
    provides a more interpretable hazard rate function h(t).
    Bandwidth is selected via Silverman's rule of thumb.
    """

    def smooth(
        self,
        event_times: np.ndarray,
        hazard_increments: np.ndarray,
        n_eval_points: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply Gaussian kernel smoothing to hazard increments.

        Args:
            event_times: Times at which events occurred.
            hazard_increments: Raw hazard increments at each event time.
            n_eval_points: Number of evaluation points for the smooth curve.

        Returns:
            Tuple (eval_times, smoothed_hazard) arrays.
        """
        if len(event_times) < 2:
            return event_times, hazard_increments

        # Silverman bandwidth
        std = float(np.std(event_times, ddof=1))
        n = len(event_times)
        bandwidth = 1.06 * std * (n ** (-1 / 5))
        if bandwidth == 0:
            bandwidth = 1.0

        t_min, t_max = float(event_times.min()), float(event_times.max())
        eval_times = np.linspace(t_min, t_max, n_eval_points)
        smoothed = np.zeros(n_eval_points)

        for i, t_eval in enumerate(eval_times):
            weights = np.exp(-0.5 * ((event_times - t_eval) / bandwidth) ** 2)
            weights /= weights.sum() if weights.sum() > 0 else 1.0
            smoothed[i] = float(np.dot(weights, hazard_increments))

        return eval_times, smoothed

class HazardRateCalculator:
    """Nelson-Aalen cumulative hazard with smoothed instantaneous hazard.

    Workflow:
        calculator = HazardRateCalculator()
        result = calculator.calculate(
            data_frame=df,
            time_column="survival_days",
            event_column="event_occurred",
            n_smooth_points=100,     # optional
        )
    """

    _MINIMUM_OBSERVATIONS: int = 5

    def __init__(self) -> None:
        self._estimator = NelsonAalenEstimator()
        self._smoother = KernelHazardSmoother()

    def calculate(
        self,
        data_frame: pd.DataFrame,
        time_column: str,
        event_column: str,
        n_smooth_points: int = 100,
    ) -> dict:
        """Compute cumulative and smoothed hazard rates.

        Args:
            data_frame: Source DataFrame.
            time_column: Time-to-event or censoring column.
            event_column: Binary event indicator.
            n_smooth_points: Number of points for smoothed hazard curve.

        Returns:
            Dict with Nelson-Aalen curve, smoothed hazard, and hazard summary.

        Raises:
            KeyError: If columns are not found.
            ValueError: If data is insufficient.
        """
        for col in (time_column, event_column):
            if col not in data_frame.columns:
                raise KeyError(f"Column '{col}' not found in DataFrame.")
        if n_smooth_points < 5:
            raise ValueError(f"n_smooth_points must be >= 5. Got {n_smooth_points}.")

        clean = data_frame[[time_column, event_column]].dropna()
        clean = clean[clean[time_column] >= 0]

        if len(clean) < self._MINIMUM_OBSERVATIONS:
            raise ValueError(
                f"At least {self._MINIMUM_OBSERVATIONS} observations required. "
                f"Got {len(clean)}."
            )

        times = clean[time_column].to_numpy(dtype=float)
        events = clean[event_column].to_numpy(dtype=float)
        points = self._estimator.estimate(times, events)

        if not points:
            raise ValueError("No event times found. Check event_column has at least one event.")

        event_times = np.array([p.time for p in points])
        hazard_increments = np.array([p.hazard_increment for p in points])
        smooth_times, smooth_hazard = self._smoother.smooth(
            event_times, hazard_increments, n_smooth_points
        )

        peak_idx = int(np.argmax(smooth_hazard))

        return {
            "nelson_aalen_curve": [
                {
                    "time": p.time,
                    "n_at_risk": p.n_at_risk,
                    "n_events": p.n_events,
                    "hazard_increment": p.hazard_increment,
                    "cumulative_hazard": p.cumulative_hazard,
                    "survival_from_hazard": p.survival_from_hazard,
                }
                for p in points
            ],
            "smoothed_hazard": {
                "times": smooth_times.tolist(),
                "hazard_rates": smooth_hazard.tolist(),
                "peak_time": round(float(smooth_times[peak_idx]), 4),
                "peak_hazard": round(float(smooth_hazard[peak_idx]), 6),
            },
            "summary": {
                "max_cumulative_hazard": round(float(points[-1].cumulative_hazard), 4),
                "mean_hazard_increment": round(float(hazard_increments.mean()), 6),
                "n_event_times": len(points),
                "n_observations": len(clean),
                "n_events": int(events.sum()),
            },
        }