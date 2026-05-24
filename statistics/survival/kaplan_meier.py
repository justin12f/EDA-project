"""Kaplan-Meier survival curve estimation."""

# #[AI_CONTEXT_START]
# - CONFIGURACIÓN DE FACTORY: Registrar este calculator en `SurvivalStatisticsFactory` (backends pandas | polars | spark) y exponerlo mediante `StatisticsInyeccionDependency`, inyectada por la Factory Maestra de Agentes junto a las demás fábricas de dominio.
# - ABSTRACCIÓN DEL DATO: Mutar constructores y `analyze`/`compute` para recibir el contenedor abstracto del backend (`pd.DataFrame`, `pl.DataFrame`/`pl.LazyFrame`, `pyspark.sql.DataFrame`) inyectado por la factory; eliminar `np.ndarray`/`pd.Series` sueltos en firmas públicas.
# - REFACTOR NATIVO: Resolver métricas con expresiones 100 % nativas del backend activo (Polars: `.select`/`.group_by`/`.agg` sin `.collect()` salvo materialización acordada; PySpark: `pyspark.sql.functions` y ventanas distribuidas; Pandas: operaciones vectorizadas). No convertir a NumPy/Pandas desde backends no-pandas.
# #[AI_CONTEXT_END]
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

@dataclass(frozen=True)
class KMTimePoint:
    """Immutable Kaplan-Meier estimate at a single time point."""

    time: float
    n_at_risk: int
    n_events: int
    n_censored: int
    survival_probability: float
    ci_lower: float
    ci_upper: float
    cumulative_hazard: float

class KaplanMeierEstimator:
    """Core KM product-limit estimator with Greenwood CI.

    S(t) = Π_{tᵢ≤t} (1 - dᵢ/nᵢ)

    where dᵢ = events at time tᵢ, nᵢ = at-risk count.
    Greenwood variance: Var(log(-log(S(t)))) = Σ dᵢ/(nᵢ(nᵢ-dᵢ))
    """

    def estimate(
        self,
        times: np.ndarray,
        events: np.ndarray,
        confidence_level: float,
    ) -> list[KMTimePoint]:
        """Compute KM estimates at each unique event time.

        Args:
            times: Time-to-event or time-to-censoring array.
            events: Binary event indicator (1=event, 0=censored).
            confidence_level: Confidence level for CI (e.g., 0.95).

        Returns:
            List of KMTimePoint at each unique event time.
        """
        n = len(times)
        unique_event_times = np.sort(np.unique(times[events == 1]))

        survival = 1.0
        greenwood_sum = 0.0
        at_risk = n
        z = float(stats.norm.ppf(1 - (1 - confidence_level) / 2))

        result: list[KMTimePoint] = []

        # Count censored before first event time as initial state
        time_cursor = 0.0

        for t in unique_event_times:
            # Update at-risk: subtract events and censorings from prior step
            censored_before = int(np.sum((times < t) & (events == 0) & (times >= time_cursor)))
            events_before = int(np.sum((times < t) & (events == 1) & (times >= time_cursor)))
            at_risk -= censored_before + events_before

            n_events = int(np.sum((times == t) & (events == 1)))
            n_censored = int(np.sum((times == t) & (events == 0)))

            if at_risk > 0 and n_events > 0:
                survival *= (1.0 - n_events / at_risk)
                if at_risk > n_events:
                    greenwood_sum += n_events / (at_risk * (at_risk - n_events))

            cum_hazard = -np.log(survival) if survival > 0 else float("inf")

            # Greenwood CI on log(-log(S)) scale (complementary log-log)
            if survival > 0 and survival < 1 and greenwood_sum > 0:
                log_log_s = np.log(-np.log(survival))
                se_log_log = np.sqrt(greenwood_sum / (np.log(survival) ** 2))
                ci_lower = float(np.exp(-np.exp(log_log_s + z * se_log_log)))
                ci_upper = float(np.exp(-np.exp(log_log_s - z * se_log_log)))
                ci_lower, ci_upper = max(0.0, ci_lower), min(1.0, ci_upper)
            else:
                ci_lower = ci_upper = float(survival)

            result.append(
                KMTimePoint(
                    time=float(t),
                    n_at_risk=int(at_risk),
                    n_events=n_events,
                    n_censored=n_censored,
                    survival_probability=round(float(survival), 6),
                    ci_lower=round(ci_lower, 6),
                    ci_upper=round(ci_upper, 6),
                    cumulative_hazard=round(cum_hazard, 6),
                )
            )

            time_cursor = float(t)

        return result

class MedianSurvivalEstimator:
    """Finds the estimated median survival time (S(t) = 0.5)."""

    def estimate(self, km_points: list[KMTimePoint]) -> float | None:
        """Return first time where S(t) <= 0.5.

        Args:
            km_points: Ordered KM time points.

        Returns:
            Median survival time or None if S(t) never reaches 0.5.
        """
        for point in km_points:
            if point.survival_probability <= 0.5:
                return float(point.time)
        return None

class KaplanMeierCalculator:
    """Kaplan-Meier survival curve with Greenwood confidence intervals.

    Workflow:
        calculator = KaplanMeierCalculator()
        result = calculator.calculate(
            data_frame=df,
            time_column="survival_days",
            event_column="event_occurred",   # 1=event, 0=censored
            confidence_level=0.95,
            group_column=None,               # optional stratification
        )
    """

    _MINIMUM_OBSERVATIONS: int = 5

    def __init__(self) -> None:
        self._estimator = KaplanMeierEstimator()
        self._median_estimator = MedianSurvivalEstimator()

    def calculate(
        self,
        data_frame: pd.DataFrame,
        time_column: str,
        event_column: str,
        confidence_level: float = 0.95,
        group_column: str | None = None,
    ) -> dict:
        """Estimate Kaplan-Meier survival curve.

        Args:
            data_frame: Source DataFrame.
            time_column: Time-to-event or censoring column (positive numeric).
            event_column: Binary event indicator (1=event, 0=censored).
            confidence_level: CI confidence level.
            group_column: Optional stratification column for group comparison.

        Returns:
            Dict with KM curve, median survival, CI, and optional group curves.

        Raises:
            KeyError: If columns are not found.
            ValueError: If data is insufficient or confidence_level is invalid.
        """
        for col in (time_column, event_column):
            if col not in data_frame.columns:
                raise KeyError(f"Column '{col}' not found in DataFrame.")
        if group_column is not None and group_column not in data_frame.columns:
            raise KeyError(f"Column '{group_column}' not found in DataFrame.")
        if not 0.0 < confidence_level < 1.0:
            raise ValueError(
                f"confidence_level must be in (0, 1). Got {confidence_level}."
            )

        clean = data_frame[[time_column, event_column] + ([group_column] if group_column else [])].dropna()
        clean = clean[clean[time_column] >= 0]

        if len(clean) < self._MINIMUM_OBSERVATIONS:
            raise ValueError(
                f"At least {self._MINIMUM_OBSERVATIONS} observations required. "
                f"Got {len(clean)}."
            )

        def _compute_curve(subset: pd.DataFrame) -> dict:
            times = subset[time_column].to_numpy(dtype=float)
            events = subset[event_column].to_numpy(dtype=float)
            points = self._estimator.estimate(times, events, confidence_level)
            median = self._median_estimator.estimate(points)
            event_rate = float(events.sum()) / len(events)

            return {
                "curve": [
                    {
                        "time": p.time,
                        "n_at_risk": p.n_at_risk,
                        "n_events": p.n_events,
                        "n_censored": p.n_censored,
                        "survival_probability": p.survival_probability,
                        "ci_lower": p.ci_lower,
                        "ci_upper": p.ci_upper,
                        "cumulative_hazard": p.cumulative_hazard,
                    }
                    for p in points
                ],
                "median_survival_time": median,
                "event_rate": round(event_rate, 4),
                "n_observations": len(subset),
                "n_events": int(events.sum()),
                "n_censored": int((events == 0).sum()),
            }

        if group_column is None:
            result = _compute_curve(clean)
            result["confidence_level"] = confidence_level
            return result

        groups = clean[group_column].unique()
        group_results: dict[str, dict] = {}
        for group in sorted(groups):
            subset = clean[clean[group_column] == group]
            if len(subset) >= self._MINIMUM_OBSERVATIONS:
                group_results[str(group)] = _compute_curve(subset)

        return {
            "groups": group_results,
            "confidence_level": confidence_level,
            "group_column": group_column,
            "n_groups": len(group_results),
        }