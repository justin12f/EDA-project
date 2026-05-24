"""Run rate projection: annualization from partial period data."""

# #[AI_CONTEXT_START]
# - CONFIGURACIÓN DE FACTORY: Registrar este calculator en `BusinessStatisticsFactory` (backends pandas | polars | spark) y exponerlo mediante `StatisticsInyeccionDependency`, inyectada por la Factory Maestra de Agentes junto a las demás fábricas de dominio.
# - ABSTRACCIÓN DEL DATO: Mutar constructores y `analyze`/`compute` para recibir el contenedor abstracto del backend (`pd.DataFrame`, `pl.DataFrame`/`pl.LazyFrame`, `pyspark.sql.DataFrame`) inyectado por la factory; eliminar `np.ndarray`/`pd.Series` sueltos en firmas públicas.
# - REFACTOR NATIVO: Resolver métricas con expresiones 100 % nativas del backend activo (Polars: `.select`/`.group_by`/`.agg` sin `.collect()` salvo materialización acordada; PySpark: `pyspark.sql.functions` y ventanas distribuidas; Pandas: operaciones vectorizadas). No convertir a NumPy/Pandas desde backends no-pandas.
# #[AI_CONTEXT_END]
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

class ProjectionMethod(str, Enum):
    """Run rate projection methodology."""

    SIMPLE = "simple"           # Linear extrapolation from elapsed fraction
    TRAILING_AVERAGE = "trailing_average"   # Average of last N periods × full periods
    WEIGHTED_RECENT = "weighted_recent"     # Exponentially weighted recent periods

@dataclass(frozen=True)
class RunRateResult:
    """Immutable run rate projection result."""

    method: str
    observed_value: float
    projected_full_period: float
    elapsed_fraction: float
    remaining_fraction: float
    confidence_note: str

class SimpleRunRateProjector:
    """Simple linear run rate: observed / elapsed_fraction.

    Assumes current pace is constant for the remainder of the period.
    Best for stable, non-seasonal metrics.
    """

    def project(
        self, observed: float, elapsed_fraction: float
    ) -> float:
        """Project full-period value.

        Args:
            observed: Metric value accumulated so far.
            elapsed_fraction: Fraction of period elapsed (e.g., 0.5 = half).

        Returns:
            Projected full-period value.
        """
        if elapsed_fraction <= 0:
            return observed
        return observed / elapsed_fraction

class TrailingAverageProjector:
    """Run rate based on the average of the last N complete periods.

    Uses historical pace instead of current partial-period data.
    Better for metrics with short-term noise.
    """

    def project(
        self,
        historical_values: np.ndarray,
        n_periods_trailing: int,
        full_periods: int,
    ) -> float:
        """Project by annualizing the trailing average.

        Args:
            historical_values: Array of complete-period values (chronological).
            n_periods_trailing: Number of recent periods to average.
            full_periods: Number of periods in the full projection window.

        Returns:
            Projected full-period total.
        """
        n = min(n_periods_trailing, len(historical_values))
        trailing_avg = float(np.mean(historical_values[-n:]))
        return trailing_avg * full_periods

class WeightedRecentProjector:
    """Run rate using exponentially weighted recent periods.

    More recent periods receive higher weight. Useful when the metric
    has a trend (accelerating or decelerating growth).
    """

    def project(
        self,
        historical_values: np.ndarray,
        full_periods: int,
        decay: float,
    ) -> float:
        """Project using EWMA of recent periods.

        Args:
            historical_values: Complete-period values (chronological).
            full_periods: Periods in the full projection window.
            decay: EWMA decay factor (higher = slower decay = longer memory).

        Returns:
            Projected full-period total.
        """
        n = len(historical_values)
        weights = np.array([decay ** (n - 1 - i) for i in range(n)])
        weights /= weights.sum()
        weighted_avg = float(np.dot(weights, historical_values))
        return weighted_avg * full_periods

class RunRateCalculator:
    """Run rate projections from partial-period or historical data.

    Workflow — from partial period (YTD/QTD):
        calculator = RunRateCalculator()
        result = calculator.calculate(
            observed_value=750_000,
            elapsed_fraction=0.625,          # 7.5 of 12 months elapsed
            methods=["simple"],
        )

    Workflow — from historical data:
        result = calculator.calculate(
            data_frame=df,
            value_column="monthly_revenue",
            full_periods=12,
            n_periods_trailing=3,
            decay=0.9,
            methods=["trailing_average", "weighted_recent"],
        )
    """

    def __init__(self) -> None:
        self._simple = SimpleRunRateProjector()
        self._trailing = TrailingAverageProjector()
        self._weighted = WeightedRecentProjector()

    def calculate(
        self,
        observed_value: float | None = None,
        elapsed_fraction: float | None = None,
        data_frame: pd.DataFrame | None = None,
        value_column: str | None = None,
        full_periods: int = 12,
        n_periods_trailing: int = 3,
        decay: float = 0.9,
        methods: list[str] | None = None,
    ) -> dict:
        """Compute run rate projections.

        Args:
            observed_value: YTD/QTD accumulated value (simple mode).
            elapsed_fraction: Fraction of period elapsed in (0, 1] (simple mode).
            data_frame: DataFrame with historical period values (historical mode).
            value_column: Numeric column in data_frame.
            full_periods: Target periods in the projection window.
            n_periods_trailing: Trailing periods for TrailingAverage method.
            decay: EWMA decay for WeightedRecent method.
            methods: Methods to run. Defaults to available based on inputs.

        Returns:
            Dict with one RunRateResult per method and comparison summary.

        Raises:
            ValueError: If neither simple nor historical inputs are provided,
                or if parameters are invalid.
        """
        _VALID_METHODS: frozenset[str] = frozenset(
            {m.value for m in ProjectionMethod}
        )

        active_methods = methods if methods is not None else list(_VALID_METHODS)
        invalid = [m for m in active_methods if m not in _VALID_METHODS]
        if invalid:
            raise ValueError(
                f"Invalid method(s): {invalid}. Available: {_VALID_METHODS}"
            )

        if not 0.0 < decay <= 1.0:
            raise ValueError(f"decay must be in (0, 1]. Got {decay}.")
        if full_periods < 1:
            raise ValueError(f"full_periods must be >= 1. Got {full_periods}.")

        results: list[RunRateResult] = []

        if "simple" in active_methods:
            if observed_value is None or elapsed_fraction is None:
                raise ValueError(
                    "'observed_value' and 'elapsed_fraction' are required for 'simple' method."
                )
            if not 0.0 < elapsed_fraction <= 1.0:
                raise ValueError(
                    f"elapsed_fraction must be in (0, 1]. Got {elapsed_fraction}."
                )
            projected = self._simple.project(observed_value, elapsed_fraction)
            results.append(
                RunRateResult(
                    method="simple",
                    observed_value=round(observed_value, 4),
                    projected_full_period=round(projected, 4),
                    elapsed_fraction=elapsed_fraction,
                    remaining_fraction=round(1.0 - elapsed_fraction, 4),
                    confidence_note=(
                        "High confidence if metric is stable."
                        if elapsed_fraction >= 0.5
                        else "Low confidence — less than 50% of period elapsed."
                    ),
                )
            )

        historical_methods = {"trailing_average", "weighted_recent"}
        if any(m in active_methods for m in historical_methods):
            if data_frame is None or value_column is None:
                raise ValueError(
                    "'data_frame' and 'value_column' are required for "
                    "trailing_average and weighted_recent methods."
                )
            if value_column not in data_frame.columns:
                raise KeyError(f"Column '{value_column}' not found in DataFrame.")

            historical = data_frame[value_column].dropna().to_numpy(dtype=float)

            if len(historical) < 2:
                raise ValueError(
                    "At least 2 historical periods required for trailing methods."
                )

            if "trailing_average" in active_methods:
                projected = self._trailing.project(
                    historical, n_periods_trailing, full_periods
                )
                results.append(
                    RunRateResult(
                        method="trailing_average",
                        observed_value=round(float(historical[-1]), 4),
                        projected_full_period=round(projected, 4),
                        elapsed_fraction=1.0,
                        remaining_fraction=0.0,
                        confidence_note=(
                            f"Based on average of last {min(n_periods_trailing, len(historical))} periods."
                        ),
                    )
                )

            if "weighted_recent" in active_methods:
                projected = self._weighted.project(historical, full_periods, decay)
                results.append(
                    RunRateResult(
                        method="weighted_recent",
                        observed_value=round(float(historical[-1]), 4),
                        projected_full_period=round(projected, 4),
                        elapsed_fraction=1.0,
                        remaining_fraction=0.0,
                        confidence_note=(
                            f"Exponentially weighted (decay={decay}). "
                            "Captures recent trend direction."
                        ),
                    )
                )

        projections = [r.projected_full_period for r in results]

        return {
            "projections": [
                {
                    "method": r.method,
                    "observed_value": r.observed_value,
                    "projected_full_period": r.projected_full_period,
                    "elapsed_fraction": r.elapsed_fraction,
                    "remaining_fraction": r.remaining_fraction,
                    "confidence_note": r.confidence_note,
                }
                for r in results
            ],
            "summary": {
                "mean_projection": round(float(np.mean(projections)), 4),
                "min_projection": round(float(np.min(projections)), 4),
                "max_projection": round(float(np.max(projections)), 4),
                "projection_range": round(float(np.max(projections) - np.min(projections)), 4),
                "n_methods": len(results),
            },
            "full_periods": full_periods,
        }
