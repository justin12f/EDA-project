"""Growth rate analysis: MoM, YoY, CAGR, and rolling growth."""

# #[AI_CONTEXT_START]
# - CONFIGURACIÓN DE FACTORY: Registrar este calculator en `BusinessStatisticsFactory` (backends pandas | polars | spark) y exponerlo mediante `StatisticsInyeccionDependency`, inyectada por la Factory Maestra de Agentes junto a las demás fábricas de dominio.
# - ABSTRACCIÓN DEL DATO: Mutar constructores y `analyze`/`compute` para recibir el contenedor abstracto del backend (`pd.DataFrame`, `pl.DataFrame`/`pl.LazyFrame`, `pyspark.sql.DataFrame`) inyectado por la factory; eliminar `np.ndarray`/`pd.Series` sueltos en firmas públicas.
# - REFACTOR NATIVO: Resolver métricas con expresiones 100 % nativas del backend activo (Polars: `.select`/`.group_by`/`.agg` sin `.collect()` salvo materialización acordada; PySpark: `pyspark.sql.functions` y ventanas distribuidas; Pandas: operaciones vectorizadas). No convertir a NumPy/Pandas desde backends no-pandas.
# #[AI_CONTEXT_END]
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

@dataclass(frozen=True)
class PeriodGrowthResult:
    """Immutable growth result for a single period transition."""

    period_from: str
    period_to: str
    value_from: float
    value_to: float
    absolute_change: float
    growth_rate_pct: float

class PeriodOverPeriodCalculator:
    """Computes growth rate between consecutive periods.

    rate(t) = (X(t) - X(t-1)) / |X(t-1)| × 100

    Returns NaN when the base value is zero to avoid division errors.
    """

    def calculate(
        self,
        values: np.ndarray,
        period_labels: list[str],
    ) -> list[PeriodGrowthResult]:
        """Compute period-over-period growth rates.

        Args:
            values: Numeric array of metric values ordered by time.
            period_labels: String labels for each period (same length as values).

        Returns:
            List of PeriodGrowthResult for each consecutive transition.
        """
        results: list[PeriodGrowthResult] = []

        for i in range(1, len(values)):
            base = float(values[i - 1])
            current = float(values[i])
            absolute_change = current - base
            growth_rate = (
                (absolute_change / abs(base) * 100)
                if base != 0.0
                else float("nan")
            )
            results.append(
                PeriodGrowthResult(
                    period_from=period_labels[i - 1],
                    period_to=period_labels[i],
                    value_from=round(base, 4),
                    value_to=round(current, 4),
                    absolute_change=round(absolute_change, 4),
                    growth_rate_pct=round(growth_rate, 4),
                )
            )

        return results

class CAGRCalculator:
    """Computes Compound Annual Growth Rate between two points in time.

    CAGR = (V_end / V_start)^(1/n_years) - 1

    where n_years is the number of years between start and end.
    """

    def calculate(
        self,
        value_start: float,
        value_end: float,
        n_years: float,
    ) -> float:
        """Compute CAGR.

        Args:
            value_start: Starting value.
            value_end: Ending value.
            n_years: Number of years in the period.

        Returns:
            CAGR as a decimal (e.g., 0.12 = 12% per year).

        Raises:
            ValueError: If value_start is zero or non-positive, or n_years <= 0.
        """
        if value_start <= 0:
            raise ValueError(
                f"value_start must be > 0 for CAGR computation. Got {value_start}."
            )
        if value_end < 0:
            raise ValueError(
                f"value_end must be >= 0. Got {value_end}."
            )
        if n_years <= 0:
            raise ValueError(
                f"n_years must be > 0. Got {n_years}."
            )
        return float((value_end / value_start) ** (1.0 / n_years) - 1)

class RollingGrowthCalculator:
    """Computes rolling n-period growth rate across a time series.

    rolling_growth(t, n) = (X(t) - X(t-n)) / |X(t-n)| × 100
    """

    def calculate(
        self,
        values: np.ndarray,
        window: int,
    ) -> np.ndarray:
        """Compute rolling growth rate.

        Args:
            values: Numeric array of metric values.
            window: Look-back window in periods.

        Returns:
            Rolling growth rate array (NaN for first `window` positions).
        """
        result = np.full(len(values), np.nan)
        for i in range(window, len(values)):
            base = float(values[i - window])
            current = float(values[i])
            if base != 0.0:
                result[i] = (current - base) / abs(base) * 100.0
        return result

class GrowthRatesCalculator:
    """Full growth rate analysis: MoM/YoY, CAGR, and rolling growth.

    Workflow:
        calculator = GrowthRatesCalculator()
        result = calculator.calculate(
            data_frame=df,
            value_column="revenue",
            date_column="month",       # optional, used as period labels
            period_window=1,           # periods for rolling growth
            n_years=None,              # auto-computed from date range if None
            periods_per_year=12,       # 12=monthly, 52=weekly, 4=quarterly
        )
    """

    _MINIMUM_PERIODS: int = 2

    def __init__(self) -> None:
        self._pop_calculator = PeriodOverPeriodCalculator()
        self._cagr_calculator = CAGRCalculator()
        self._rolling_calculator = RollingGrowthCalculator()

    def calculate(
        self,
        data_frame: pd.DataFrame,
        value_column: str,
        date_column: str | None = None,
        period_window: int = 1,
        n_years: float | None = None,
        periods_per_year: int = 12,
    ) -> dict:
        """Compute growth rate analysis.

        Args:
            data_frame: Source DataFrame (time-ordered).
            value_column: Numeric metric column.
            date_column: Optional date/period label column.
            period_window: Look-back periods for rolling growth rate.
            n_years: Override for CAGR period length. Auto-computed if None.
            periods_per_year: Periods in a year for CAGR auto-computation.

        Returns:
            Dict with period-over-period rates, CAGR, rolling growth, and summary.

        Raises:
            KeyError: If columns are not found.
            ValueError: If data is insufficient or parameters are invalid.
        """
        if value_column not in data_frame.columns:
            raise KeyError(f"Column '{value_column}' not found in DataFrame.")
        if date_column is not None and date_column not in data_frame.columns:
            raise KeyError(f"Column '{date_column}' not found in DataFrame.")
        if period_window < 1:
            raise ValueError(f"period_window must be >= 1. Got {period_window}.")
        if periods_per_year < 1:
            raise ValueError(f"periods_per_year must be >= 1. Got {periods_per_year}.")

        clean = data_frame[[value_column] + ([date_column] if date_column else [])].dropna()

        if len(clean) < self._MINIMUM_PERIODS:
            raise ValueError(
                f"At least {self._MINIMUM_PERIODS} periods required. "
                f"Got {len(clean)}."
            )

        values = clean[value_column].to_numpy(dtype=float)
        period_labels = (
            clean[date_column].astype(str).tolist()
            if date_column else [str(i) for i in range(len(values))]
        )

        pop_results = self._pop_calculator.calculate(values, period_labels)

        value_start = float(values[0])
        value_end = float(values[-1])
        total_periods = len(values) - 1
        effective_n_years = (
            n_years if n_years is not None
            else total_periods / periods_per_year
        )

        cagr: float | None = None
        if value_start > 0 and value_end >= 0 and effective_n_years > 0:
            cagr = self._cagr_calculator.calculate(
                value_start, value_end, effective_n_years
            )

        rolling_growth = self._rolling_calculator.calculate(values, period_window)

        valid_pop = [r.growth_rate_pct for r in pop_results if not np.isnan(r.growth_rate_pct)]
        valid_rolling = rolling_growth[~np.isnan(rolling_growth)]

        return {
            "period_over_period": [
                {
                    "period_from": r.period_from,
                    "period_to": r.period_to,
                    "value_from": r.value_from,
                    "value_to": r.value_to,
                    "absolute_change": r.absolute_change,
                    "growth_rate_pct": r.growth_rate_pct,
                }
                for r in pop_results
            ],
            "cagr": round(cagr, 6) if cagr is not None else None,
            "cagr_pct": round(cagr * 100, 4) if cagr is not None else None,
            "rolling_growth": rolling_growth.tolist(),
            "summary": {
                "total_absolute_change": round(value_end - value_start, 4),
                "total_growth_pct": round(
                    (value_end - value_start) / abs(value_start) * 100, 4
                ) if value_start != 0 else None,
                "mean_period_growth_pct": round(float(np.mean(valid_pop)), 4) if valid_pop else None,
                "median_period_growth_pct": round(float(np.median(valid_pop)), 4) if valid_pop else None,
                "best_period": max(pop_results, key=lambda r: r.growth_rate_pct if not np.isnan(r.growth_rate_pct) else -float("inf")).period_to if pop_results else None,
                "worst_period": min(pop_results, key=lambda r: r.growth_rate_pct if not np.isnan(r.growth_rate_pct) else float("inf")).period_to if pop_results else None,
                "n_periods": len(values),
                "periods_per_year": periods_per_year,
            },
        }
