"""Churn rate analysis by period and cohort."""

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
class ChurnPeriodResult:
    """Immutable churn result for a single period."""

    period: str
    n_customers_start: int
    n_churned: int
    n_new: int
    n_customers_end: int
    churn_rate: float
    retention_rate: float
    net_growth: int

class PeriodChurnCalculator:
    """Computes churn rate for each period from a customer state log.

    Churn rate = churned / customers_at_start_of_period
    Retention rate = 1 - churn_rate
    """

    def calculate(
        self,
        data_frame: pd.DataFrame,
        period_column: str,
        customers_start_column: str,
        churned_column: str,
        new_customers_column: str | None,
    ) -> list[ChurnPeriodResult]:
        """Compute period-level churn metrics.

        Args:
            data_frame: Aggregated DataFrame with one row per period.
            period_column: Period label column.
            customers_start_column: Customers at start of period column.
            churned_column: Number of churned customers column.
            new_customers_column: New customers added column (optional).

        Returns:
            List of ChurnPeriodResult for each period.
        """
        results: list[ChurnPeriodResult] = []

        for _, row in data_frame.iterrows():
            n_start = int(row[customers_start_column])
            n_churned = int(row[churned_column])
            n_new = int(row[new_customers_column]) if new_customers_column else 0
            n_end = n_start - n_churned + n_new
            churn_rate = n_churned / n_start if n_start > 0 else 0.0
            retention_rate = 1.0 - churn_rate

            results.append(
                ChurnPeriodResult(
                    period=str(row[period_column]),
                    n_customers_start=n_start,
                    n_churned=n_churned,
                    n_new=n_new,
                    n_customers_end=max(n_end, 0),
                    churn_rate=round(churn_rate, 6),
                    retention_rate=round(retention_rate, 6),
                    net_growth=n_new - n_churned,
                )
            )

        return results

class ChurnFromEventsCalculator:
    """Derives churn from a subscription/activity event log.

    A customer is considered churned in period P if they were active
    in period P-1 but have no activity in period P.
    """

    def calculate(
        self,
        data_frame: pd.DataFrame,
        user_column: str,
        period_column: str,
    ) -> pd.DataFrame:
        """Compute period-level churn from activity log.

        Args:
            data_frame: Activity log with user-period rows.
            user_column: User identifier column.
            period_column: Period identifier column (sortable).

        Returns:
            Aggregated DataFrame with period, customers_start, churned, new columns.

        Raises:
            ValueError: If fewer than 2 periods are present.
        """
        df = data_frame[[user_column, period_column]].drop_duplicates()
        periods = sorted(df[period_column].unique())

        if len(periods) < 2:
            raise ValueError(
                f"At least 2 periods required for churn computation. "
                f"Got {len(periods)}."
            )

        rows: list[dict] = []

        for i in range(1, len(periods)):
            prev_period = periods[i - 1]
            curr_period = periods[i]

            prev_users = set(df[df[period_column] == prev_period][user_column])
            curr_users = set(df[df[period_column] == curr_period][user_column])

            churned = prev_users - curr_users
            new_users = curr_users - prev_users

            rows.append({
                "period": str(curr_period),
                "customers_start": len(prev_users),
                "churned": len(churned),
                "new_customers": len(new_users),
            })

        return pd.DataFrame(rows)

class ChurnRateCalculator:
    """Churn rate analysis from aggregated data or event logs.

    Workflow — from aggregated data:
        calculator = ChurnRateCalculator()
        result = calculator.calculate(
            data_frame=df,
            period_column="month",
            customers_start_column="customers_start",
            churned_column="churned",
            new_customers_column="new_customers",   # optional
        )

    Workflow — from event log:
        result = calculator.calculate(
            data_frame=df,
            user_column="user_id",
            period_column="month",
            mode="events",
        )
    """

    _MINIMUM_PERIODS: int = 2

    def __init__(self) -> None:
        self._period_calculator = PeriodChurnCalculator()
        self._events_calculator = ChurnFromEventsCalculator()

    def calculate(
        self,
        data_frame: pd.DataFrame,
        period_column: str,
        mode: str = "aggregated",
        customers_start_column: str | None = None,
        churned_column: str | None = None,
        new_customers_column: str | None = None,
        user_column: str | None = None,
    ) -> dict:
        """Compute period-level churn metrics.

        Args:
            data_frame: Source DataFrame.
            period_column: Period label/identifier column.
            mode: 'aggregated' or 'events'.
            customers_start_column: Required for 'aggregated' mode.
            churned_column: Required for 'aggregated' mode.
            new_customers_column: Optional for 'aggregated' mode.
            user_column: Required for 'events' mode.

        Returns:
            Dict with per-period churn rates, averages, and trend.

        Raises:
            ValueError: If mode is invalid or required columns are missing.
        """
        _VALID_MODES: frozenset[str] = frozenset({"aggregated", "events"})

        if mode not in _VALID_MODES:
            raise ValueError(
                f"mode must be one of {_VALID_MODES}. Got '{mode}'."
            )
        if period_column not in data_frame.columns:
            raise KeyError(f"Column '{period_column}' not found in DataFrame.")

        if mode == "aggregated":
            if customers_start_column is None or churned_column is None:
                raise ValueError(
                    "'customers_start_column' and 'churned_column' are required "
                    "for mode='aggregated'."
                )
            for col in (customers_start_column, churned_column):
                if col not in data_frame.columns:
                    raise KeyError(f"Column '{col}' not found in DataFrame.")

            results = self._period_calculator.calculate(
                data_frame, period_column,
                customers_start_column, churned_column, new_customers_column
            )
        else:
            if user_column is None:
                raise ValueError(
                    "'user_column' is required for mode='events'."
                )
            if user_column not in data_frame.columns:
                raise KeyError(f"Column '{user_column}' not found in DataFrame.")

            agg_df = self._events_calculator.calculate(
                data_frame, user_column, period_column
            )
            results = self._period_calculator.calculate(
                agg_df, "period", "customers_start", "churned", "new_customers"
            )

        if len(results) < self._MINIMUM_PERIODS:
            raise ValueError(
                f"At least {self._MINIMUM_PERIODS} periods required. "
                f"Got {len(results)}."
            )

        churn_rates = [r.churn_rate for r in results]
        retention_rates = [r.retention_rate for r in results]

        return {
            "periods": [
                {
                    "period": r.period,
                    "n_customers_start": r.n_customers_start,
                    "n_churned": r.n_churned,
                    "n_new": r.n_new,
                    "n_customers_end": r.n_customers_end,
                    "churn_rate": r.churn_rate,
                    "churn_rate_pct": round(r.churn_rate * 100, 4),
                    "retention_rate": r.retention_rate,
                    "net_growth": r.net_growth,
                }
                for r in results
            ],
            "summary": {
                "mean_churn_rate": round(float(np.mean(churn_rates)), 6),
                "mean_retention_rate": round(float(np.mean(retention_rates)), 6),
                "peak_churn_period": results[int(np.argmax(churn_rates))].period,
                "lowest_churn_period": results[int(np.argmin(churn_rates))].period,
                "trend": "improving" if churn_rates[-1] < churn_rates[0]
                         else "worsening" if churn_rates[-1] > churn_rates[0]
                         else "stable",
                "n_periods": len(results),
            },
        }
