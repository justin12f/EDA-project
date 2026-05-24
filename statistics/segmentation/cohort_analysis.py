"""Cohort retention analysis: user retention matrix by acquisition cohort."""

# #[AI_CONTEXT_START]
# - CONFIGURACIÓN DE FACTORY: Registrar este calculator en `SegmentationStatisticsFactory` (backends pandas | polars | spark) y exponerlo mediante `StatisticsInyeccionDependency`, inyectada por la Factory Maestra de Agentes junto a las demás fábricas de dominio.
# - ABSTRACCIÓN DEL DATO: Mutar constructores y `analyze`/`compute` para recibir el contenedor abstracto del backend (`pd.DataFrame`, `pl.DataFrame`/`pl.LazyFrame`, `pyspark.sql.DataFrame`) inyectado por la factory; eliminar `np.ndarray`/`pd.Series` sueltos en firmas públicas.
# - REFACTOR NATIVO: Resolver métricas con expresiones 100 % nativas del backend activo (Polars: `.select`/`.group_by`/`.agg` sin `.collect()` salvo materialización acordada; PySpark: `pyspark.sql.functions` y ventanas distribuidas; Pandas: operaciones vectorizadas). No convertir a NumPy/Pandas desde backends no-pandas.
# #[AI_CONTEXT_END]
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

@dataclass(frozen=True)
class CohortRetentionRow:
    """Immutable retention row for one cohort."""

    cohort_label: str
    cohort_size: int
    retention_rates: dict[int, float]
    periods_observed: int

class CohortAssigner:
    """Assigns each user to their acquisition cohort based on first activity date."""

    def assign(
        self,
        data_frame: pd.DataFrame,
        user_column: str,
        date_column: str,
        period: str,
    ) -> pd.DataFrame:
        """Assign cohort labels to all rows."""
        df = data_frame.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        df["activity_period"] = df[date_column].dt.to_period(period)
        cohort_map = (
            df.groupby(user_column)["activity_period"]
            .min()
            .rename("cohort_period")
        )
        df = df.join(cohort_map, on=user_column)
        return df

class CohortPeriodOffsetCalculator:
    """Computes the integer period offset between cohort and activity periods."""

    def calculate(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """Add period_number column representing periods since cohort start."""
        df = data_frame.copy()
        df["period_number"] = (
            df["activity_period"] - df["cohort_period"]
        ).apply(lambda x: x.n)
        return df

class RetentionMatrixBuilder:
    """Builds the cohort × period retention count matrix."""

    def build(self, data_frame: pd.DataFrame, user_column: str) -> pd.DataFrame:
        """Build raw cohort retention count matrix."""
        cohort_data = (
            data_frame.groupby(["cohort_period", "period_number"])[user_column]
            .nunique()
            .reset_index()
        )
        cohort_data.columns = ["cohort_period", "period_number", "n_users"]
        matrix = cohort_data.pivot_table(
            index="cohort_period",
            columns="period_number",
            values="n_users",
        )
        return matrix

class RetentionRateNormalizer:
    """Normalizes retention counts to rates by dividing by cohort size (period 0)."""

    def normalize(self, count_matrix: pd.DataFrame) -> pd.DataFrame:
        """Divide each row by its period-0 value."""
        cohort_sizes = count_matrix.iloc[:, 0]
        return count_matrix.div(cohort_sizes, axis=0).round(4)

class CohortAnalysisCalculator:
    """Cohort retention analysis: acquisition cohorts tracked over time."""

    _MINIMUM_USERS: int = 10
    _MINIMUM_PERIODS: int = 2
    _VALID_PERIODS: frozenset[str] = frozenset({"M", "W", "Q"})

    def __init__(self) -> None:
        self._assigner = CohortAssigner()
        self._offset_calc = CohortPeriodOffsetCalculator()
        self._matrix_builder = RetentionMatrixBuilder()
        self._normalizer = RetentionRateNormalizer()

    def calculate(
        self,
        data_frame: pd.DataFrame,
        user_column: str,
        date_column: str,
        period: str = "M",
    ) -> dict:
        """Build cohort retention matrix and extract retention insights."""
        for col in (user_column, date_column):
            if col not in data_frame.columns:
                raise KeyError(f"Column '{col}' not found in DataFrame.")

        if period not in self._VALID_PERIODS:
            raise ValueError(
                f"period must be one of {self._VALID_PERIODS}. Got '{period}'."
            )

        clean = data_frame[[user_column, date_column]].dropna()
        n_unique_users = clean[user_column].nunique()

        if n_unique_users < self._MINIMUM_USERS:
            raise ValueError(
                f"At least {self._MINIMUM_USERS} unique users required. "
                f"Got {n_unique_users}."
            )

        assigned = self._assigner.assign(clean, user_column, date_column, period)
        with_offsets = self._offset_calc.calculate(assigned)
        count_matrix = self._matrix_builder.build(with_offsets, user_column)
        rate_matrix = self._normalizer.normalize(count_matrix)

        cohort_sizes = count_matrix.iloc[:, 0].to_dict()
        avg_retention_by_period = rate_matrix.mean(axis=0).to_dict()
        period_1_retention = (
            float(rate_matrix.iloc[:, 1].mean())
            if rate_matrix.shape[1] > 1 else None
        )

        return {
            "retention_rate_matrix": {
                str(cohort): {
                    int(p): round(float(rate), 4)
                    for p, rate in row.items()
                    if not np.isnan(rate)
                }
                for cohort, row in rate_matrix.iterrows()
            },
            "cohort_sizes": {str(k): int(v) for k, v in cohort_sizes.items()},
            "avg_retention_by_period": {
                int(p): round(float(r), 4)
                for p, r in avg_retention_by_period.items()
                if not np.isnan(r)
            },
            "period_1_avg_retention": (
                round(period_1_retention, 4)
                if period_1_retention is not None else None
            ),
            "n_cohorts": len(count_matrix),
            "n_unique_users": n_unique_users,
            "period_granularity": period,
        }
