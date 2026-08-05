"""Partial correlation controlling for one or more confounding variables."""

# #[AI_CONTEXT_START]
# - CONFIGURACIÓN DE FACTORY: Registrar este calculator en `RelationalStatisticsFactory` (backends pandas | polars | spark) y exponerlo mediante `StatisticsInyeccionDependency`, inyectada por la Factory Maestra de Agentes junto a las demás fábricas de dominio.
# - ABSTRACCIÓN DEL DATO: Mutar constructores y `analyze`/`compute` para recibir el contenedor abstracto del backend (`pd.DataFrame`, `pl.DataFrame`/`pl.LazyFrame`, `pyspark.sql.DataFrame`) inyectado por la factory; eliminar `np.ndarray`/`pd.Series` sueltos en firmas públicas.
# - REFACTOR NATIVO: Resolver métricas con expresiones 100 % nativas del backend activo (Polars: `.select`/`.group_by`/`.agg` sin `.collect()` salvo materialización acordada; PySpark: `pyspark.sql.functions` y ventanas distribuidas; Pandas: operaciones vectorizadas). No convertir a NumPy/Pandas desde backends no-pandas.
# #[AI_CONTEXT_END]
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

class ResidualExtractor:
    """Extracts OLS residuals of one variable regressed on covariates.

    The residuals represent the portion of a variable's variance that is
    orthogonal to (unexplained by) the covariates — i.e., the part that
    is *not* driven by the confounders.
    """

    def extract(
        self,
        series: pd.Series,
        covariates: pd.DataFrame,
    ) -> np.ndarray:
        """Regress series on covariates and return residuals.

        Args:
            series: Target variable to partial out.
            covariates: DataFrame of control variables.

        Returns:
            OLS residual array (same length as series).
        """
        x = np.column_stack(
            [np.ones(len(covariates)), covariates.to_numpy(dtype=float)]
        )
        y = series.to_numpy(dtype=float)
        coefficients, *_ = np.linalg.lstsq(x, y, rcond=None)
        y_pred = x @ coefficients
        return y - y_pred

class PartialCorrelationCalculator:
    """Computes partial correlation between two variables controlling for others.

    The partial correlation is the Pearson correlation between the residuals
    of regressing each variable on the set of covariates. This isolates the
    direct linear relationship between x and y after removing the shared
    variance attributable to the control variables.

    Workflow:
        calculator = PartialCorrelationCalculator()
        result = calculator.calculate(
            data_frame=df,
            column_x="income",
            column_y="savings",
            control_columns=["age", "education_years"],
            significance_level=0.05,
        )
    """

    _MINIMUM_SAMPLE_SIZE: int = 5

    def __init__(self) -> None:
        self._residual_extractor = ResidualExtractor()

    def calculate(
        self,
        data_frame: pd.DataFrame,
        column_x: str,
        column_y: str,
        control_columns: list[str],
        significance_level: float = 0.05,
    ) -> dict:
        """Compute partial correlation with significance test.

        Args:
            data_frame: Source DataFrame.
            column_x: First variable.
            column_y: Second variable.
            control_columns: Variables to control for (confounders).
            significance_level: Alpha threshold for significance test.

        Returns:
            Dict with partial r, p-value, df, and significance verdict.

        Raises:
            KeyError: If any column is not found in the DataFrame.
            ValueError: If insufficient data after dropping NaNs.
        """
        all_columns = [column_x, column_y] + control_columns
        missing_cols = [c for c in all_columns if c not in data_frame.columns]
        if missing_cols:
            raise KeyError(
                f"Columns not found in DataFrame: {missing_cols}"
            )

        overlap = set([column_x, column_y]) & set(control_columns)
        if overlap:
            raise ValueError(
                f"Control columns cannot include the target variables. "
                f"Overlap detected: {overlap}"
            )

        clean = data_frame[all_columns].dropna()
        n = len(clean)
        k = len(control_columns)
        df = n - k - 2

        if df < self._MINIMUM_SAMPLE_SIZE:
            raise ValueError(
                f"Insufficient degrees of freedom (df={df}). "
                f"Need at least {self._MINIMUM_SAMPLE_SIZE} after accounting for "
                f"{k} control variable(s) and 2 target variables."
            )

        covariates = clean[control_columns]
        residuals_x = self._residual_extractor.extract(clean[column_x], covariates)
        residuals_y = self._residual_extractor.extract(clean[column_y], covariates)

        partial_r, p_value = stats.pearsonr(residuals_x, residuals_y)

        return {
            "column_x": column_x,
            "column_y": column_y,
            "control_columns": control_columns,
            "partial_r": round(float(partial_r), 6),
            "p_value": round(float(p_value), 6),
            "degrees_of_freedom": df,
            "n_observations": n,
            "is_significant": float(p_value) < significance_level,
            "significance_level": significance_level,
        }
