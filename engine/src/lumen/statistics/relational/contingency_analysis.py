"""Contingency table analysis with odds ratio and relative risk."""

# #[AI_CONTEXT_START]
# - CONFIGURACIÓN DE FACTORY: Registrar este calculator en `RelationalStatisticsFactory` (backends pandas | polars | spark) y exponerlo mediante `StatisticsInyeccionDependency`, inyectada por la Factory Maestra de Agentes junto a las demás fábricas de dominio.
# - ABSTRACCIÓN DEL DATO: Mutar constructores y `analyze`/`compute` para recibir el contenedor abstracto del backend (`pd.DataFrame`, `pl.DataFrame`/`pl.LazyFrame`, `pyspark.sql.DataFrame`) inyectado por la factory; eliminar `np.ndarray`/`pd.Series` sueltos en firmas públicas.
# - REFACTOR NATIVO: Resolver métricas con expresiones 100 % nativas del backend activo (Polars: `.select`/`.group_by`/`.agg` sin `.collect()` salvo materialización acordada; PySpark: `pyspark.sql.functions` y ventanas distribuidas; Pandas: operaciones vectorizadas). No convertir a NumPy/Pandas desde backends no-pandas.
# #[AI_CONTEXT_END]
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

@dataclass(frozen=True)
class TwoByTwoTable:
    """Validated 2×2 contingency table cell counts.

    Layout:
        |           | Outcome+ | Outcome- |
        |-----------|----------|----------|
        | Exposed+  |    a     |    b     |
        | Exposed-  |    c     |    d     |
    """

    a: float  # exposed + outcome+
    b: float  # exposed + outcome-
    c: float  # exposed- + outcome+
    d: float  # exposed- + outcome-

class OddsRatioCalculator:
    """Calculates odds ratio with Woolf (log) confidence interval.

    OR = (a×d) / (b×c)
    The OR measures how much more likely exposure is among those with
    the outcome vs. without it.
    """

    def calculate(
        self, table: TwoByTwoTable, confidence_level: float
    ) -> dict:
        """Calculate OR and CI.

        Args:
            table: 2×2 contingency table.
            confidence_level: Desired CI level.

        Returns:
            Dict with or_value, ci_lower, ci_upper, interpretation.

        Raises:
            ValueError: If any cell is zero (OR undefined).
        """
        if any(v == 0 for v in (table.a, table.b, table.c, table.d)):
            raise ValueError(
                "Odds ratio is undefined when any cell in the 2×2 table is zero."
            )

        or_value = (table.a * table.d) / (table.b * table.c)
        log_or = float(np.log(or_value))
        se_log_or = float(np.sqrt(
            1/table.a + 1/table.b + 1/table.c + 1/table.d
        ))

        alpha = 1.0 - confidence_level
        z = float(stats.norm.ppf(1 - alpha / 2))

        ci_lower = float(np.exp(log_or - z * se_log_or))
        ci_upper = float(np.exp(log_or + z * se_log_or))

        interpretation = (
            "increased odds of outcome with exposure"
            if or_value > 1
            else "decreased odds of outcome with exposure"
            if or_value < 1
            else "no association"
        )

        return {
            "or_value": round(or_value, 4),
            "log_or": round(log_or, 4),
            "ci_lower": round(ci_lower, 4),
            "ci_upper": round(ci_upper, 4),
            "confidence_level": confidence_level,
            "interpretation": interpretation,
        }

class RelativeRiskCalculator:
    """Calculates relative risk (risk ratio) with log CI.

    RR = [a/(a+b)] / [c/(c+d)]
    Measures how much more likely the outcome is in exposed vs. unexposed groups.
    Valid only for cohort studies (not case-control).
    """

    def calculate(
        self, table: TwoByTwoTable, confidence_level: float
    ) -> dict:
        """Calculate RR and CI.

        Args:
            table: 2×2 contingency table.
            confidence_level: Desired CI level.

        Returns:
            Dict with rr_value, ci_lower, ci_upper, interpretation.

        Raises:
            ValueError: If either row total is zero.
        """
        exposed_total = table.a + table.b
        unexposed_total = table.c + table.d

        if exposed_total == 0:
            raise ValueError("Exposed group total is zero. RR is undefined.")
        if unexposed_total == 0:
            raise ValueError("Unexposed group total is zero. RR is undefined.")

        risk_exposed = table.a / exposed_total
        risk_unexposed = table.c / unexposed_total

        if risk_unexposed == 0:
            raise ValueError(
                "Risk in unexposed group is zero. RR is undefined (infinite)."
            )

        rr = risk_exposed / risk_unexposed
        log_rr = float(np.log(rr))
        se_log_rr = float(np.sqrt(
            (1 - risk_exposed) / (table.a + 1e-10) +
            (1 - risk_unexposed) / (table.c + 1e-10)
        ))

        alpha = 1.0 - confidence_level
        z = float(stats.norm.ppf(1 - alpha / 2))

        ci_lower = float(np.exp(log_rr - z * se_log_rr))
        ci_upper = float(np.exp(log_rr + z * se_log_rr))

        interpretation = (
            "exposure increases risk"
            if rr > 1
            else "exposure decreases risk"
            if rr < 1
            else "no difference in risk"
        )

        return {
            "rr_value": round(rr, 4),
            "risk_exposed": round(risk_exposed, 4),
            "risk_unexposed": round(risk_unexposed, 4),
            "ci_lower": round(ci_lower, 4),
            "ci_upper": round(ci_upper, 4),
            "confidence_level": confidence_level,
            "interpretation": interpretation,
        }

class ContingencyAnalysisCalculator:
    """Full contingency table analysis: chi-square, OR, RR, and Cramér's V.

    Workflow:
        calculator = ContingencyAnalysisCalculator()
        result = calculator.calculate(
            data_frame=df,
            column_exposure="vaccinated",
            column_outcome="infected",
            significance_level=0.05,
            confidence_level=0.95,
        )
    """

    def __init__(self) -> None:
        self._or_calculator = OddsRatioCalculator()
        self._rr_calculator = RelativeRiskCalculator()

    def calculate(
        self,
        data_frame: pd.DataFrame,
        column_exposure: str,
        column_outcome: str,
        significance_level: float = 0.05,
        confidence_level: float = 0.95,
    ) -> dict:
        """Run full contingency analysis.

        Args:
            data_frame: Source DataFrame.
            column_exposure: Binary exposure variable column name.
            column_outcome: Binary outcome variable column name.
            significance_level: Alpha for chi-square test.
            confidence_level: CI level for OR and RR.

        Returns:
            Dict with chi-square, OR, RR, Cramér's V, and contingency table.

        Raises:
            KeyError: If columns are missing.
            ValueError: If columns are not binary.
        """
        for col in (column_exposure, column_outcome):
            if col not in data_frame.columns:
                raise KeyError(f"Column '{col}' not found in DataFrame.")

        clean = data_frame[[column_exposure, column_outcome]].dropna()
        contingency_table = pd.crosstab(
            clean[column_exposure], clean[column_outcome]
        )

        if contingency_table.shape != (2, 2):
            raise ValueError(
                f"Contingency analysis requires exactly 2×2 tables. "
                f"Got shape {contingency_table.shape}. "
                "Ensure both columns are binary."
            )

        observed = contingency_table.to_numpy().astype(float)
        a, b, c, d = observed[1, 1], observed[1, 0], observed[0, 1], observed[0, 0]
        table = TwoByTwoTable(a=a, b=b, c=c, d=d)

        chi2, p_value, dof, _ = stats.chi2_contingency(observed)
        n = int(observed.sum())
        cramers_v = float(np.sqrt(chi2 / (n * (min(observed.shape) - 1))))

        or_result = self._or_calculator.calculate(table, confidence_level)
        rr_result = self._rr_calculator.calculate(table, confidence_level)

        return {
            "column_exposure": column_exposure,
            "column_outcome": column_outcome,
            "contingency_table": contingency_table.to_dict(),
            "n_observations": n,
            "chi_square": {
                "statistic": round(float(chi2), 4),
                "p_value": round(float(p_value), 6),
                "degrees_of_freedom": int(dof),
                "reject_null": float(p_value) < significance_level,
            },
            "cramers_v": round(cramers_v, 4),
            "odds_ratio": or_result,
            "relative_risk": rr_result,
        }
