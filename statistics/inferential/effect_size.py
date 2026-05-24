"""Effect size calculators for hypothesis tests."""

# #[AI_CONTEXT_START]
# - CONFIGURACIÓN DE FACTORY: Registrar este calculator en `InferentialStatisticsFactory` (backends pandas | polars | spark) y exponerlo mediante `StatisticsInyeccionDependency`, inyectada por la Factory Maestra de Agentes junto a las demás fábricas de dominio.
# - ABSTRACCIÓN DEL DATO: Mutar constructores y `analyze`/`compute` para recibir el contenedor abstracto del backend (`pd.DataFrame`, `pl.DataFrame`/`pl.LazyFrame`, `pyspark.sql.DataFrame`) inyectado por la factory; eliminar `np.ndarray`/`pd.Series` sueltos en firmas públicas.
# - REFACTOR NATIVO: Resolver métricas con expresiones 100 % nativas del backend activo (Polars: `.select`/`.group_by`/`.agg` sin `.collect()` salvo materialización acordada; PySpark: `pyspark.sql.functions` y ventanas distribuidas; Pandas: operaciones vectorizadas). No convertir a NumPy/Pandas desde backends no-pandas.
# #[AI_CONTEXT_END]
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from scipy import stats

class EffectSizeInterpreter:
    """Interprets magnitude of an effect size by convention (Cohen, 1988)."""

    _COHENS_D_THRESHOLDS: list[tuple[float, str]] = [
        (0.8, "large"),
        (0.5, "medium"),
        (0.2, "small"),
        (0.0, "negligible"),
    ]

    _CRAMERS_V_THRESHOLDS: list[tuple[float, str]] = [
        (0.5, "large"),
        (0.3, "medium"),
        (0.1, "small"),
        (0.0, "negligible"),
    ]

    _ETA_SQUARED_THRESHOLDS: list[tuple[float, str]] = [
        (0.14, "large"),
        (0.06, "medium"),
        (0.01, "small"),
        (0.0, "negligible"),
    ]

    def interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d magnitude."""
        return self._classify(abs(d), self._COHENS_D_THRESHOLDS)

    def interpret_cramers_v(self, v: float) -> str:
        """Interpret Cramér's V magnitude."""
        return self._classify(abs(v), self._CRAMERS_V_THRESHOLDS)

    def interpret_eta_squared(self, eta2: float) -> str:
        """Interpret eta-squared (η²) magnitude."""
        return self._classify(abs(eta2), self._ETA_SQUARED_THRESHOLDS)

    def _classify(self, value: float, thresholds: list[tuple[float, str]]) -> str:
        for threshold, label in thresholds:
            if value >= threshold:
                return label
        return "negligible"

class CohensDCalculator:
    """Cohen's d for the standardized mean difference between two groups.

    Uses pooled standard deviation (equal variance assumption).
    """

    def calculate(self, group_a: np.ndarray, group_b: np.ndarray) -> float:
        """Calculate Cohen's d.

        Args:
            group_a: Numerical array for group A.
            group_b: Numerical array for group B.

        Returns:
            Cohen's d value.

        Raises:
            ValueError: If either group has fewer than 2 observations.
        """
        if len(group_a) < 2 or len(group_b) < 2:
            raise ValueError(
                "Both groups must have at least 2 observations for Cohen's d."
            )

        pooled_std = float(np.sqrt(
            (group_a.var(ddof=1) * (len(group_a) - 1) +
             group_b.var(ddof=1) * (len(group_b) - 1)) /
            (len(group_a) + len(group_b) - 2)
        ))

        if pooled_std == 0.0:
            return 0.0

        return float((group_a.mean() - group_b.mean()) / pooled_std)

class EtaSquaredCalculator:
    """Eta-squared (η²) for one-way ANOVA — proportion of variance explained."""

    def calculate(self, groups: dict[str, np.ndarray]) -> float:
        """Calculate eta-squared from ANOVA groups.

        Args:
            groups: Dict mapping group name to numerical array.

        Returns:
            Eta-squared value in [0, 1].

        Raises:
            ValueError: If fewer than 2 groups are provided.
        """
        if len(groups) < 2:
            raise ValueError(
                "Eta-squared requires at least 2 groups."
            )

        all_data = np.concatenate(list(groups.values()))
        grand_mean = float(all_data.mean())

        ss_between = float(sum(
            len(data) * (float(data.mean()) - grand_mean) ** 2
            for data in groups.values()
        ))
        ss_total = float(np.sum((all_data - grand_mean) ** 2))

        return ss_between / ss_total if ss_total > 0 else 0.0

class EffectSizeCalculator:
    """Unified effect size dispatcher.

    Workflow:
        calculator = EffectSizeCalculator()

        # Cohen's d (two groups)
        result = calculator.calculate(
            "cohens_d", group_a=arr_a, group_b=arr_b
        )

        # Eta-squared (ANOVA)
        result = calculator.calculate(
            "eta_squared",
            groups={"a": arr_a, "b": arr_b, "c": arr_c}
        )
    """

    def __init__(self) -> None:
        self._cohens_d = CohensDCalculator()
        self._eta_squared = EtaSquaredCalculator()
        self._interpreter = EffectSizeInterpreter()

    def calculate(self, effect_type: str, **kwargs) -> dict:
        """Calculate the requested effect size.

        Args:
            effect_type: One of 'cohens_d', 'eta_squared'.
            **kwargs: Arguments forwarded to the specific calculator.

        Returns:
            Dictionary with effect size value and interpretation.

        Raises:
            KeyError: If effect_type is not recognized.
        """
        if effect_type == "cohens_d":
            group_a: np.ndarray = kwargs["group_a"]
            group_b: np.ndarray = kwargs["group_b"]
            d = self._cohens_d.calculate(group_a, group_b)
            return {
                "effect_type": "cohens_d",
                "value": d,
                "absolute_value": abs(d),
                "interpretation": self._interpreter.interpret_cohens_d(d),
            }

        if effect_type == "eta_squared":
            groups: dict[str, np.ndarray] = kwargs["groups"]
            eta2 = self._eta_squared.calculate(groups)
            return {
                "effect_type": "eta_squared",
                "value": eta2,
                "interpretation": self._interpreter.interpret_eta_squared(eta2),
            }

        raise KeyError(
            f"effect_type '{effect_type}' not recognized. "
            f"Available: 'cohens_d', 'eta_squared'."
        )
