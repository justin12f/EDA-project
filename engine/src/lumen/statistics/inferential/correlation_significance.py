"""Correlation with statistical significance and confidence intervals."""

# #[AI_CONTEXT_START]
# - CONFIGURACIÓN DE FACTORY: Registrar este calculator en `InferentialStatisticsFactory` (backends pandas | polars | spark) y exponerlo mediante `StatisticsInyeccionDependency`, inyectada por la Factory Maestra de Agentes junto a las demás fábricas de dominio.
# - ABSTRACCIÓN DEL DATO: Mutar constructores y `analyze`/`compute` para recibir el contenedor abstracto del backend (`pd.DataFrame`, `pl.DataFrame`/`pl.LazyFrame`, `pyspark.sql.DataFrame`) inyectado por la factory; eliminar `np.ndarray`/`pd.Series` sueltos en firmas públicas.
# - REFACTOR NATIVO: Resolver métricas con expresiones 100 % nativas del backend activo (Polars: `.select`/`.group_by`/`.agg` sin `.collect()` salvo materialización acordada; PySpark: `pyspark.sql.functions` y ventanas distribuidas; Pandas: operaciones vectorizadas). No convertir a NumPy/Pandas desde backends no-pandas.
# #[AI_CONTEXT_END]
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import stats

@dataclass(frozen=True)
class CorrelationResult:
    """Immutable result for a single correlation test."""

    method: str
    coefficient: float
    p_value: float
    is_significant: bool
    confidence_interval_lower: float
    confidence_interval_upper: float
    confidence_level: float
    interpretation: str

class CorrelationInterpreter:
    """Interprets the magnitude of a correlation coefficient."""

    _THRESHOLDS: list[tuple[float, str]] = [
        (0.9, "very strong"),
        (0.7, "strong"),
        (0.5, "moderate"),
        (0.3, "weak"),
        (0.0, "negligible"),
    ]

    def interpret(self, coefficient: float, is_significant: bool) -> str:
        """Build interpretation string for a correlation coefficient.

        Args:
            coefficient: Correlation coefficient in [-1, 1].
            is_significant: Whether the correlation is statistically significant.

        Returns:
            Descriptive interpretation string.
        """
        direction = "positive" if coefficient >= 0 else "negative"
        abs_coeff = abs(coefficient)
        magnitude = "negligible"

        for threshold, label in self._THRESHOLDS:
            if abs_coeff >= threshold:
                magnitude = label
                break

        significance_label = "significant" if is_significant else "not significant"
        return f"{magnitude} {direction} correlation ({significance_label})"

class FisherZTransformer:
    """Fisher Z-transformation for correlation confidence intervals.

    Transforms r → z = atanh(r), computes CI in z-space, then
    back-transforms. Valid for |r| < 1.
    """

    def confidence_interval(
        self, r: float, n: int, confidence_level: float
    ) -> tuple[float, float]:
        """Calculate CI for a correlation coefficient via Fisher Z.

        Args:
            r: Sample correlation coefficient.
            n: Number of paired observations.
            confidence_level: Confidence level (e.g., 0.95).

        Returns:
            Tuple (lower_bound, upper_bound) for the correlation.

        Raises:
            ValueError: If n < 4 or |r| >= 1.
        """
        if n < 4:
            raise ValueError(
                f"Fisher Z confidence interval requires n ≥ 4. Got n={n}."
            )
        if abs(r) >= 1.0:
            raise ValueError(
                f"Correlation must be strictly in (-1, 1) for Fisher Z. Got r={r}."
            )

        alpha = 1.0 - confidence_level
        z_critical = float(stats.norm.ppf(1 - alpha / 2))
        z_r = float(np.arctanh(r))
        se = 1.0 / np.sqrt(n - 3)

        lower = float(np.tanh(z_r - z_critical * se))
        upper = float(np.tanh(z_r + z_critical * se))
        return lower, upper

class CorrelationSignificanceCalculator:
    """Computes Pearson or Spearman correlation with significance and CI.

    Workflow:
        calculator = CorrelationSignificanceCalculator()
        result = calculator.calculate(
            x=df["feature"],
            y=df["target"],
            method="pearson",        # "pearson" | "spearman"
            significance_level=0.05,
            confidence_level=0.95,
        )
    """

    _VALID_METHODS: frozenset[str] = frozenset({"pearson", "spearman"})
    _MINIMUM_SAMPLE_SIZE: int = 4

    def __init__(self) -> None:
        self._interpreter = CorrelationInterpreter()
        self._fisher = FisherZTransformer()

    def calculate(
        self,
        x: np.ndarray,
        y: np.ndarray,
        method: str = "pearson",
        significance_level: float = 0.05,
        confidence_level: float = 0.95,
    ) -> dict:
        """Calculate correlation with significance test and confidence interval.

        Args:
            x: First numerical array.
            y: Second numerical array (same length as x).
            method: 'pearson' or 'spearman'.
            significance_level: Alpha threshold.
            confidence_level: CI confidence level (e.g., 0.95).

        Returns:
            Dictionary with coefficient, p-value, CI, and interpretation.

        Raises:
            ValueError: If inputs are invalid.
        """
        if method not in self._VALID_METHODS:
            raise ValueError(
                f"method must be one of {self._VALID_METHODS}. Got '{method}'."
            )
        if len(x) != len(y):
            raise ValueError(
                f"x and y must have equal length. Got {len(x)} and {len(y)}."
            )
        if len(x) < self._MINIMUM_SAMPLE_SIZE:
            raise ValueError(
                f"At least {self._MINIMUM_SAMPLE_SIZE} observations required. "
                f"Got {len(x)}."
            )

        if method == "pearson":
            coefficient, p_value = stats.pearsonr(x, y)
        else:
            coefficient, p_value = stats.spearmanr(x, y)

        r = float(coefficient)
        p = float(p_value)
        is_significant = p < significance_level
        lower, upper = self._fisher.confidence_interval(r, len(x), confidence_level)

        return {
            "method": method,
            "coefficient": r,
            "p_value": p,
            "is_significant": is_significant,
            "significance_level": significance_level,
            "confidence_interval": {
                "lower": lower,
                "upper": upper,
                "confidence_level": confidence_level,
            },
            "interpretation": self._interpreter.interpret(r, is_significant),
            "n": len(x),
        }
