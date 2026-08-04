"""Confidence interval calculators for means, proportions and differences."""

# #[AI_CONTEXT_START]
# - CONFIGURACIÓN DE FACTORY: Registrar este calculator en `InferentialStatisticsFactory` (backends pandas | polars | spark) y exponerlo mediante `StatisticsInyeccionDependency`, inyectada por la Factory Maestra de Agentes junto a las demás fábricas de dominio.
# - ABSTRACCIÓN DEL DATO: Mutar constructores y `analyze`/`compute` para recibir el contenedor abstracto del backend (`pd.DataFrame`, `pl.DataFrame`/`pl.LazyFrame`, `pyspark.sql.DataFrame`) inyectado por la factory; eliminar `np.ndarray`/`pd.Series` sueltos en firmas públicas.
# - REFACTOR NATIVO: Resolver métricas con expresiones 100 % nativas del backend activo (Polars: `.select`/`.group_by`/`.agg` sin `.collect()` salvo materialización acordada; PySpark: `pyspark.sql.functions` y ventanas distribuidas; Pandas: operaciones vectorizadas). No convertir a NumPy/Pandas desde backends no-pandas.
# #[AI_CONTEXT_END]
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from scipy import stats

class BaseConfidenceInterval(ABC):
    """Abstract base for all confidence interval calculators."""

    @abstractmethod
    def calculate(self, confidence_level: float, **kwargs) -> dict:
        """Calculate the confidence interval.

        Args:
            confidence_level: Desired confidence level (e.g., 0.95).
            **kwargs: Calculator-specific parameters.

        Returns:
            Dict with lower, upper, center, margin_of_error and metadata.
        """

class MeanConfidenceInterval(BaseConfidenceInterval):
    """T-distribution based CI for a population mean.

    Uses t-distribution (not z) regardless of sample size, which is more
    conservative and correct for unknown population variance.
    """

    def calculate(self, confidence_level: float, data: np.ndarray) -> dict:
        """Calculate CI for the mean of a numerical array.

        Args:
            confidence_level: Desired confidence level.
            data: 1D numerical array (length ≥ 2).

        Returns:
            CI dict with lower, upper, mean, std, se, n, df.

        Raises:
            ValueError: If data has fewer than 2 elements.
        """
        if len(data) < 2:
            raise ValueError(
                f"At least 2 observations required for mean CI. Got {len(data)}."
            )

        n = len(data)
        mean = float(data.mean())
        se = float(stats.sem(data))
        df = n - 1
        alpha = 1.0 - confidence_level
        t_critical = float(stats.t.ppf(1 - alpha / 2, df=df))
        margin = t_critical * se

        return {
            "ci_type": "mean",
            "lower": mean - margin,
            "upper": mean + margin,
            "center": mean,
            "margin_of_error": margin,
            "standard_error": se,
            "t_critical": t_critical,
            "df": df,
            "n": n,
            "confidence_level": confidence_level,
        }

class ProportionConfidenceInterval(BaseConfidenceInterval):
    """Wilson score CI for a population proportion.

    Preferred over the normal approximation (Wald) because it remains
    valid for proportions near 0 or 1 and small samples.
    """

    def calculate(
        self,
        confidence_level: float,
        n_successes: int,
        n_total: int,
    ) -> dict:
        """Calculate Wilson score CI for a proportion.

        Args:
            confidence_level: Desired confidence level.
            n_successes: Number of successes (events).
            n_total: Total number of observations.

        Returns:
            CI dict with lower, upper, proportion, n.

        Raises:
            ValueError: If n_total < 1 or n_successes > n_total.
        """
        if n_total < 1:
            raise ValueError(f"n_total must be ≥ 1. Got {n_total}.")
        if n_successes > n_total:
            raise ValueError(
                f"n_successes ({n_successes}) cannot exceed n_total ({n_total})."
            )
        if n_successes < 0:
            raise ValueError(f"n_successes must be ≥ 0. Got {n_successes}.")

        alpha = 1.0 - confidence_level
        z = float(stats.norm.ppf(1 - alpha / 2))
        p_hat = n_successes / n_total
        denominator = 1 + z**2 / n_total
        center = (p_hat + z**2 / (2 * n_total)) / denominator
        spread = z * float(np.sqrt(p_hat * (1 - p_hat) / n_total + z**2 / (4 * n_total**2))) / denominator

        return {
            "ci_type": "proportion_wilson",
            "lower": max(0.0, center - spread),
            "upper": min(1.0, center + spread),
            "center": center,
            "observed_proportion": p_hat,
            "n_successes": n_successes,
            "n_total": n_total,
            "z_critical": z,
            "confidence_level": confidence_level,
        }

class MeanDifferenceConfidenceInterval(BaseConfidenceInterval):
    """Welch's CI for the difference between two independent group means.

    Uses Welch-Satterthwaite degrees of freedom approximation for
    groups with unequal variances and/or unequal sizes.
    """

    def calculate(
        self,
        confidence_level: float,
        group_a: np.ndarray,
        group_b: np.ndarray,
    ) -> dict:
        """Calculate CI for the difference of two independent means.

        Args:
            confidence_level: Desired confidence level.
            group_a: Numerical array for group A.
            group_b: Numerical array for group B.

        Returns:
            CI dict with lower, upper, difference, and group statistics.

        Raises:
            ValueError: If either group has fewer than 2 observations.
        """
        if len(group_a) < 2:
            raise ValueError(
                f"group_a needs at least 2 observations. Got {len(group_a)}."
            )
        if len(group_b) < 2:
            raise ValueError(
                f"group_b needs at least 2 observations. Got {len(group_b)}."
            )

        n_a, n_b = len(group_a), len(group_b)
        mean_a, mean_b = float(group_a.mean()), float(group_b.mean())
        var_a = float(group_a.var(ddof=1))
        var_b = float(group_b.var(ddof=1))

        se = float(np.sqrt(var_a / n_a + var_b / n_b))
        df = self._welch_satterthwaite_df(var_a, var_b, n_a, n_b)

        alpha = 1.0 - confidence_level
        t_critical = float(stats.t.ppf(1 - alpha / 2, df=df))
        diff = mean_a - mean_b
        margin = t_critical * se

        return {
            "ci_type": "mean_difference_welch",
            "lower": diff - margin,
            "upper": diff + margin,
            "center": diff,
            "margin_of_error": margin,
            "standard_error": se,
            "t_critical": t_critical,
            "df": round(df, 4),
            "group_a": {"mean": mean_a, "var": var_a, "n": n_a},
            "group_b": {"mean": mean_b, "var": var_b, "n": n_b},
            "confidence_level": confidence_level,
        }

    def _welch_satterthwaite_df(
        self, var_a: float, var_b: float, n_a: int, n_b: int
    ) -> float:
        """Welch-Satterthwaite approximation for effective degrees of freedom.

        Args:
            var_a: Variance of group A.
            var_b: Variance of group B.
            n_a: Size of group A.
            n_b: Size of group B.

        Returns:
            Approximate degrees of freedom.
        """
        numerator = (var_a / n_a + var_b / n_b) ** 2
        denominator = (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
        return numerator / denominator if denominator > 0 else 1.0

_CI_REGISTRY: dict[str, BaseConfidenceInterval] = {
    "mean": MeanConfidenceInterval(),
    "proportion": ProportionConfidenceInterval(),
    "mean_difference": MeanDifferenceConfidenceInterval(),
}

class ConfidenceIntervalCalculator:
    """Unified entry point for all confidence interval types.

    Workflow:
        calculator = ConfidenceIntervalCalculator()

        # Mean CI
        result = calculator.calculate("mean", data=arr, confidence_level=0.95)

        # Proportion CI
        result = calculator.calculate(
            "proportion", n_successes=45, n_total=200, confidence_level=0.95
        )

        # Difference of means CI
        result = calculator.calculate(
            "mean_difference", group_a=arr_a, group_b=arr_b, confidence_level=0.95
        )
    """

    def calculate(self, ci_type: str, confidence_level: float = 0.95, **kwargs) -> dict:
        """Dispatch to the appropriate CI calculator.

        Args:
            ci_type: One of 'mean', 'proportion', 'mean_difference'.
            confidence_level: Desired confidence level.
            **kwargs: Arguments forwarded to the specific calculator.

        Returns:
            CI result dictionary.

        Raises:
            KeyError: If ci_type is not registered.
            ValueError: If confidence_level is not in (0, 1).
        """
        if ci_type not in _CI_REGISTRY:
            raise KeyError(
                f"ci_type '{ci_type}' not found. "
                f"Available: {list(_CI_REGISTRY.keys())}"
            )
        if not 0.0 < confidence_level < 1.0:
            raise ValueError(
                f"confidence_level must be in (0, 1). Got {confidence_level}."
            )

        return _CI_REGISTRY[ci_type].calculate(
            confidence_level=confidence_level, **kwargs
        )
