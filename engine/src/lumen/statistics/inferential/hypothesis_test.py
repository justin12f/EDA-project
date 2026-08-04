"""Hypothesis testing module for one-sample and two-sample tests."""

# #[AI_CONTEXT_START]
# - CONFIGURACIÓN DE FACTORY: Registrar este calculator en `InferentialStatisticsFactory` (backends pandas | polars | spark) y exponerlo mediante `StatisticsInyeccionDependency`, inyectada por la Factory Maestra de Agentes junto a las demás fábricas de dominio.
# - ABSTRACCIÓN DEL DATO: Mutar constructores y `analyze`/`compute` para recibir el contenedor abstracto del backend (`pd.DataFrame`, `pl.DataFrame`/`pl.LazyFrame`, `pyspark.sql.DataFrame`) inyectado por la factory; eliminar `np.ndarray`/`pd.Series` sueltos en firmas públicas.
# - REFACTOR NATIVO: Resolver métricas con expresiones 100 % nativas del backend activo (Polars: `.select`/`.group_by`/`.agg` sin `.collect()` salvo materialización acordada; PySpark: `pyspark.sql.functions` y ventanas distribuidas; Pandas: operaciones vectorizadas). No convertir a NumPy/Pandas desde backends no-pandas.
# #[AI_CONTEXT_END]
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import stats

@dataclass(frozen=True)
class HypothesisTestResult:
    """Immutable result for a single hypothesis test."""

    test_name: str
    statistic: float
    p_value: float
    reject_null: bool
    significance_level: float
    alternative: str
    interpretation: str

class HypothesisInterpreter:
    """Generates a human-readable interpretation of a hypothesis test result."""

    def interpret(
        self,
        test_name: str,
        reject_null: bool,
        p_value: float,
        significance_level: float,
    ) -> str:
        """Build interpretation string.

        Args:
            test_name: Name of the test applied.
            reject_null: Whether H0 was rejected.
            p_value: Computed p-value.
            significance_level: Alpha threshold used.

        Returns:
            Human-readable interpretation string.
        """
        verdict = "rejected" if reject_null else "not rejected"
        return (
            f"{test_name}: H0 is {verdict} "
            f"(p={p_value:.6f}, α={significance_level}). "
            f"""{'Statistically significant difference detected.'
               if reject_null
               else 'No statistically significant difference detected.'}"""
        )

class BaseHypothesisTest(ABC):
    """Abstract base for all hypothesis tests."""

    @abstractmethod
    def test(
        self,
        group_a: np.ndarray,
        group_b: np.ndarray | None,
        significance_level: float,
        alternative: str,
    ) -> HypothesisTestResult:
        """Execute the hypothesis test.

        Args:
            group_a: Primary data array.
            group_b: Secondary data array (None for one-sample tests).
            significance_level: Alpha threshold.
            alternative: One of 'two-sided', 'less', 'greater'.

        Returns:
            HypothesisTestResult with all test details.
        """

class TTest(BaseHypothesisTest):
    """Student's T-test.

    Supports:
        - One-sample: tests if mean differs from a population mean (group_b=None,
          pass popmean via kwargs workaround through group_b as scalar array).
        - Two-sample independent: tests if two group means differ.

    Assumes approximately normal distributions or large samples (n > 30).
    """

    def test(
        self,
        group_a: np.ndarray,
        group_b: np.ndarray | None,
        significance_level: float,
        alternative: str,
    ) -> HypothesisTestResult:
        if group_b is None:
            statistic, p_value = stats.ttest_1samp(
                group_a, popmean=0.0, alternative=alternative
            )
            test_name = "t_test_one_sample"
        else:
            statistic, p_value = stats.ttest_ind(
                group_a, group_b, alternative=alternative, equal_var=False
            )
            test_name = "t_test_two_sample_welch"

        reject_null = float(p_value) < significance_level
        interpretation = HypothesisInterpreter().interpret(
            test_name, reject_null, float(p_value), significance_level
        )

        return HypothesisTestResult(
            test_name=test_name,
            statistic=float(statistic),
            p_value=float(p_value),
            reject_null=reject_null,
            significance_level=significance_level,
            alternative=alternative,
            interpretation=interpretation,
        )

class MannWhitneyTest(BaseHypothesisTest):
    """Mann-Whitney U test — non-parametric alternative to independent T-test.

    Does not assume normality. Tests if one distribution is stochastically
    greater than the other. Requires at least 8 observations per group
    for reliable asymptotic approximation.
    """

    _MINIMUM_SAMPLE_SIZE: int = 8

    def test(
        self,
        group_a: np.ndarray,
        group_b: np.ndarray | None,
        significance_level: float,
        alternative: str,
    ) -> HypothesisTestResult:
        if group_b is None:
            raise ValueError(
                "MannWhitneyTest requires two groups. group_b cannot be None."
            )
        if len(group_a) < self._MINIMUM_SAMPLE_SIZE or len(group_b) < self._MINIMUM_SAMPLE_SIZE:
            raise ValueError(
                f"Both groups need at least {self._MINIMUM_SAMPLE_SIZE} "
                f"observations for reliable Mann-Whitney results."
            )

        statistic, p_value = stats.mannwhitneyu(
            group_a, group_b, alternative=alternative
        )
        reject_null = float(p_value) < significance_level
        interpretation = HypothesisInterpreter().interpret(
            "mann_whitney_u", reject_null, float(p_value), significance_level
        )

        return HypothesisTestResult(
            test_name="mann_whitney_u",
            statistic=float(statistic),
            p_value=float(p_value),
            reject_null=reject_null,
            significance_level=significance_level,
            alternative=alternative,
            interpretation=interpretation,
        )

class WilcoxonTest(BaseHypothesisTest):
    """Wilcoxon signed-rank test — non-parametric test for paired samples.

    Tests if the median difference between paired observations is zero.
    Requires both groups to have the same length.
    """

    def test(
        self,
        group_a: np.ndarray,
        group_b: np.ndarray | None,
        significance_level: float,
        alternative: str,
    ) -> HypothesisTestResult:
        if group_b is None:
            raise ValueError(
                "WilcoxonTest requires two paired groups. group_b cannot be None."
            )
        if len(group_a) != len(group_b):
            raise ValueError(
                f"Wilcoxon test requires equal-length arrays. "
                f"Got {len(group_a)} and {len(group_b)}."
            )

        statistic, p_value = stats.wilcoxon(
            group_a, group_b, alternative=alternative
        )
        reject_null = float(p_value) < significance_level
        interpretation = HypothesisInterpreter().interpret(
            "wilcoxon_signed_rank", reject_null, float(p_value), significance_level
        )

        return HypothesisTestResult(
            test_name="wilcoxon_signed_rank",
            statistic=float(statistic),
            p_value=float(p_value),
            reject_null=reject_null,
            significance_level=significance_level,
            alternative=alternative,
            interpretation=interpretation,
        )

_TEST_REGISTRY: dict[str, BaseHypothesisTest] = {
    "t_test": TTest(),
    "mann_whitney": MannWhitneyTest(),
    "wilcoxon": WilcoxonTest(),
}

class HypothesisTestSuite:
    """Orchestrates hypothesis testing across registered test strategies.

    Workflow:
        suite = HypothesisTestSuite()
        result = suite.run(
            group_a=data_a,
            group_b=data_b,
            test="t_test",              # "t_test" | "mann_whitney" | "wilcoxon"
            significance_level=0.05,
            alternative="two-sided",    # "two-sided" | "less" | "greater"
        )
    """

    _VALID_ALTERNATIVES: frozenset[str] = frozenset(
        {"two-sided", "less", "greater"}
    )

    def run(
        self,
        group_a: np.ndarray,
        group_b: np.ndarray | None,
        test: str = "t_test",
        significance_level: float = 0.05,
        alternative: str = "two-sided",
    ) -> dict:
        """Run the specified hypothesis test.

        Args:
            group_a: Primary numerical array.
            group_b: Secondary array (None for one-sample tests).
            test: Test key. One of 't_test', 'mann_whitney', 'wilcoxon'.
            significance_level: Alpha threshold in (0, 1).
            alternative: Hypothesis direction.

        Returns:
            Dictionary with test result fields.

        Raises:
            KeyError: If test key is not registered.
            ValueError: If significance_level or alternative are invalid.
        """
        if test not in _TEST_REGISTRY:
            raise KeyError(
                f"Test '{test}' not found. "
                f"Available: {list(_TEST_REGISTRY.keys())}"
            )
        if not 0.0 < significance_level < 1.0:
            raise ValueError(
                f"significance_level must be in (0, 1). Got {significance_level}."
            )
        if alternative not in self._VALID_ALTERNATIVES:
            raise ValueError(
                f"alternative must be one of {self._VALID_ALTERNATIVES}. "
                f"Got '{alternative}'."
            )

        result = _TEST_REGISTRY[test].test(
            group_a, group_b, significance_level, alternative
        )

        return {
            "test_name": result.test_name,
            "statistic": result.statistic,
            "p_value": result.p_value,
            "reject_null": result.reject_null,
            "significance_level": result.significance_level,
            "alternative": result.alternative,
            "interpretation": result.interpretation,
        }
