"""ANOVA testing module with post-hoc analysis."""

# #[AI_CONTEXT_START]
# - CONFIGURACIÓN DE FACTORY: Registrar este calculator en `InferentialStatisticsFactory` (backends pandas | polars | spark) y exponerlo mediante `StatisticsInyeccionDependency`, inyectada por la Factory Maestra de Agentes junto a las demás fábricas de dominio.
# - ABSTRACCIÓN DEL DATO: Mutar constructores y `analyze`/`compute` para recibir el contenedor abstracto del backend (`pd.DataFrame`, `pl.DataFrame`/`pl.LazyFrame`, `pyspark.sql.DataFrame`) inyectado por la factory; eliminar `np.ndarray`/`pd.Series` sueltos en firmas públicas.
# - REFACTOR NATIVO: Resolver métricas con expresiones 100 % nativas del backend activo (Polars: `.select`/`.group_by`/`.agg` sin `.collect()` salvo materialización acordada; PySpark: `pyspark.sql.functions` y ventanas distribuidas; Pandas: operaciones vectorizadas). No convertir a NumPy/Pandas desde backends no-pandas.
# #[AI_CONTEXT_END]
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
from itertools import combinations

@dataclass(frozen=True)
class TukeyPairResult:
    """Immutable result for a single Tukey HSD pairwise comparison."""

    group_i: str
    group_j: str
    mean_difference: float
    p_value: float
    significant: bool

class TukeyHSDPostHoc:
    """Tukey HSD pairwise comparison using the studentized range distribution.

    Computes all pairwise mean differences and adjusts for family-wise error rate.
    Valid only after a significant one-way ANOVA result.
    """

    def calculate(
        self,
        groups: dict[str, np.ndarray],
        significance_level: float,
    ) -> list[dict]:
        """Run pairwise Tukey HSD comparisons.

        Args:
            groups: Dict mapping group name to its data array.
            significance_level: Alpha threshold for significance.

        Returns:
            List of pairwise comparison result dicts.

        Raises:
            ValueError: If fewer than 2 groups are provided.
        """
        if len(groups) < 2:
            raise ValueError(
                "Tukey HSD requires at least 2 groups for pairwise comparison."
            )

        all_data = np.concatenate(list(groups.values()))
        grand_n = len(all_data)
        k = len(groups)
        df_within = grand_n - k
        ms_within = self._calculate_ms_within(groups)
        results: list[dict] = []

        for (name_i, data_i), (name_j, data_j) in combinations(groups.items(), 2):
            n_harmonic = self._harmonic_mean_n(len(data_i), len(data_j))
            se = float(np.sqrt(ms_within / n_harmonic))

            if se == 0.0:
                continue

            q_statistic = abs(float(data_i.mean()) - float(data_j.mean())) / se
            p_value = float(
                stats.studentized_range.sf(q_statistic, k, df_within)
            )

            results.append(
                TukeyPairResult(
                    group_i=name_i,
                    group_j=name_j,
                    mean_difference=float(data_i.mean()) - float(data_j.mean()),
                    p_value=p_value,
                    significant=p_value < significance_level,
                ).__dict__
            )

        return results

    def _calculate_ms_within(self, groups: dict[str, np.ndarray]) -> float:
        """Calculate mean square within groups (pooled variance).

        Args:
            groups: Dict of group name to data array.

        Returns:
            MS_within as a float.
        """
        ss_within = sum(
            float(np.sum((data - data.mean()) ** 2)) for data in groups.values()
        )
        df_within = sum(len(data) - 1 for data in groups.values())
        return ss_within / df_within if df_within > 0 else 0.0

    def _harmonic_mean_n(self, n_i: int, n_j: int) -> float:
        """Harmonic mean of two sample sizes for unequal-n Tukey.

        Args:
            n_i: Sample size of group i.
            n_j: Sample size of group j.

        Returns:
            Harmonic mean of n_i and n_j.
        """
        return (2 * n_i * n_j) / (n_i + n_j)

class OneWayAnovaCalculator:
    """One-way ANOVA with optional Tukey HSD post-hoc test.

    Workflow:
        calculator = OneWayAnovaCalculator()
        result = calculator.calculate(
            groups={"control": arr1, "treatment_a": arr2, "treatment_b": arr3},
            significance_level=0.05,
            run_post_hoc=True,
        )
    """

    _MINIMUM_GROUPS: int = 2
    _MINIMUM_OBSERVATIONS_PER_GROUP: int = 2

    def __init__(self) -> None:
        self._post_hoc = TukeyHSDPostHoc()

    def calculate(
        self,
        groups: dict[str, np.ndarray],
        significance_level: float = 0.05,
        run_post_hoc: bool = True,
    ) -> dict:
        """Run one-way ANOVA.

        Args:
            groups: Dict mapping group name to numerical data array.
            significance_level: Alpha threshold.
            run_post_hoc: If True and ANOVA is significant, runs Tukey HSD.

        Returns:
            Dictionary with F-statistic, p-value, verdict, and optional post-hoc.

        Raises:
            ValueError: If fewer than 2 groups or any group has fewer than
                2 observations.
        """
        if len(groups) < self._MINIMUM_GROUPS:
            raise ValueError(
                f"ANOVA requires at least {self._MINIMUM_GROUPS} groups. "
                f"Got {len(groups)}."
            )

        undersized = [
            name
            for name, data in groups.items()
            if len(data) < self._MINIMUM_OBSERVATIONS_PER_GROUP
        ]
        if undersized:
            raise ValueError(
                f"Groups {undersized} have fewer than "
                f"{self._MINIMUM_OBSERVATIONS_PER_GROUP} observations."
            )

        f_statistic, p_value = stats.f_oneway(*groups.values())
        reject_null = float(p_value) < significance_level

        result: dict = {
            "test_name": "one_way_anova",
            "f_statistic": float(f_statistic),
            "p_value": float(p_value),
            "reject_null": reject_null,
            "significance_level": significance_level,
            "n_groups": len(groups),
            "group_means": {
                name: float(data.mean()) for name, data in groups.items()
            },
            "group_sizes": {name: len(data) for name, data in groups.items()},
            "post_hoc": None,
        }

        if reject_null and run_post_hoc:
            result["post_hoc"] = {
                "method": "tukey_hsd",
                "comparisons": self._post_hoc.calculate(groups, significance_level),
            }

        return result
