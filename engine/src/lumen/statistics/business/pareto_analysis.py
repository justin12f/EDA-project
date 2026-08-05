"""Pareto (80/20) analysis for any metric-entity pair."""

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
class ParetoRecord:
    """Immutable Pareto record for a single entity."""

    entity: str
    value: float
    rank: int
    cumulative_value: float
    cumulative_share: float
    entity_share: float
    is_vital_few: bool

class ParetoThresholdFinder:
    """Finds the minimum entity count that accounts for target cumulative share.

    The '80/20 rule' is a special case: this class generalizes to any
    target share (e.g., 90/10, 70/30).
    """

    def find(
        self, cumulative_shares: np.ndarray, target_share: float
    ) -> int:
        """Find the index where cumulative share first exceeds target.

        Args:
            cumulative_shares: Sorted cumulative share array (ascending by value).
            target_share: Target cumulative share (e.g., 0.8 for 80%).

        Returns:
            Index (0-based) of the first entity that pushes cumulative share
            above target.
        """
        for i, share in enumerate(cumulative_shares):
            if share >= target_share:
                return i
        return len(cumulative_shares) - 1

class ParetoConcentrationCalculator:
    """Computes Gini coefficient as a concentration measure for the distribution.

    Gini = 0: perfect equality (all entities contribute equally).
    Gini = 1: perfect concentration (one entity produces all value).
    """

    def calculate(self, values: np.ndarray) -> float:
        """Compute Gini coefficient from sorted values.

        Args:
            values: Numeric array of entity values (unsorted).

        Returns:
            Gini coefficient in [0, 1].
        """
        sorted_vals = np.sort(values)
        n = len(sorted_vals)
        cumsum = np.cumsum(sorted_vals)
        total = cumsum[-1]
        if total == 0:
            return 0.0
        gini = (2 * np.sum(np.arange(1, n + 1) * sorted_vals) / (n * total)) - (n + 1) / n
        return round(float(gini), 6)

class ParetoAnalysisCalculator:
    """Pareto (80/20) analysis with concentration metrics.

    Workflow:
        calculator = ParetoAnalysisCalculator()
        result = calculator.calculate(
            data_frame=df,
            entity_column="product_sku",
            value_column="revenue",
            target_share=0.8,     # optional, default 80%
        )
    """

    _MINIMUM_ENTITIES: int = 2

    def __init__(self) -> None:
        self._threshold_finder = ParetoThresholdFinder()
        self._gini_calculator = ParetoConcentrationCalculator()

    def calculate(
        self,
        data_frame: pd.DataFrame,
        entity_column: str,
        value_column: str,
        target_share: float = 0.8,
    ) -> dict:
        """Run Pareto analysis.

        Args:
            data_frame: Source DataFrame.
            entity_column: Column identifying entities (products, customers, etc.).
            value_column: Numeric metric column (revenue, frequency, etc.).
            target_share: Cumulative share threshold for the 'vital few'.

        Returns:
            Dict with ranked entities, Pareto threshold, and Gini coefficient.

        Raises:
            KeyError: If columns are not found.
            ValueError: If data is insufficient or target_share is invalid.
        """
        for col in (entity_column, value_column):
            if col not in data_frame.columns:
                raise KeyError(f"Column '{col}' not found in DataFrame.")
        if not 0.0 < target_share < 1.0:
            raise ValueError(
                f"target_share must be in (0, 1). Got {target_share}."
            )

        aggregated = (
            data_frame.groupby(entity_column)[value_column]
            .sum()
            .reset_index()
            .sort_values(value_column, ascending=False)
            .reset_index(drop=True)
        )

        if len(aggregated) < self._MINIMUM_ENTITIES:
            raise ValueError(
                f"At least {self._MINIMUM_ENTITIES} entities required. "
                f"Got {len(aggregated)}."
            )

        values = aggregated[value_column].to_numpy(dtype=float)
        total = float(values.sum())

        if total == 0:
            raise ValueError(
                f"Total value in '{value_column}' is zero. Cannot compute shares."
            )

        cumulative_values = np.cumsum(values)
        cumulative_shares = cumulative_values / total
        entity_shares = values / total

        threshold_idx = self._threshold_finder.find(cumulative_shares, target_share)
        gini = self._gini_calculator.calculate(values)

        n_vital_few = threshold_idx + 1
        vital_few_entity_share = round(n_vital_few / len(values), 4)
        actual_value_share = round(float(cumulative_shares[threshold_idx]), 4)

        records: list[ParetoRecord] = [
            ParetoRecord(
                entity=str(aggregated[entity_column].iloc[i]),
                value=round(float(values[i]), 4),
                rank=i + 1,
                cumulative_value=round(float(cumulative_values[i]), 4),
                cumulative_share=round(float(cumulative_shares[i]), 6),
                entity_share=round(float(entity_shares[i]), 6),
                is_vital_few=i <= threshold_idx,
            )
            for i in range(len(values))
        ]

        return {
            "entities": [
                {
                    "entity": r.entity,
                    "value": r.value,
                    "rank": r.rank,
                    "cumulative_value": r.cumulative_value,
                    "cumulative_share": r.cumulative_share,
                    "entity_share": r.entity_share,
                    "is_vital_few": r.is_vital_few,
                }
                for r in records
            ],
            "pareto_summary": {
                "n_vital_few": n_vital_few,
                "n_total_entities": len(records),
                "vital_few_entity_share": vital_few_entity_share,
                "vital_few_value_share": actual_value_share,
                "target_share": target_share,
                "interpretation": (
                    f"{vital_few_entity_share*100:.1f}% of entities "
                    f"({n_vital_few} of {len(records)}) account for "
                    f"{actual_value_share*100:.1f}% of total {value_column}."
                ),
            },
            "concentration_metrics": {
                "gini_coefficient": gini,
                "interpretation": (
                    "Highly concentrated" if gini > 0.7
                    else "Moderately concentrated" if gini > 0.4
                    else "Relatively uniform"
                ),
            },
            "total_value": round(total, 4),
            "entity_column": entity_column,
            "value_column": value_column,
        }
