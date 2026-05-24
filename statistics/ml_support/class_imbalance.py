"""Class imbalance detection and resampling strategy recommendation."""

# #[AI_CONTEXT_START]
# - CONFIGURACIÓN DE FACTORY: Registrar este calculator en `MlSupportStatisticsFactory` (backends pandas | polars | spark) y exponerlo mediante `StatisticsInyeccionDependency`, inyectada por la Factory Maestra de Agentes junto a las demás fábricas de dominio.
# - ABSTRACCIÓN DEL DATO: Mutar constructores y `analyze`/`compute` para recibir el contenedor abstracto del backend (`pd.DataFrame`, `pl.DataFrame`/`pl.LazyFrame`, `pyspark.sql.DataFrame`) inyectado por la factory; eliminar `np.ndarray`/`pd.Series` sueltos en firmas públicas.
# - REFACTOR NATIVO: Resolver métricas con expresiones 100 % nativas del backend activo (Polars: `.select`/`.group_by`/`.agg` sin `.collect()` salvo materialización acordada; PySpark: `pyspark.sql.functions` y ventanas distribuidas; Pandas: operaciones vectorizadas). No convertir a NumPy/Pandas desde backends no-pandas.
# #[AI_CONTEXT_END]
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

@dataclass(frozen=True)
class ClassMetrics:
    """Immutable metrics for a single class."""

    label: str
    count: int
    proportion: float
    is_minority: bool

class GiniImpurityCalculator:
    """Computes Gini impurity as a class imbalance severity measure.

    Gini = 1 - Σpᵢ²
    Gini = 0 → perfectly pure (all one class)
    Gini = 1 - 1/k → perfectly balanced (k classes)
    """

    def calculate(self, proportions: np.ndarray) -> float:
        """Compute Gini impurity.

        Args:
            proportions: Class proportion array summing to 1.

        Returns:
            Gini impurity value in [0, 1).
        """
        return float(1.0 - np.sum(proportions ** 2))

class ImbalanceRatioCalculator:
    """Computes the imbalance ratio: majority_count / minority_count.

    IR = 1 → perfectly balanced.
    IR > 3 → mild imbalance.
    IR > 10 → severe imbalance.
    """

    def calculate(self, class_counts: np.ndarray) -> float:
        """Compute imbalance ratio.

        Args:
            class_counts: Array of per-class observation counts.

        Returns:
            Imbalance ratio (majority / minority).
        """
        minority = float(class_counts.min())
        majority = float(class_counts.max())
        return majority / minority if minority > 0 else float("inf")

class StrategyAdvisor:
    """Recommends a resampling or algorithmic strategy based on imbalance ratio.

    Strategy tiers:
        IR < 3:   No action needed — standard training is fine.
        IR 3-10:  Class weighting or mild oversampling (SMOTE).
        IR 10-50: SMOTE + Tomek links or ADASYN.
        IR > 50:  Anomaly detection framing or aggressive oversampling.
    """

    _STRATEGY_TIERS: list[tuple[float, str]] = [
        (50.0, "anomaly_detection_or_aggressive_oversampling"),
        (10.0, "smote_tomek_or_adasyn"),
        (3.0,  "class_weighting_or_smote"),
        (0.0,  "no_resampling_needed"),
    ]

    def advise(self, imbalance_ratio: float) -> str:
        """Return recommended strategy string.

        Args:
            imbalance_ratio: Majority / minority class count ratio.

        Returns:
            Strategy recommendation string.
        """
        for threshold, strategy in self._STRATEGY_TIERS:
            if imbalance_ratio >= threshold:
                return strategy
        return "no_resampling_needed"

class ClassImbalanceCalculator:
    """Detects and quantifies class imbalance with resampling recommendations.

    Workflow:
        calculator = ClassImbalanceCalculator()
        result = calculator.calculate(
            series=df["churn"],
            minority_threshold=0.3,  # optional
        )
    """

    def __init__(self) -> None:
        self._gini = GiniImpurityCalculator()
        self._ir_calculator = ImbalanceRatioCalculator()
        self._advisor = StrategyAdvisor()

    def calculate(
        self,
        series: pd.Series,
        minority_threshold: float = 0.3,
    ) -> dict:
        """Analyse class distribution and recommend resampling strategy.

        Args:
            series: Categorical target Series.
            minority_threshold: Proportion below which a class is minority.

        Returns:
            Dict with class metrics, imbalance ratio, Gini, and recommendation.

        Raises:
            ValueError: If series has fewer than 2 unique values or is empty.
        """
        if not 0.0 < minority_threshold < 1.0:
            raise ValueError(
                f"minority_threshold must be in (0, 1). Got {minority_threshold}."
            )

        clean = series.dropna()
        if len(clean) == 0:
            raise ValueError("Target series is empty after dropping NaN values.")

        value_counts = clean.value_counts()
        n_classes = len(value_counts)

        if n_classes < 2:
            raise ValueError(
                f"Class imbalance analysis requires at least 2 classes. "
                f"Got {n_classes}."
            )

        n_total = len(clean)
        counts = value_counts.to_numpy()
        proportions = counts / n_total

        gini = self._gini.calculate(proportions)
        ir = self._ir_calculator.calculate(counts)
        strategy = self._advisor.advise(ir)

        max_balanced_gini = 1.0 - 1.0 / n_classes
        gini_normalized = gini / max_balanced_gini if max_balanced_gini > 0 else 1.0

        classes: list[ClassMetrics] = [
            ClassMetrics(
                label=str(label),
                count=int(count),
                proportion=round(float(prop), 6),
                is_minority=float(prop) < minority_threshold,
            )
            for label, count, prop in zip(
                value_counts.index, counts, proportions
            )
        ]

        minority_classes = [c.label for c in classes if c.is_minority]
        majority_classes = [c.label for c in classes if not c.is_minority]

        return {
            "classes": [
                {
                    "label": c.label,
                    "count": c.count,
                    "proportion": c.proportion,
                    "is_minority": c.is_minority,
                }
                for c in classes
            ],
            "n_classes": n_classes,
            "n_observations": n_total,
            "minority_classes": minority_classes,
            "majority_classes": majority_classes,
            "imbalance_ratio": round(ir, 4),
            "gini_impurity": round(gini, 6),
            "gini_normalized": round(gini_normalized, 6),
            "is_imbalanced": ir > 3.0,
            "recommended_strategy": strategy,
            "minority_threshold": minority_threshold,
        }
