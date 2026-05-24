"""Near-zero variance feature detection and filtering."""

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
class FeatureVarianceResult:
    """Immutable variance analysis result for a single feature."""

    feature: str
    variance: float
    std: float
    coefficient_of_variation: float
    unique_ratio: float
    most_frequent_ratio: float
    is_near_zero_variance: bool
    reason: str

class VarianceComputer:
    """Computes raw variance, std, and CV for a single feature array."""

    def compute(self, values: np.ndarray) -> tuple[float, float, float]:
        """Compute variance, std, and coefficient of variation.

        Args:
            values: 1D numerical array (no NaN).

        Returns:
            Tuple (variance, std, cv). CV is inf when mean is zero.
        """
        variance = float(np.var(values, ddof=1))
        std = float(np.std(values, ddof=1))
        mean = float(np.mean(values))
        cv = std / abs(mean) if mean != 0.0 else float("inf")
        return variance, std, cv

class FrequencyRatioComputer:
    """Computes unique value ratio and most-frequent value ratio.

    Both ratios are used alongside variance to identify quasi-constant
    features that have technically non-zero variance but are dominated
    by a single value (e.g., 99% zeros).
    """

    def compute(self, series: pd.Series) -> tuple[float, float]:
        """Compute unique ratio and most-frequent value ratio.

        Args:
            series: Pandas Series (any dtype, NaN already dropped).

        Returns:
            Tuple (unique_ratio, most_frequent_ratio).
        """
        n = len(series)
        if n == 0:
            return 0.0, 1.0

        unique_ratio = float(series.nunique()) / n
        most_frequent_count = int(series.value_counts().iloc[0])
        most_frequent_ratio = most_frequent_count / n

        return unique_ratio, most_frequent_ratio

class NearZeroVarianceClassifier:
    """Classifies a feature as near-zero variance based on configurable thresholds.

    A feature fails if ANY of these conditions holds:
        1. variance < variance_threshold
        2. unique_ratio < unique_ratio_threshold
        3. most_frequent_ratio > frequency_ratio_threshold

    This mirrors the caret::nearZeroVar heuristic from R.
    """

    def classify(
        self,
        variance: float,
        unique_ratio: float,
        most_frequent_ratio: float,
        variance_threshold: float,
        unique_ratio_threshold: float,
        frequency_ratio_threshold: float,
    ) -> tuple[bool, str]:
        """Classify feature and return (is_nzv, reason).

        Args:
            variance: Feature variance.
            unique_ratio: Proportion of unique values.
            most_frequent_ratio: Proportion of the most frequent value.
            variance_threshold: Minimum acceptable variance.
            unique_ratio_threshold: Minimum acceptable unique ratio.
            frequency_ratio_threshold: Maximum acceptable dominant-value ratio.

        Returns:
            Tuple (is_near_zero_variance, reason_string).
        """
        if variance < variance_threshold:
            return True, f"variance={variance:.6f} < threshold={variance_threshold}"
        if unique_ratio < unique_ratio_threshold:
            return True, f"unique_ratio={unique_ratio:.4f} < threshold={unique_ratio_threshold}"
        if most_frequent_ratio > frequency_ratio_threshold:
            return True, (
                f"most_frequent_ratio={most_frequent_ratio:.4f} "
                f"> threshold={frequency_ratio_threshold}"
            )
        return False, "acceptable"

class FeatureVarianceCalculator:
    """Detects near-zero variance features across all numeric columns.

    Workflow:
        calculator = FeatureVarianceCalculator()
        result = calculator.calculate(
            data_frame=df,
            variance_threshold=1e-4,
            unique_ratio_threshold=0.01,
            frequency_ratio_threshold=0.95,
        )
    """

    _MINIMUM_OBSERVATIONS: int = 5

    def __init__(self) -> None:
        self._variance_computer = VarianceComputer()
        self._frequency_computer = FrequencyRatioComputer()
        self._classifier = NearZeroVarianceClassifier()

    def calculate(
        self,
        data_frame: pd.DataFrame,
        variance_threshold: float = 1e-4,
        unique_ratio_threshold: float = 0.01,
        frequency_ratio_threshold: float = 0.95,
    ) -> dict:
        """Analyse all numeric columns for near-zero variance.

        Args:
            data_frame: Source DataFrame (numeric columns only evaluated).
            variance_threshold: Minimum acceptable variance.
            unique_ratio_threshold: Minimum acceptable unique ratio.
            frequency_ratio_threshold: Maximum dominant-value ratio.

        Returns:
            Dict with per-feature results, flagged features, and recommendation.

        Raises:
            ValueError: If thresholds are out of range or data is insufficient.
        """
        if data_frame.shape[0] < self._MINIMUM_OBSERVATIONS:
            raise ValueError(
                f"At least {self._MINIMUM_OBSERVATIONS} observations required. "
                f"Got {data_frame.shape[0]}."
            )
        if not 0.0 <= unique_ratio_threshold <= 1.0:
            raise ValueError(
                f"unique_ratio_threshold must be in [0, 1]. "
                f"Got {unique_ratio_threshold}."
            )
        if not 0.0 <= frequency_ratio_threshold <= 1.0:
            raise ValueError(
                f"frequency_ratio_threshold must be in [0, 1]. "
                f"Got {frequency_ratio_threshold}."
            )

        numeric_df = data_frame.select_dtypes(include=[np.number])

        if numeric_df.empty:
            raise ValueError("No numeric columns found in DataFrame.")

        results: list[FeatureVarianceResult] = []

        for col in numeric_df.columns:
            clean = numeric_df[col].dropna()
            if len(clean) < 2:
                results.append(
                    FeatureVarianceResult(
                        feature=col,
                        variance=0.0,
                        std=0.0,
                        coefficient_of_variation=float("inf"),
                        unique_ratio=0.0,
                        most_frequent_ratio=1.0,
                        is_near_zero_variance=True,
                        reason="insufficient non-null observations",
                    )
                )
                continue

            values = clean.to_numpy(dtype=float)
            variance, std, cv = self._variance_computer.compute(values)
            unique_ratio, most_frequent_ratio = self._frequency_computer.compute(clean)
            is_nzv, reason = self._classifier.classify(
                variance, unique_ratio, most_frequent_ratio,
                variance_threshold, unique_ratio_threshold, frequency_ratio_threshold,
            )

            results.append(
                FeatureVarianceResult(
                    feature=col,
                    variance=round(variance, 8),
                    std=round(std, 8),
                    coefficient_of_variation=round(cv, 6),
                    unique_ratio=round(unique_ratio, 6),
                    most_frequent_ratio=round(most_frequent_ratio, 6),
                    is_near_zero_variance=is_nzv,
                    reason=reason,
                )
            )

        results.sort(key=lambda r: r.variance)
        flagged = [r.feature for r in results if r.is_near_zero_variance]

        return {
            "features": [
                {
                    "feature": r.feature,
                    "variance": r.variance,
                    "std": r.std,
                    "coefficient_of_variation": r.coefficient_of_variation,
                    "unique_ratio": r.unique_ratio,
                    "most_frequent_ratio": r.most_frequent_ratio,
                    "is_near_zero_variance": r.is_near_zero_variance,
                    "reason": r.reason,
                }
                for r in results
            ],
            "flagged_features": flagged,
            "n_flagged": len(flagged),
            "n_total_features": len(results),
            "thresholds": {
                "variance": variance_threshold,
                "unique_ratio": unique_ratio_threshold,
                "frequency_ratio": frequency_ratio_threshold,
            },
            "recommendation": (
                f"Consider dropping {flagged} before model training."
                if flagged else "No near-zero variance features detected."
            ),
        }
