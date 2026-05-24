"""Population split analysis: statistical comparison between two groups."""

# #[AI_CONTEXT_START]
# - CONFIGURACIÓN DE FACTORY: Registrar este calculator en `SegmentationStatisticsFactory` (backends pandas | polars | spark) y exponerlo mediante `StatisticsInyeccionDependency`, inyectada por la Factory Maestra de Agentes junto a las demás fábricas de dominio.
# - ABSTRACCIÓN DEL DATO: Mutar constructores y `analyze`/`compute` para recibir el contenedor abstracto del backend (`pd.DataFrame`, `pl.DataFrame`/`pl.LazyFrame`, `pyspark.sql.DataFrame`) inyectado por la factory; eliminar `np.ndarray`/`pd.Series` sueltos en firmas públicas.
# - REFACTOR NATIVO: Resolver métricas con expresiones 100 % nativas del backend activo (Polars: `.select`/`.group_by`/`.agg` sin `.collect()` salvo materialización acordada; PySpark: `pyspark.sql.functions` y ventanas distribuidas; Pandas: operaciones vectorizadas). No convertir a NumPy/Pandas desde backends no-pandas.
# #[AI_CONTEXT_END]
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

@dataclass(frozen=True)
class FeatureSplitResult:
    """Immutable comparison result for a single feature across two groups."""

    feature: str
    group_a_mean: float
    group_b_mean: float
    mean_difference: float
    mean_difference_pct: float
    t_statistic: float
    p_value: float
    is_significant: bool
    effect_size_cohens_d: float
    effect_magnitude: str

class WelchTTestComparator:
    """Compares two groups on a single numeric feature via Welch's T-test."""

    def compare(
        self,
        group_a: np.ndarray,
        group_b: np.ndarray,
        significance_level: float,
    ) -> tuple[float, float, bool]:
        """Run Welch T-test between two groups."""
        statistic, p_value = stats.ttest_ind(
            group_a, group_b, equal_var=False, alternative="two-sided"
        )
        return float(statistic), float(p_value), float(p_value) < significance_level

class CohensDComputer:
    """Computes Cohen's d effect size between two numeric arrays."""

    def compute(self, group_a: np.ndarray, group_b: np.ndarray) -> float:
        """Compute pooled-std Cohen's d."""
        pooled_std = float(np.sqrt(
            (group_a.var(ddof=1) * (len(group_a) - 1) +
             group_b.var(ddof=1) * (len(group_b) - 1)) /
            (len(group_a) + len(group_b) - 2)
        ))
        if pooled_std == 0.0:
            return 0.0
        return float((group_a.mean() - group_b.mean()) / pooled_std)

class EffectMagnitudeClassifier:
    """Classifies Cohen's d into magnitude labels (Cohen 1988)."""

    _THRESHOLDS: list[tuple[float, str]] = [
        (0.8, "large"),
        (0.5, "medium"),
        (0.2, "small"),
        (0.0, "negligible"),
    ]

    def classify(self, cohens_d: float) -> str:
        """Classify absolute Cohen's d."""
        abs_d = abs(cohens_d)
        for threshold, label in self._THRESHOLDS:
            if abs_d >= threshold:
                return label
        return "negligible"

class CategoricalDistributionComparator:
    """Compares categorical feature distributions between two groups via Chi-square."""

    def compare(
        self,
        series_a: pd.Series,
        series_b: pd.Series,
        significance_level: float,
    ) -> dict:
        """Compare categorical distributions."""
        all_categories = set(series_a.unique()) | set(series_b.unique())
        n_a, n_b = len(series_a), len(series_b)
        counts_a = series_a.value_counts()
        counts_b = series_b.value_counts()
        observed = np.array([
            [counts_a.get(cat, 0), counts_b.get(cat, 0)]
            for cat in all_categories
        ])
        if observed.shape[0] < 2:
            return {
                "chi2_statistic": None,
                "p_value": None,
                "is_significant": False,
                "note": "Insufficient categories for chi-square test.",
            }
        chi2, p_value, _, _ = stats.chi2_contingency(observed)
        return {
            "chi2_statistic": round(float(chi2), 4),
            "p_value": round(float(p_value), 6),
            "is_significant": float(p_value) < significance_level,
            "proportions_a": {
                str(cat): round(counts_a.get(cat, 0) / n_a, 4)
                for cat in all_categories
            },
            "proportions_b": {
                str(cat): round(counts_b.get(cat, 0) / n_b, 4)
                for cat in all_categories
            },
        }

class PopulationSplitsCalculator:
    """Statistical comparison of all features between two population groups."""

    _MINIMUM_PER_GROUP: int = 5

    def __init__(self) -> None:
        self._t_test = WelchTTestComparator()
        self._cohens_d = CohensDComputer()
        self._effect_classifier = EffectMagnitudeClassifier()
        self._categorical_comparator = CategoricalDistributionComparator()

    def calculate(
        self,
        data_frame: pd.DataFrame,
        split_column: str,
        group_a_value,
        group_b_value,
        feature_columns: list[str] | None = None,
        significance_level: float = 0.05,
    ) -> dict:
        """Compare all features between two population groups."""
        if split_column not in data_frame.columns:
            raise KeyError(f"split_column '{split_column}' not found.")

        mask_a = data_frame[split_column] == group_a_value
        mask_b = data_frame[split_column] == group_b_value

        if mask_a.sum() < self._MINIMUM_PER_GROUP:
            raise ValueError(
                f"Group A (value={group_a_value}) has fewer than "
                f"{self._MINIMUM_PER_GROUP} members. Got {mask_a.sum()}."
            )
        if mask_b.sum() < self._MINIMUM_PER_GROUP:
            raise ValueError(
                f"Group B (value={group_b_value}) has fewer than "
                f"{self._MINIMUM_PER_GROUP} members. Got {mask_b.sum()}."
            )

        all_features = [c for c in data_frame.columns if c != split_column]
        features = feature_columns if feature_columns is not None else all_features
        missing = [c for c in features if c not in data_frame.columns]
        if missing:
            raise KeyError(f"Feature columns not found: {missing}")

        group_a_df = data_frame[mask_a]
        group_b_df = data_frame[mask_b]
        numeric_results: list[FeatureSplitResult] = []
        categorical_results: dict[str, dict] = {}

        for feature in features:
            if pd.api.types.is_numeric_dtype(data_frame[feature]):
                a_vals = group_a_df[feature].dropna().to_numpy(dtype=float)
                b_vals = group_b_df[feature].dropna().to_numpy(dtype=float)
                if len(a_vals) < 2 or len(b_vals) < 2:
                    continue
                t_stat, p_val, is_sig = self._t_test.compare(
                    a_vals, b_vals, significance_level
                )
                d = self._cohens_d.compute(a_vals, b_vals)
                magnitude = self._effect_classifier.classify(d)
                mean_a = float(a_vals.mean())
                mean_b = float(b_vals.mean())
                diff = mean_a - mean_b
                diff_pct = diff / abs(mean_b) * 100 if mean_b != 0 else float("inf")
                numeric_results.append(
                    FeatureSplitResult(
                        feature=feature,
                        group_a_mean=round(mean_a, 4),
                        group_b_mean=round(mean_b, 4),
                        mean_difference=round(diff, 4),
                        mean_difference_pct=round(diff_pct, 2),
                        t_statistic=round(t_stat, 4),
                        p_value=round(p_val, 6),
                        is_significant=is_sig,
                        effect_size_cohens_d=round(d, 4),
                        effect_magnitude=magnitude,
                    )
                )
            elif pd.api.types.is_object_dtype(data_frame[feature]) or \
                    pd.api.types.is_categorical_dtype(data_frame[feature]):
                categorical_results[feature] = self._categorical_comparator.compare(
                    group_a_df[feature].dropna(),
                    group_b_df[feature].dropna(),
                    significance_level,
                )

        numeric_results.sort(key=lambda r: r.p_value)
        significant = [r.feature for r in numeric_results if r.is_significant]

        return {
            "numeric_comparisons": [
                {
                    "feature": r.feature,
                    "group_a_mean": r.group_a_mean,
                    "group_b_mean": r.group_b_mean,
                    "mean_difference": r.mean_difference,
                    "mean_difference_pct": r.mean_difference_pct,
                    "t_statistic": r.t_statistic,
                    "p_value": r.p_value,
                    "is_significant": r.is_significant,
                    "effect_size_cohens_d": r.effect_size_cohens_d,
                    "effect_magnitude": r.effect_magnitude,
                }
                for r in numeric_results
            ],
            "categorical_comparisons": categorical_results,
            "significant_features": significant,
            "n_significant": len(significant),
            "group_a": {"value": group_a_value, "n": int(mask_a.sum())},
            "group_b": {"value": group_b_value, "n": int(mask_b.sum())},
            "significance_level": significance_level,
        }
