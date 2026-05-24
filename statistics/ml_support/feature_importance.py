"""Tree-based feature importance extraction and permutation importance."""

# #[AI_CONTEXT_START]
# - CONFIGURACIÓN DE FACTORY: Registrar este calculator en `MlSupportStatisticsFactory` (backends pandas | polars | spark) y exponerlo mediante `StatisticsInyeccionDependency`, inyectada por la Factory Maestra de Agentes junto a las demás fábricas de dominio.
# - ABSTRACCIÓN DEL DATO: Mutar constructores y `analyze`/`compute` para recibir el contenedor abstracto del backend (`pd.DataFrame`, `pl.DataFrame`/`pl.LazyFrame`, `pyspark.sql.DataFrame`) inyectado por la factory; eliminar `np.ndarray`/`pd.Series` sueltos en firmas públicas.
# - REFACTOR NATIVO: Resolver métricas con expresiones 100 % nativas del backend activo (Polars: `.select`/`.group_by`/`.agg` sin `.collect()` salvo materialización acordada; PySpark: `pyspark.sql.functions` y ventanas distribuidas; Pandas: operaciones vectorizadas). No convertir a NumPy/Pandas desde backends no-pandas.
# #[AI_CONTEXT_END]
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance

@dataclass(frozen=True)
class ImportanceScore:
    """Immutable importance record for a single feature."""

    rank: int
    feature: str
    importance: float
    std: float | None
    normalized_importance: float
    method: str

class BaseImportanceExtractor(ABC):
    """Abstract base for all feature importance extraction strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return extractor name."""

    @abstractmethod
    def extract(
        self,
        x: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
        random_seed: int,
        **kwargs,
    ) -> list[ImportanceScore]:
        """Extract feature importances.

        Args:
            x: Feature matrix.
            y: Target array.
            feature_names: Column names.
            random_seed: Random seed for reproducibility.
            **kwargs: Extractor-specific hyperparameters.

        Returns:
            List of ImportanceScore objects (unranked).
        """

class GiniImportanceExtractor(BaseImportanceExtractor):
    """Mean Decrease in Impurity (MDI / Gini importance) from a Random Forest.

    Fast to compute (no additional fitting needed — derived from tree
    structure). Can overestimate importance of high-cardinality features.
    """

    @property
    def name(self) -> str:
        return "gini_importance"

    def extract(
        self,
        x: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
        random_seed: int,
        **kwargs,
    ) -> list[ImportanceScore]:
        target_type = kwargs.get("target_type", "classification")
        n_estimators = kwargs.get("n_estimators", 100)

        model_class = (
            RandomForestClassifier
            if target_type == "classification"
            else RandomForestRegressor
        )
        model = model_class(
            n_estimators=n_estimators,
            random_state=random_seed,
            n_jobs=-1,
        )
        model.fit(x, y)

        importances = model.feature_importances_
        stds = np.std(
            [tree.feature_importances_ for tree in model.estimators_], axis=0
        )
        total = float(importances.sum()) or 1.0

        return [
            ImportanceScore(
                rank=0,
                feature=name,
                importance=round(float(imp), 6),
                std=round(float(std), 6),
                normalized_importance=round(float(imp) / total, 6),
                method=self.name,
            )
            for name, imp, std in zip(feature_names, importances, stds)
        ]

class PermutationImportanceExtractor(BaseImportanceExtractor):
    """Permutation feature importance (MDA — Mean Decrease in Accuracy).

    Measures how much the model score drops when a feature's values are
    randomly shuffled. More reliable than Gini for high-cardinality features
    and better reflects actual predictive contribution. Slower to compute.
    """

    @property
    def name(self) -> str:
        return "permutation_importance"

    def extract(
        self,
        x: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
        random_seed: int,
        **kwargs,
    ) -> list[ImportanceScore]:
        target_type = kwargs.get("target_type", "classification")
        n_estimators = kwargs.get("n_estimators", 100)
        n_repeats = kwargs.get("n_repeats", 10)

        model_class = (
            RandomForestClassifier
            if target_type == "classification"
            else RandomForestRegressor
        )
        model = model_class(
            n_estimators=n_estimators,
            random_state=random_seed,
            n_jobs=-1,
        )
        model.fit(x, y)

        perm_result = permutation_importance(
            model, x, y,
            n_repeats=n_repeats,
            random_state=random_seed,
            n_jobs=-1,
        )

        importances = perm_result.importances_mean
        stds = perm_result.importances_std
        total = float(np.abs(importances).sum()) or 1.0

        return [
            ImportanceScore(
                rank=0,
                feature=name,
                importance=round(float(imp), 6),
                std=round(float(std), 6),
                normalized_importance=round(abs(float(imp)) / total, 6),
                method=self.name,
            )
            for name, imp, std in zip(feature_names, importances, stds)
        ]

_EXTRACTOR_REGISTRY: dict[str, BaseImportanceExtractor] = {
    "gini": GiniImportanceExtractor(),
    "permutation": PermutationImportanceExtractor(),
}

class ImportanceRanker:
    """Sorts and ranks ImportanceScore objects descending by |importance|."""

    def rank(
        self, scores: list[ImportanceScore], top_n: int | None
    ) -> list[ImportanceScore]:
        sorted_scores = sorted(scores, key=lambda s: abs(s.importance), reverse=True)
        ranked = [
            ImportanceScore(
                rank=i + 1,
                feature=s.feature,
                importance=s.importance,
                std=s.std,
                normalized_importance=s.normalized_importance,
                method=s.method,
            )
            for i, s in enumerate(sorted_scores)
        ]
        return ranked[:top_n] if top_n is not None else ranked

class FeatureImportanceCalculator:
    """Tree-based feature importance with Gini and/or permutation methods.

    Workflow:
        calculator = FeatureImportanceCalculator()
        result = calculator.calculate(
            data_frame=df,
            target_column="churn",
            feature_columns=["age", "income"],   # optional
            methods=["gini", "permutation"],      # optional
            target_type="classification",
            n_estimators=100,
            n_repeats=10,                         # permutation only
            top_n=15,
            random_seed=42,
        )
    """

    _MINIMUM_SAMPLES: int = 20

    def __init__(self) -> None:
        self._ranker = ImportanceRanker()

    def calculate(
        self,
        data_frame: pd.DataFrame,
        target_column: str,
        feature_columns: list[str] | None = None,
        methods: list[str] | None = None,
        target_type: str = "classification",
        n_estimators: int = 100,
        n_repeats: int = 10,
        top_n: int | None = None,
        random_seed: int = 42,
    ) -> dict:
        """Compute feature importances from tree models.

        Args:
            data_frame: Source DataFrame.
            target_column: Target variable column name.
            feature_columns: Feature columns subset. Defaults to all numeric.
            methods: Importance methods. Defaults to ['gini', 'permutation'].
            target_type: 'classification' or 'regression'.
            n_estimators: Number of trees in the Random Forest.
            n_repeats: Permutation repetitions (permutation method only).
            top_n: Return only top N features per method.
            random_seed: Seed for reproducibility.

        Returns:
            Dict with per-method importances and consensus ranking.

        Raises:
            KeyError: If columns or methods are not found.
            ValueError: If data is insufficient.
        """
        _VALID_TARGET_TYPES: frozenset[str] = frozenset({"classification", "regression"})

        if target_column not in data_frame.columns:
            raise KeyError(f"Target column '{target_column}' not found.")
        if target_type not in _VALID_TARGET_TYPES:
            raise ValueError(
                f"target_type must be one of {_VALID_TARGET_TYPES}. Got '{target_type}'."
            )

        active_methods = methods if methods is not None else list(_EXTRACTOR_REGISTRY.keys())
        invalid = [m for m in active_methods if m not in _EXTRACTOR_REGISTRY]
        if invalid:
            raise KeyError(
                f"Unknown method(s): {invalid}. "
                f"Available: {list(_EXTRACTOR_REGISTRY.keys())}"
            )

        numeric_cols = [
            c for c in data_frame.select_dtypes(include=[np.number]).columns
            if c != target_column
        ]
        features_to_use = feature_columns if feature_columns is not None else numeric_cols
        missing = [c for c in features_to_use if c not in data_frame.columns]
        if missing:
            raise KeyError(f"Feature columns not found: {missing}")

        clean = data_frame[features_to_use + [target_column]].dropna()

        if len(clean) < self._MINIMUM_SAMPLES:
            raise ValueError(
                f"At least {self._MINIMUM_SAMPLES} observations required. "
                f"Got {len(clean)}."
            )

        x = clean[features_to_use].to_numpy(dtype=float)
        y = clean[target_column].to_numpy()

        results_per_method: dict[str, list[dict]] = {}
        extractor_kwargs = {
            "target_type": target_type,
            "n_estimators": n_estimators,
            "n_repeats": n_repeats,
        }

        for method_key in active_methods:
            extractor = _EXTRACTOR_REGISTRY[method_key]
            raw = extractor.extract(x, y, features_to_use, random_seed, **extractor_kwargs)
            ranked = self._ranker.rank(raw, top_n)
            results_per_method[method_key] = [
                {
                    "rank": s.rank,
                    "feature": s.feature,
                    "importance": s.importance,
                    "std": s.std,
                    "normalized_importance": s.normalized_importance,
                }
                for s in ranked
            ]

        return {
            "importances": results_per_method,
            "target_column": target_column,
            "target_type": target_type,
            "methods_used": active_methods,
            "n_estimators": n_estimators,
            "n_features_evaluated": len(features_to_use),
            "n_observations": len(clean),
        }
