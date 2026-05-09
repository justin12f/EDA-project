"""Univariate feature selection scoring: Chi2, ANOVA F, and Mutual Information."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    chi2,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.preprocessing import MinMaxScaler


@dataclass(frozen=True)
class FeatureScore:
    """Immutable score record for a single feature."""

    rank: int
    feature: str
    score: float
    p_value: float | None
    normalized_score: float
    method: str


class BaseFeatureScoringMethod(ABC):
    """Abstract base for all univariate feature scoring strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return method name."""

    @abstractmethod
    def score(
        self,
        x: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
    ) -> list[FeatureScore]:
        """Compute scores for all features.

        Args:
            x: Feature matrix (n_samples × n_features).
            y: Target array (n_samples,).
            feature_names: Column names aligned with x columns.

        Returns:
            List of FeatureScore objects (unranked, unsorted).
        """


class Chi2Scorer(BaseFeatureScoringMethod):
    """Chi-square test score for categorical targets and non-negative features.

    Measures statistical dependence between each feature and the target.
    Requires non-negative features — apply MinMaxScaler before scoring.
    Only valid for classification tasks.
    """

    @property
    def name(self) -> str:
        return "chi2"

    def score(
        self,
        x: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
    ) -> list[FeatureScore]:
        if np.any(x < 0):
            x = MinMaxScaler().fit_transform(x)

        scores, p_values = chi2(x, y)
        return self._build_scores(scores, p_values, feature_names)

    def _build_scores(
        self,
        scores: np.ndarray,
        p_values: np.ndarray,
        feature_names: list[str],
    ) -> list[FeatureScore]:
        max_score = float(scores.max()) if scores.max() > 0 else 1.0
        return [
            FeatureScore(
                rank=0,
                feature=name,
                score=round(float(s), 6),
                p_value=round(float(p), 8),
                normalized_score=round(float(s) / max_score, 6),
                method=self.name,
            )
            for name, s, p in zip(feature_names, scores, p_values)
        ]


class ANOVAFScorer(BaseFeatureScoringMethod):
    """ANOVA F-score between each feature and a categorical/continuous target.

    Selects `f_classif` for classification and `f_regression` for regression
    based on the target_type parameter passed at construction time.
    """

    def __init__(self, target_type: str = "classification") -> None:
        self._target_type = target_type

    @property
    def name(self) -> str:
        return f"anova_f_{self._target_type}"

    def score(
        self,
        x: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
    ) -> list[FeatureScore]:
        fn = f_classif if self._target_type == "classification" else f_regression
        scores, p_values = fn(x, y)
        scores = np.nan_to_num(scores, nan=0.0)
        p_values = np.nan_to_num(p_values, nan=1.0)

        max_score = float(scores.max()) if scores.max() > 0 else 1.0
        return [
            FeatureScore(
                rank=0,
                feature=name,
                score=round(float(s), 6),
                p_value=round(float(p), 8),
                normalized_score=round(float(s) / max_score, 6),
                method=self.name,
            )
            for name, s, p in zip(feature_names, scores, p_values)
        ]


class MIScorer(BaseFeatureScoringMethod):
    """Mutual Information scorer for classification or regression targets.

    Non-parametric — no distribution assumption. Handles non-linear
    relationships that ANOVA F misses. Computationally heavier.
    """

    def __init__(
        self, target_type: str = "classification", random_seed: int = 42
    ) -> None:
        self._target_type = target_type
        self._random_seed = random_seed

    @property
    def name(self) -> str:
        return f"mutual_information_{self._target_type}"

    def score(
        self,
        x: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
    ) -> list[FeatureScore]:
        fn = (
            mutual_info_classif
            if self._target_type == "classification"
            else mutual_info_regression
        )
        scores = fn(x, y, random_state=self._random_seed)

        max_score = float(scores.max()) if scores.max() > 0 else 1.0
        return [
            FeatureScore(
                rank=0,
                feature=name,
                score=round(float(s), 6),
                p_value=None,
                normalized_score=round(float(s) / max_score, 6),
                method=self.name,
            )
            for name, s in zip(feature_names, scores)
        ]


class ScoreRanker:
    """Ranks and optionally filters a list of FeatureScore objects."""

    def rank(
        self,
        scores: list[FeatureScore],
        top_n: int | None,
    ) -> list[FeatureScore]:
        """Sort by score descending and assign ranks.

        Args:
            scores: Unranked FeatureScore list.
            top_n: Retain only top N. None = all.

        Returns:
            Ranked and optionally filtered list.
        """
        sorted_scores = sorted(scores, key=lambda s: s.score, reverse=True)
        ranked = [
            FeatureScore(
                rank=i + 1,
                feature=s.feature,
                score=s.score,
                p_value=s.p_value,
                normalized_score=s.normalized_score,
                method=s.method,
            )
            for i, s in enumerate(sorted_scores)
        ]
        return ranked[:top_n] if top_n is not None else ranked


class FeatureSelectionCalculator:
    """Univariate feature scoring with chi2, ANOVA F, and mutual information.

    Workflow:
        calculator = FeatureSelectionCalculator()
        result = calculator.calculate(
            data_frame=df,
            target_column="churn",
            feature_columns=["age", "income"],  # optional
            methods=["chi2", "anova_f", "mutual_information"],
            target_type="classification",        # "classification" | "regression"
            top_n=10,
            random_seed=42,
        )
    """

    _MINIMUM_SAMPLES: int = 10

    def __init__(self) -> None:
        self._ranker = ScoreRanker()

    def calculate(
        self,
        data_frame: pd.DataFrame,
        target_column: str,
        feature_columns: list[str] | None = None,
        methods: list[str] | None = None,
        target_type: str = "classification",
        top_n: int | None = None,
        random_seed: int = 42,
    ) -> dict:
        """Run all requested scoring methods.

        Args:
            data_frame: Source DataFrame.
            target_column: Target variable column name.
            feature_columns: Subset of feature columns. Defaults to all numeric.
            methods: Scoring methods to run. Defaults to all three.
            target_type: 'classification' or 'regression'.
            top_n: Retain only top N features per method.
            random_seed: Seed for MI scorer reproducibility.

        Returns:
            Dict with per-method ranked scores and consensus top features.

        Raises:
            KeyError: If columns are not found.
            ValueError: If target_type is invalid or data is insufficient.
        """
        _VALID_METHODS: frozenset[str] = frozenset({"chi2", "anova_f", "mutual_information"})
        _VALID_TARGET_TYPES: frozenset[str] = frozenset({"classification", "regression"})

        if target_column not in data_frame.columns:
            raise KeyError(f"Target column '{target_column}' not found.")
        if target_type not in _VALID_TARGET_TYPES:
            raise ValueError(
                f"target_type must be one of {_VALID_TARGET_TYPES}. Got '{target_type}'."
            )

        active_methods = methods if methods is not None else list(_VALID_METHODS)
        invalid_methods = [m for m in active_methods if m not in _VALID_METHODS]
        if invalid_methods:
            raise KeyError(
                f"Unknown method(s): {invalid_methods}. Available: {_VALID_METHODS}"
            )

        numeric_cols = data_frame.select_dtypes(include=[np.number]).columns.tolist()
        if target_column in numeric_cols:
            numeric_cols.remove(target_column)

        features_to_use = feature_columns if feature_columns is not None else numeric_cols
        missing = [c for c in features_to_use if c not in data_frame.columns]
        if missing:
            raise KeyError(f"Feature columns not found: {missing}")

        all_cols = features_to_use + [target_column]
        clean = data_frame[all_cols].dropna()

        if len(clean) < self._MINIMUM_SAMPLES:
            raise ValueError(
                f"At least {self._MINIMUM_SAMPLES} non-null observations required. "
                f"Got {len(clean)}."
            )

        x = clean[features_to_use].to_numpy(dtype=float)
        y = clean[target_column].to_numpy()
        feature_names = features_to_use

        scorer_map: dict[str, BaseFeatureScoringMethod] = {
            "chi2": Chi2Scorer(),
            "anova_f": ANOVAFScorer(target_type),
            "mutual_information": MIScorer(target_type, random_seed),
        }

        results_per_method: dict[str, list[dict]] = {}
        top_features_per_method: dict[str, list[str]] = {}

        for method_key in active_methods:
            scorer = scorer_map[method_key]
            raw_scores = scorer.score(x, y, feature_names)
            ranked = self._ranker.rank(raw_scores, top_n)

            results_per_method[method_key] = [
                {
                    "rank": s.rank,
                    "feature": s.feature,
                    "score": s.score,
                    "p_value": s.p_value,
                    "normalized_score": s.normalized_score,
                }
                for s in ranked
            ]
            top_features_per_method[method_key] = [s.feature for s in ranked]

        consensus = self._build_consensus(top_features_per_method, top_n)

        return {
            "scores": results_per_method,
            "top_features_per_method": top_features_per_method,
            "consensus_top_features": consensus,
            "target_column": target_column,
            "target_type": target_type,
            "methods_used": active_methods,
            "n_features_evaluated": len(feature_names),
            "n_observations": len(clean),
            "top_n": top_n,
        }

    def _build_consensus(
        self,
        top_features_per_method: dict[str, list[str]],
        top_n: int | None,
    ) -> list[str]:
        """Build consensus feature list ranked by frequency of appearance.

        Args:
            top_features_per_method: Per-method top feature lists.
            top_n: Max consensus features to return.

        Returns:
            Features ranked by how many methods selected them.
        """
        vote_counts: dict[str, int] = {}
        for features in top_features_per_method.values():
            for feature in features:
                vote_counts[feature] = vote_counts.get(feature, 0) + 1

        ranked = sorted(vote_counts.items(), key=lambda x: x[1], reverse=True)
        consensus = [f for f, _ in ranked]
        return consensus[:top_n] if top_n is not None else consensus
