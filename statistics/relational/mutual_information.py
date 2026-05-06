"""Mutual information feature relevance scoring."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    mutual_info_classif,
    mutual_info_regression,
)


@dataclass(frozen=True)
class MIScore:
    """Immutable mutual information score for a single feature."""

    feature: str
    mi_score: float
    normalized_score: float
    rank: int


class TargetTypeDetector:
    """Determines whether a target variable is continuous or categorical.

    Uses cardinality heuristic: if unique values <= threshold, treat as
    categorical (classification task). Otherwise treat as continuous.
    """

    _CATEGORICAL_CARDINALITY_THRESHOLD: int = 20

    def detect(self, target: pd.Series) -> str:
        """Detect whether the target is 'continuous' or 'categorical'.

        Args:
            target: Target Series.

        Returns:
            'categorical' or 'continuous'.
        """
        if not pd.api.types.is_numeric_dtype(target):
            return "categorical"

        n_unique = target.nunique()
        return (
            "categorical"
            if n_unique <= self._CATEGORICAL_CARDINALITY_THRESHOLD
            else "continuous"
        )


class MIScoreNormalizer:
    """Normalizes MI scores to [0, 1] by dividing by the maximum score.

    A score of 1.0 means that feature carries the most information about
    the target among all features evaluated.
    """

    def normalize(self, scores: np.ndarray) -> np.ndarray:
        """Normalize MI scores to [0, 1].

        Args:
            scores: Raw MI score array.

        Returns:
            Normalized score array.
        """
        max_score = float(scores.max())
        if max_score == 0.0:
            return np.zeros_like(scores)
        return scores / max_score


class MutualInformationCalculator:
    """Scores features by their mutual information with a target variable.

    Automatically selects classifier MI (discrete target) or regression MI
    (continuous target) based on target type detection.

    Workflow:
        calculator = MutualInformationCalculator()
        result = calculator.calculate(
            features=df[["age", "income", "category"]],
            target=df["churn"],
            target_type="auto",   # "auto" | "categorical" | "continuous"
            top_n=10,             # optional
            random_seed=42,       # optional
        )
    """

    _MINIMUM_SAMPLES: int = 10

    def __init__(self) -> None:
        self._target_detector = TargetTypeDetector()
        self._normalizer = MIScoreNormalizer()

    def calculate(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        target_type: str = "auto",
        top_n: int | None = None,
        random_seed: int = 42,
    ) -> dict:
        """Compute mutual information scores for all features.

        Args:
            features: DataFrame of predictor features (numeric and/or categorical).
            target: Target Series aligned with features.
            target_type: 'auto', 'categorical', or 'continuous'.
            top_n: Return only top N features by MI score.
            random_seed: Seed for reproducibility (sklearn MI uses random forests).

        Returns:
            Dict with ranked feature MI scores and target type detected.

        Raises:
            ValueError: If target_type is invalid or sample size is too small.
        """
        _VALID_TARGET_TYPES: frozenset[str] = frozenset(
            {"auto", "categorical", "continuous"}
        )

        if target_type not in _VALID_TARGET_TYPES:
            raise ValueError(
                f"target_type must be one of {_VALID_TARGET_TYPES}. "
                f"Got '{target_type}'."
            )

        aligned = features.join(target.rename("__target__")).dropna()

        if len(aligned) < self._MINIMUM_SAMPLES:
            raise ValueError(
                f"At least {self._MINIMUM_SAMPLES} non-null observations required. "
                f"Got {len(aligned)}."
            )

        x = aligned.drop(columns="__target__")
        y = aligned["__target__"]

        # One-hot encode any object/categorical columns for sklearn compatibility
        x_encoded = pd.get_dummies(x, drop_first=False)

        detected_type = (
            self._target_detector.detect(y)
            if target_type == "auto"
            else target_type
        )

        if detected_type == "categorical":
            raw_scores = mutual_info_classif(
                x_encoded.to_numpy(),
                y.to_numpy(),
                random_state=random_seed,
            )
        else:
            raw_scores = mutual_info_regression(
                x_encoded.to_numpy(),
                y.to_numpy(),
                random_state=random_seed,
            )

        normalized = self._normalizer.normalize(raw_scores)
        feature_names = x_encoded.columns.tolist()

        scored: list[MIScore] = [
            MIScore(
                feature=name,
                mi_score=float(raw_scores[i]),
                normalized_score=float(normalized[i]),
                rank=0,
            )
            for i, name in enumerate(feature_names)
        ]

        scored.sort(key=lambda s: s.mi_score, reverse=True)
        scored = [
            MIScore(
                feature=s.feature,
                mi_score=s.mi_score,
                normalized_score=s.normalized_score,
                rank=rank + 1,
            )
            for rank, s in enumerate(scored)
        ]

        if top_n is not None:
            scored = scored[:top_n]

        return {
            "scores": [
                {
                    "rank": s.rank,
                    "feature": s.feature,
                    "mi_score": round(s.mi_score, 6),
                    "normalized_score": round(s.normalized_score, 6),
                }
                for s in scored
            ],
            "target_type_detected": detected_type,
            "n_features_evaluated": len(feature_names),
            "n_observations": len(aligned),
            "top_n": top_n,
        }
