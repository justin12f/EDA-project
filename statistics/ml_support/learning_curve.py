"""Learning curve analysis for bias-variance diagnosis."""

# #[AI_CONTEXT_START]
# - CONFIGURACIÓN DE FACTORY: Registrar este calculator en `MlSupportStatisticsFactory` (backends pandas | polars | spark) y exponerlo mediante `StatisticsInyeccionDependency`, inyectada por la Factory Maestra de Agentes junto a las demás fábricas de dominio.
# - ABSTRACCIÓN DEL DATO: Mutar constructores y `analyze`/`compute` para recibir el contenedor abstracto del backend (`pd.DataFrame`, `pl.DataFrame`/`pl.LazyFrame`, `pyspark.sql.DataFrame`) inyectado por la factory; eliminar `np.ndarray`/`pd.Series` sueltos en firmas públicas.
# - REFACTOR NATIVO: Resolver métricas con expresiones 100 % nativas del backend activo (Polars: `.select`/`.group_by`/`.agg` sin `.collect()` salvo materialización acordada; PySpark: `pyspark.sql.functions` y ventanas distribuidas; Pandas: operaciones vectorizadas). No convertir a NumPy/Pandas desde backends no-pandas.
# #[AI_CONTEXT_END]
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score

@dataclass(frozen=True)
class LearningCurvePoint:
    """Immutable result for a single training size checkpoint."""

    train_size: int
    train_score_mean: float
    train_score_std: float
    val_score_mean: float
    val_score_std: float
    bias_variance_state: str

class BiasVarianceDiagnostic:
    """Classifies model state based on training vs validation score gap.

    States:
        - 'high_bias':            Both scores are low — underfitting.
        - 'high_variance':        Large gap between train and val — overfitting.
        - 'good_fit':             Small gap, both scores acceptable.
        - 'insufficient_data':    Validation score is still improving steeply.
    """

    def __init__(
        self,
        gap_threshold: float,
        min_acceptable_score: float,
    ) -> None:
        self._gap_threshold = gap_threshold
        self._min_score = min_acceptable_score

    def classify(
        self,
        train_score: float,
        val_score: float,
    ) -> str:
        """Classify bias-variance state from train and val scores.

        Args:
            train_score: Mean training score.
            val_score: Mean validation score.

        Returns:
            Bias-variance state label.
        """
        gap = train_score - val_score

        if val_score < self._min_score and train_score < self._min_score:
            return "high_bias"
        if gap > self._gap_threshold:
            return "high_variance"
        if val_score < self._min_score:
            return "insufficient_data"
        return "good_fit"

class TrainingSizeGenerator:
    """Generates logarithmically spaced training size checkpoints."""

    def generate(
        self,
        n_total: int,
        n_checkpoints: int,
        min_train_size: int,
    ) -> list[int]:
        """Generate training size checkpoints.

        Args:
            n_total: Total number of training observations.
            n_checkpoints: Number of points on the curve.
            min_train_size: Minimum training size to include.

        Returns:
            Sorted list of integer training sizes.
        """
        raw = np.logspace(
            np.log10(min_train_size),
            np.log10(n_total),
            num=n_checkpoints,
        )
        sizes = sorted(set(int(np.round(s)) for s in raw))
        return [s for s in sizes if min_train_size <= s <= n_total]

class LearningCurveCalculator:
    """Generates learning curves to diagnose bias vs variance.

    Workflow:
        from sklearn.linear_model import LogisticRegression

        calculator = LearningCurveCalculator()
        result = calculator.calculate(
            data_frame=df,
            target_column="churn",
            feature_columns=["age", "income"],  # optional
            estimator=LogisticRegression(),
            n_checkpoints=10,
            cv=5,
            scoring="accuracy",
            gap_threshold=0.1,
            min_acceptable_score=0.6,
            random_seed=42,
        )
    """

    _MINIMUM_SAMPLES: int = 20
    _DEFAULT_MIN_TRAIN_SIZE: int = 10

    def __init__(self) -> None:
        self._size_generator = TrainingSizeGenerator()

    def calculate(
        self,
        data_frame: pd.DataFrame,
        target_column: str,
        estimator: BaseEstimator,
        feature_columns: list[str] | None = None,
        n_checkpoints: int = 10,
        cv: int = 5,
        scoring: str = "accuracy",
        gap_threshold: float = 0.1,
        min_acceptable_score: float = 0.6,
        random_seed: int = 42,
    ) -> dict:
        """Compute learning curve across training size checkpoints.

        Args:
            data_frame: Source DataFrame.
            target_column: Target variable column name.
            estimator: Sklearn-compatible estimator (unfitted).
            feature_columns: Feature subset. Defaults to all numeric.
            n_checkpoints: Number of training size points to evaluate.
            cv: Number of cross-validation folds.
            scoring: Sklearn scoring metric string.
            gap_threshold: Train-val gap above which 'high_variance' is flagged.
            min_acceptable_score: Minimum val score for 'good_fit'.
            random_seed: Seed for reproducibility (shuffling).

        Returns:
            Dict with learning curve points, final diagnosis, and recommendation.

        Raises:
            KeyError: If columns are not found.
            ValueError: If data is insufficient or parameters are invalid.
        """
        if target_column not in data_frame.columns:
            raise KeyError(f"Target column '{target_column}' not found.")
        if cv < 2:
            raise ValueError(f"cv must be >= 2. Got {cv}.")
        if n_checkpoints < 2:
            raise ValueError(f"n_checkpoints must be >= 2. Got {n_checkpoints}.")

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

        rng = np.random.default_rng(random_seed)
        shuffled_idx = rng.permutation(len(clean))
        x_all = clean[features_to_use].to_numpy(dtype=float)[shuffled_idx]
        y_all = clean[target_column].to_numpy()[shuffled_idx]

        n_total = len(clean)
        min_train = max(self._DEFAULT_MIN_TRAIN_SIZE, cv)
        train_sizes = self._size_generator.generate(n_total, n_checkpoints, min_train)

        diagnostic = BiasVarianceDiagnostic(gap_threshold, min_acceptable_score)
        curve_points: list[LearningCurvePoint] = []

        for size in train_sizes:
            x_train = x_all[:size]
            y_train = y_all[:size]

            if len(np.unique(y_train)) < 2:
                continue

            train_scores = cross_val_score(
                estimator, x_train, y_train, cv=min(cv, size), scoring=scoring
            )
            val_scores = cross_val_score(
                estimator, x_all, y_all, cv=cv, scoring=scoring
            )

            state = diagnostic.classify(
                float(train_scores.mean()), float(val_scores.mean())
            )

            curve_points.append(
                LearningCurvePoint(
                    train_size=size,
                    train_score_mean=round(float(train_scores.mean()), 6),
                    train_score_std=round(float(train_scores.std()), 6),
                    val_score_mean=round(float(val_scores.mean()), 6),
                    val_score_std=round(float(val_scores.std()), 6),
                    bias_variance_state=state,
                )
            )

        final_state = curve_points[-1].bias_variance_state if curve_points else "unknown"
        recommendation = self._build_recommendation(final_state)

        return {
            "curve": [
                {
                    "train_size": p.train_size,
                    "train_score_mean": p.train_score_mean,
                    "train_score_std": p.train_score_std,
                    "val_score_mean": p.val_score_mean,
                    "val_score_std": p.val_score_std,
                    "bias_variance_state": p.bias_variance_state,
                }
                for p in curve_points
            ],
            "final_diagnosis": final_state,
            "recommendation": recommendation,
            "scoring": scoring,
            "cv": cv,
            "n_observations": n_total,
            "gap_threshold": gap_threshold,
        }

    def _build_recommendation(self, state: str) -> str:
        """Map bias-variance state to an actionable recommendation.

        Args:
            state: Bias-variance state label.

        Returns:
            Actionable recommendation string.
        """
        recommendations: dict[str, str] = {
            "high_bias": (
                "Model is underfitting. Try a more complex model, add features, "
                "or reduce regularization."
            ),
            "high_variance": (
                "Model is overfitting. Add more training data, apply regularization, "
                "reduce model complexity, or use dropout."
            ),
            "good_fit": (
                "Model generalizes well. Consider fine-tuning hyperparameters "
                "or ensembling for marginal gains."
            ),
            "insufficient_data": (
                "Validation score still improving with more data. "
                "Collect more training examples."
            ),
            "unknown": "Could not determine diagnosis. Check data quality.",
        }
        return recommendations.get(state, "No recommendation available.")
