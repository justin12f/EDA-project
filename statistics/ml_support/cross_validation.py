"""K-Fold cross-validation with stratification and repeated CV support."""

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
from sklearn.model_selection import (
    KFold,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_score,
)

@dataclass(frozen=True)
class CVResult:
    """Immutable cross-validation result."""

    strategy: str
    scoring: str
    fold_scores: list[float]
    mean_score: float
    std_score: float
    min_score: float
    max_score: float
    confidence_interval_lower: float
    confidence_interval_upper: float
    n_folds: int
    n_repeats: int

class CVStrategySelector:
    """Selects the appropriate CV strategy based on task and configuration.

    Strategies:
        - 'kfold':                Standard K-Fold.
        - 'stratified_kfold':     Preserves class proportions per fold.
        - 'repeated_kfold':       K-Fold repeated n times with different splits.
        - 'repeated_stratified':  Repeated stratified K-Fold.
    """

    _VALID_STRATEGIES: frozenset[str] = frozenset(
        {"kfold", "stratified_kfold", "repeated_kfold", "repeated_stratified"}
    )

    def select(
        self,
        strategy: str,
        n_folds: int,
        n_repeats: int,
        random_seed: int,
    ):
        """Instantiate and return the appropriate sklearn CV object.

        Args:
            strategy: CV strategy key.
            n_folds: Number of folds.
            n_repeats: Number of repetitions (repeated strategies only).
            random_seed: Random seed for shuffle reproducibility.

        Returns:
            Sklearn cross-validator object.

        Raises:
            KeyError: If strategy is not recognized.
        """
        if strategy not in self._VALID_STRATEGIES:
            raise KeyError(
                f"strategy '{strategy}' not recognized. "
                f"Available: {self._VALID_STRATEGIES}"
            )

        if strategy == "kfold":
            return KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
        if strategy == "stratified_kfold":
            return StratifiedKFold(
                n_splits=n_folds, shuffle=True, random_state=random_seed
            )
        if strategy == "repeated_kfold":
            return RepeatedKFold(
                n_splits=n_folds, n_repeats=n_repeats, random_state=random_seed
            )
        return RepeatedStratifiedKFold(
            n_splits=n_folds, n_repeats=n_repeats, random_state=random_seed
        )

class CVConfidenceIntervalCalculator:
    """Computes a t-distribution confidence interval for CV score mean."""

    def calculate(
        self,
        scores: np.ndarray,
        confidence_level: float,
    ) -> tuple[float, float]:
        """Compute CI for the mean CV score.

        Args:
            scores: Array of per-fold scores.
            confidence_level: Desired confidence level.

        Returns:
            Tuple (lower_bound, upper_bound).
        """
        from scipy import stats as scipy_stats
        n = len(scores)
        mean = float(scores.mean())
        se = float(scipy_stats.sem(scores))
        alpha = 1.0 - confidence_level
        t_crit = float(scipy_stats.t.ppf(1 - alpha / 2, df=n - 1))
        margin = t_crit * se
        return mean - margin, mean + margin

class CrossValidationCalculator:
    """Flexible K-Fold cross-validation with CI and repeated CV support.

    Workflow:
        from sklearn.ensemble import RandomForestClassifier

        calculator = CrossValidationCalculator()
        result = calculator.calculate(
            data_frame=df,
            target_column="churn",
            estimator=RandomForestClassifier(n_estimators=100),
            feature_columns=["age", "income"],  # optional
            strategy="stratified_kfold",
            n_folds=5,
            n_repeats=3,                        # repeated strategies only
            scoring="f1_weighted",
            confidence_level=0.95,
            random_seed=42,
        )
    """

    _MINIMUM_SAMPLES: int = 10

    def __init__(self) -> None:
        self._strategy_selector = CVStrategySelector()
        self._ci_calculator = CVConfidenceIntervalCalculator()

    def calculate(
        self,
        data_frame: pd.DataFrame,
        target_column: str,
        estimator: BaseEstimator,
        feature_columns: list[str] | None = None,
        strategy: str = "stratified_kfold",
        n_folds: int = 5,
        n_repeats: int = 3,
        scoring: str = "accuracy",
        confidence_level: float = 0.95,
        random_seed: int = 42,
        n_jobs: int = -1,
    ) -> dict:
        """Run cross-validation with the specified strategy.

        Args:
            data_frame: Source DataFrame.
            target_column: Target variable column name.
            estimator: Sklearn-compatible estimator (unfitted).
            feature_columns: Feature subset. Defaults to all numeric.
            strategy: CV strategy key.
            n_folds: Number of folds.
            n_repeats: Repetitions for repeated strategies.
            scoring: Sklearn scoring string.
            confidence_level: CI confidence level.
            random_seed: Seed for reproducibility.
            n_jobs: Parallel jobs for cross_val_score (-1 = all cores).

        Returns:
            Dict with fold scores, mean/std, CI, and model summary.

        Raises:
            KeyError: If columns or strategy are invalid.
            ValueError: If data is insufficient.
        """
        if target_column not in data_frame.columns:
            raise KeyError(f"Target column '{target_column}' not found.")
        if n_folds < 2:
            raise ValueError(f"n_folds must be >= 2. Got {n_folds}.")
        if n_repeats < 1:
            raise ValueError(f"n_repeats must be >= 1. Got {n_repeats}.")
        if not 0.0 < confidence_level < 1.0:
            raise ValueError(
                f"confidence_level must be in (0, 1). Got {confidence_level}."
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
        if len(clean) < n_folds:
            raise ValueError(
                f"n_folds ({n_folds}) cannot exceed n_observations ({len(clean)})."
            )

        x = clean[features_to_use].to_numpy(dtype=float)
        y = clean[target_column].to_numpy()

        cv_object = self._strategy_selector.select(
            strategy, n_folds, n_repeats, random_seed
        )
        scores = cross_val_score(
            estimator, x, y, cv=cv_object, scoring=scoring, n_jobs=n_jobs
        )

        ci_lower, ci_upper = self._ci_calculator.calculate(scores, confidence_level)
        total_folds = len(scores)
        actual_repeats = total_folds // n_folds

        cv_result = CVResult(
            strategy=strategy,
            scoring=scoring,
            fold_scores=[round(float(s), 6) for s in scores],
            mean_score=round(float(scores.mean()), 6),
            std_score=round(float(scores.std()), 6),
            min_score=round(float(scores.min()), 6),
            max_score=round(float(scores.max()), 6),
            confidence_interval_lower=round(ci_lower, 6),
            confidence_interval_upper=round(ci_upper, 6),
            n_folds=n_folds,
            n_repeats=actual_repeats,
        )

        return {
            "strategy": cv_result.strategy,
            "scoring": cv_result.scoring,
            "fold_scores": cv_result.fold_scores,
            "mean_score": cv_result.mean_score,
            "std_score": cv_result.std_score,
            "min_score": cv_result.min_score,
            "max_score": cv_result.max_score,
            "confidence_interval": {
                "lower": cv_result.confidence_interval_lower,
                "upper": cv_result.confidence_interval_upper,
                "confidence_level": confidence_level,
            },
            "n_folds": cv_result.n_folds,
            "n_repeats": cv_result.n_repeats,
            "total_fits": total_folds,
            "n_observations": len(clean),
            "n_features": len(features_to_use),
        }
