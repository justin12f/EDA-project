"""Interaction effect detection between feature pairs relative to a target."""

# #[AI_CONTEXT_START]
# - CONFIGURACIÓN DE FACTORY: Registrar este calculator en `RelationalStatisticsFactory` (backends pandas | polars | spark) y exponerlo mediante `StatisticsInyeccionDependency`, inyectada por la Factory Maestra de Agentes junto a las demás fábricas de dominio.
# - ABSTRACCIÓN DEL DATO: Mutar constructores y `analyze`/`compute` para recibir el contenedor abstracto del backend (`pd.DataFrame`, `pl.DataFrame`/`pl.LazyFrame`, `pyspark.sql.DataFrame`) inyectado por la factory; eliminar `np.ndarray`/`pd.Series` sueltos en firmas públicas.
# - REFACTOR NATIVO: Resolver métricas con expresiones 100 % nativas del backend activo (Polars: `.select`/`.group_by`/`.agg` sin `.collect()` salvo materialización acordada; PySpark: `pyspark.sql.functions` y ventanas distribuidas; Pandas: operaciones vectorizadas). No convertir a NumPy/Pandas desde backends no-pandas.
# #[AI_CONTEXT_END]
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from lumen.models.linear_regression import LinearRegression
from lumen.evaluation.score import SquaredR
from lumen.preproccesing.scalers.implementations.standard_scaler import StandardScaler

@dataclass(frozen=True)
class InteractionResult:
    """Immutable result for a single feature pair interaction analysis."""

    feature_a: str
    feature_b: str
    r2_additive: float
    r2_with_interaction: float
    interaction_gain: float
    interaction_coefficient: float
    has_meaningful_interaction: bool

class AdditiveModelEvaluator:
    """Fits y ~ X_a + X_b and returns R²."""

    def evaluate(
        self,
        x_a: np.ndarray,
        x_b: np.ndarray,
        y: np.ndarray,
    ) -> float:
        """Fit additive model and return R².

        Args:
            x_a: Feature A array.
            x_b: Feature B array.
            y: Target array.

        Returns:
            R² of the additive model.
        """
        x = np.column_stack([x_a, x_b])
        model = LinearRegression(type_of_prediction="ordinary_least_squares", complexity="multiple")
        model.fit(x, y)
        return float(SquaredR().squared_r(y, model.predict(x)))

class InteractionModelEvaluator:
    """Fits y ~ X_a + X_b + X_a×X_b and returns R² + interaction coefficient."""

    def evaluate(
        self,
        x_a: np.ndarray,
        x_b: np.ndarray,
        y: np.ndarray,
    ) -> tuple[float, float]:
        """Fit interaction model and return (R², interaction_coefficient).

        Args:
            x_a: Feature A array.
            x_b: Feature B array.
            y: Target array.

        Returns:
            Tuple (R², coefficient of the interaction term X_a×X_b).
        """
        interaction_term = x_a * x_b
        x = np.column_stack([x_a, x_b, interaction_term])
        model = LinearRegression(type_of_prediction="ordinary_least_squares", complexity="multiple")
        model.fit(x, y)
        r2 = float(SquaredR().squared_r(y, model.predict(x)))
        interaction_coef = float(model.model.coefficients_[3])
        return r2, interaction_coef

class InteractionGainClassifier:
    """Classifies whether the R² gain from adding an interaction is meaningful."""

    def __init__(self, min_gain_threshold: float) -> None:
        self._threshold = min_gain_threshold

    def is_meaningful(self, r2_additive: float, r2_interaction: float) -> bool:
        """Determine if the R² improvement from the interaction is meaningful.

        Args:
            r2_additive: R² of the model without interaction.
            r2_interaction: R² of the model with interaction term.

        Returns:
            True if gain exceeds threshold.
        """
        return (r2_interaction - r2_additive) >= self._threshold

class InteractionEffectsCalculator:
    """Detects feature pairs whose interaction improves target prediction.

    For every pair of numeric features, fits two models:
        1. Additive:     y ~ A + B
        2. Interaction:  y ~ A + B + A×B

    If R² improves meaningfully, the pair has an interaction effect.

    Workflow:
        calculator = InteractionEffectsCalculator()
        result = calculator.calculate(
            data_frame=df,
            target_column="price",
            feature_columns=["sqft", "rooms", "age"],  # optional
            min_gain_threshold=0.01,                    # optional
            top_n=10,                                   # optional
        )
    """

    _MINIMUM_SAMPLES: int = 20
    _MINIMUM_FEATURES: int = 2

    def __init__(self) -> None:
        self._additive_evaluator = AdditiveModelEvaluator()
        self._interaction_evaluator = InteractionModelEvaluator()

    def calculate(
        self,
        data_frame: pd.DataFrame,
        target_column: str,
        feature_columns: list[str] | None = None,
        min_gain_threshold: float = 0.01,
        top_n: int | None = None,
    ) -> dict:
        """Detect interaction effects across all feature pairs.

        Args:
            data_frame: Source DataFrame.
            target_column: Numeric target variable.
            feature_columns: Feature columns to evaluate. Defaults to all
                numeric columns excluding target.
            min_gain_threshold: Minimum R² gain to flag as meaningful.
            top_n: Return only top N pairs by interaction gain.

        Returns:
            Dict with ranked interaction results and flagged pairs.

        Raises:
            KeyError: If target or any feature column is not found.
            ValueError: If fewer than 2 features or insufficient data.
        """
        if target_column not in data_frame.columns:
            raise KeyError(f"Target column '{target_column}' not found in DataFrame.")

        if feature_columns is not None:
            missing = [c for c in feature_columns if c not in data_frame.columns]
            if missing:
                raise KeyError(f"Feature columns not found: {missing}")
            numeric_features = feature_columns
        else:
            numeric_features = [
                col
                for col in data_frame.select_dtypes(include=[np.number]).columns
                if col != target_column
            ]

        if len(numeric_features) < self._MINIMUM_FEATURES:
            raise ValueError(
                f"At least {self._MINIMUM_FEATURES} numeric features required. "
                f"Got {len(numeric_features)}."
            )

        all_cols = numeric_features + [target_column]
        clean = data_frame[all_cols].dropna()

        if len(clean) < self._MINIMUM_SAMPLES:
            raise ValueError(
                f"At least {self._MINIMUM_SAMPLES} observations required. "
                f"Got {len(clean)}."
            )

        y = clean[target_column].to_numpy(dtype=float)
        scaler = StandardScaler()
        scaled = scaler.fit_transform(clean[numeric_features].to_numpy(dtype=float))

        gain_classifier = InteractionGainClassifier(min_gain_threshold)
        results: list[InteractionResult] = []

        for i, j in combinations(range(len(numeric_features)), 2):
            x_a = scaled[:, i]
            x_b = scaled[:, j]

            r2_additive = self._additive_evaluator.evaluate(x_a, x_b, y)
            r2_interaction, interaction_coef = self._interaction_evaluator.evaluate(
                x_a, x_b, y
            )

            gain = r2_interaction - r2_additive
            meaningful = gain_classifier.is_meaningful(r2_additive, r2_interaction)

            results.append(
                InteractionResult(
                    feature_a=numeric_features[i],
                    feature_b=numeric_features[j],
                    r2_additive=round(r2_additive, 6),
                    r2_with_interaction=round(r2_interaction, 6),
                    interaction_gain=round(gain, 6),
                    interaction_coefficient=round(interaction_coef, 6),
                    has_meaningful_interaction=meaningful,
                )
            )

        results.sort(key=lambda r: r.interaction_gain, reverse=True)

        if top_n is not None:
            results = results[:top_n]

        flagged = [r for r in results if r.has_meaningful_interaction]

        return {
            "pairs": [
                {
                    "feature_a": r.feature_a,
                    "feature_b": r.feature_b,
                    "r2_additive": r.r2_additive,
                    "r2_with_interaction": r.r2_with_interaction,
                    "interaction_gain": r.interaction_gain,
                    "interaction_coefficient": r.interaction_coefficient,
                    "has_meaningful_interaction": r.has_meaningful_interaction,
                }
                for r in results
            ],
            "meaningful_pairs": [
                {
                    "feature_a": r.feature_a,
                    "feature_b": r.feature_b,
                    "gain": r.interaction_gain,
                }
                for r in flagged
            ],
            "n_pairs_evaluated": len(results),
            "n_meaningful": len(flagged),
            "min_gain_threshold": min_gain_threshold,
            "target_column": target_column,
            "n_observations": len(clean),
        }
