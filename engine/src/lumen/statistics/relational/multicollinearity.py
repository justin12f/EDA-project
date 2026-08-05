"""Variance Inflation Factor (VIF) multicollinearity detection."""

# #[AI_CONTEXT_START]
# - CONFIGURACIÓN DE FACTORY: Registrar este calculator en `RelationalStatisticsFactory` (backends pandas | polars | spark) y exponerlo mediante `StatisticsInyeccionDependency`, inyectada por la Factory Maestra de Agentes junto a las demás fábricas de dominio.
# - ABSTRACCIÓN DEL DATO: Mutar constructores y `analyze`/`compute` para recibir el contenedor abstracto del backend (`pd.DataFrame`, `pl.DataFrame`/`pl.LazyFrame`, `pyspark.sql.DataFrame`) inyectado por la factory; eliminar `np.ndarray`/`pd.Series` sueltos en firmas públicas.
# - REFACTOR NATIVO: Resolver métricas con expresiones 100 % nativas del backend activo (Polars: `.select`/`.group_by`/`.agg` sin `.collect()` salvo materialización acordada; PySpark: `pyspark.sql.functions` y ventanas distribuidas; Pandas: operaciones vectorizadas). No convertir a NumPy/Pandas desde backends no-pandas.
# #[AI_CONTEXT_END]
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from lumen.models.linear_regression import LinearRegression

@dataclass(frozen=True)
class VIFResult:
    """Immutable VIF result for a single feature."""

    feature: str
    vif: float
    risk_level: str

class VIFRiskClassifier:
    """Classifies multicollinearity risk based on VIF thresholds.

    Thresholds follow standard econometric convention:
        VIF < 5:   Acceptable
        VIF 5-10:  Moderate concern
        VIF > 10:  High multicollinearity — feature is nearly redundant
    """

    _THRESHOLDS: list[tuple[float, str]] = [
        (10.0, "high"),
        (5.0,  "moderate"),
        (0.0,  "acceptable"),
    ]

    def classify(self, vif: float) -> str:
        """Classify VIF value into a risk level.

        Args:
            vif: VIF value for a feature.

        Returns:
            Risk level string: 'high', 'moderate', or 'acceptable'.
        """
        for threshold, label in self._THRESHOLDS:
            if vif >= threshold:
                return label
        return "acceptable"

class SingleFeatureVIFCalculator:
    """Calculates VIF for one feature by regressing it on all others.

    VIF(Xⱼ) = 1 / (1 - R²ⱼ) where R²ⱼ is the coefficient of determination
    of the OLS regression of Xⱼ on all remaining features.
    """

    def calculate(
        self,
        feature_index: int,
        feature_matrix: np.ndarray,
    ) -> float:
        """Calculate VIF for a single feature column.

        Args:
            feature_index: Column index of the target feature.
            feature_matrix: Full numeric feature matrix (no intercept column).

        Returns:
            VIF value. Returns inf if R² = 1 (perfect collinearity).
        """
        n_features = feature_matrix.shape[1]
        predictor_indices = [i for i in range(n_features) if i != feature_index]

        x_others = feature_matrix[:, predictor_indices]
        y_target = feature_matrix[:, feature_index]

        model = LinearRegression(
            type_of_prediction="analytical",
            complexity="multiple",
        )
        model.fit(x=x_others, y=y_target)
        y_pred = model.predict(x_others)

        ss_res = float(np.sum((y_target - y_pred) ** 2))
        ss_tot = float(np.sum((y_target - y_target.mean()) ** 2))

        if ss_tot == 0.0:
            return float("inf")

        r_squared = 1.0 - ss_res / ss_tot

        if r_squared >= 1.0:
            return float("inf")

        return 1.0 / (1.0 - r_squared)

class MulticollinearityCalculator:
    """Calculates VIF for all features and detects multicollinearity.

    Workflow:
        calculator = MulticollinearityCalculator()
        result = calculator.calculate(
            data_frame=df[["age", "income", "credit_score"]],
            high_vif_threshold=10.0,  # optional
        )
    """

    _MINIMUM_FEATURES: int = 2
    _MINIMUM_OBSERVATIONS: int = 10

    def __init__(self) -> None:
        self._vif_calculator = SingleFeatureVIFCalculator()
        self._risk_classifier = VIFRiskClassifier()

    def calculate(
        self,
        data_frame: pd.DataFrame,
        high_vif_threshold: float = 10.0,
    ) -> dict:
        """Compute VIF for all columns in the DataFrame.

        Args:
            data_frame: Numeric-only DataFrame with features to analyse.
            high_vif_threshold: VIF value above which a feature is flagged.

        Returns:
            Dict with per-feature VIF, risk levels, and flagged features.

        Raises:
            ValueError: If fewer than 2 features or too few observations.
        """
        if data_frame.shape[1] < self._MINIMUM_FEATURES:
            raise ValueError(
                f"VIF analysis requires at least {self._MINIMUM_FEATURES} features. "
                f"Got {data_frame.shape[1]}."
            )
        if data_frame.shape[0] < self._MINIMUM_OBSERVATIONS:
            raise ValueError(
                f"At least {self._MINIMUM_OBSERVATIONS} observations required. "
                f"Got {data_frame.shape[0]}."
            )

        clean_df = data_frame.dropna()
        feature_matrix = clean_df.to_numpy(dtype=float)
        columns = clean_df.columns.tolist()

        vif_results: list[VIFResult] = [
            VIFResult(
                feature=columns[i],
                vif=self._vif_calculator.calculate(i, feature_matrix),
                risk_level=self._risk_classifier.classify(
                    self._vif_calculator.calculate(i, feature_matrix)
                ),
            )
            for i in range(len(columns))
        ]

        vif_results.sort(key=lambda r: r.vif, reverse=True)
        flagged = [r for r in vif_results if r.vif >= high_vif_threshold]

        return {
            "features": [
                {
                    "feature": r.feature,
                    "vif": round(r.vif, 4),
                    "risk_level": r.risk_level,
                }
                for r in vif_results
            ],
            "flagged_features": [r.feature for r in flagged],
            "n_high_vif": len(flagged),
            "high_vif_threshold": high_vif_threshold,
            "n_observations": len(clean_df),
            "recommendation": (
                "Remove or combine flagged features before fitting linear models."
                if flagged else
                "No critical multicollinearity detected."
            ),
        }
