"""Module to calculate regression model scores.

Backend-agnostic — uses only numpy. Accepts numpy arrays directly.
Callers convert from their backend before calling.
"""

# #[AI_CONTEXT_START]
# - CONFIGURACIÓN DE FACTORY: `EvaluationScoreFactory` (pandas | polars | spark) en `EvaluationInyeccionDependency`, inyectada por Factory Maestra.
# - ABSTRACCIÓN DEL DATO: Métricas sobre columnas o vectores materializados del backend, no solo `np.ndarray` en API pública.
# - REFACTOR NATIVO: MSE/MAE/R² con expresiones nativas del backend activo.
# #[AI_CONTEXT_END]
from __future__ import annotations
import numpy as np

class MeanSquareError:
    """Calculate the Mean Squared Error."""

    def mean_square_error(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> float:
        """Calculate the MSE value.

        Args:
            y_true: Ground truth values (1-D array).
            y_pred: Predicted values (1-D array).

        Returns:
            Mean squared error as a float.
        """
        y_true = np.ravel(y_true)
        y_pred = np.ravel(y_pred)
        return float(np.mean((y_true - y_pred) ** 2))

class RootMeanSquareError:
    """Calculate the Root Mean Squared Error."""

    def root_mean_square_error(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> float:
        """Calculate the RMSE value.

        Args:
            y_true: Ground truth values (1-D array).
            y_pred: Predicted values (1-D array).

        Returns:
            Root mean squared error as a float.
        """
        mse = MeanSquareError().mean_square_error(y_true, y_pred)
        return float(np.sqrt(mse))

class SquaredR:
    """Calculate the R² (coefficient of determination)."""

    def squared_r(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> float:
        """Calculate the R² value.

        Args:
            y_true: Ground truth values (1-D array).
            y_pred: Predicted values (1-D array).

        Returns:
            R² value as a float.
        """
        y_true = np.ravel(y_true)
        y_pred = np.ravel(y_pred)
        y_true_mean = y_true.mean()
        numerator = np.sum((y_pred - y_true) ** 2)
        denominator = np.sum((y_true - y_true_mean) ** 2)
        return float(1 - (numerator / denominator))

class Score:
    """Calculate all regression model scores."""

    def get_score(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> dict[str, float]:
        """Calculate MSE, RMSE, and R² metrics.

        Args:
            y_true: Ground truth values (1-D array).
            y_pred: Predicted values (1-D array).

        Returns:
            Dict with keys ``mean_square_error``, ``root_mean_square_error``,
            and ``squared_r``.
        """
        return {
            "mean_square_error": MeanSquareError().mean_square_error(y_true, y_pred),
            "root_mean_square_error": RootMeanSquareError().root_mean_square_error(y_true, y_pred),
            "squared_r": SquaredR().squared_r(y_true, y_pred),
        }
