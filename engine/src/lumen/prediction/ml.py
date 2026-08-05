"""Learned models, wrapped from scikit-learn.

Thin wrappers on purpose. The value this layer adds is not reimplementing
gradient boosting — it is that every model here reports the same `FitReport`,
raises the same errors on too little data, and is reachable through the same
registry as a least-squares line. A person comparing "polynomial vs random
forest" should not have to learn two APIs to do it.

`random_state` is pinned by default. A proposal a human approves must produce
the same model when the worker runs it; a fit that differs run to run cannot be
reviewed.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from lumen.prediction.base import (
    Family,
    FitReport,
    Prediction,
    Predictor,
    Task,
    check_training_data,
)

DEFAULT_RANDOM_STATE = 20260804


class SklearnPredictor(Predictor):
    """Base for the sklearn family. Subclasses only declare how to build one."""

    family = Family.ML
    minimum_samples = 5

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self._model: Any = None
        self._classes: np.ndarray | None = None

    def _build(self) -> Any:  # pragma: no cover - overridden
        raise NotImplementedError

    def fit(self, X: np.ndarray, y: np.ndarray) -> FitReport:
        if self.task is Task.CLASSIFICATION:
            X = np.asarray(X, dtype=float)
            y = np.asarray(y).ravel()
            if X.shape[0] != y.shape[0]:
                raise ValueError(f"X has {X.shape[0]} rows but y has {y.shape[0]}")
            if X.shape[0] < self.minimum_samples:
                raise ValueError(
                    f"Need at least {self.minimum_samples} rows to fit {self.name}; "
                    f"got {X.shape[0]}."
                )
            self._classes = np.unique(y)
            if self._classes.size < 2:
                raise ValueError(
                    f"{self.name} needs at least two classes; the target has one value. "
                    "A classifier over one class predicts a constant."
                )
        else:
            X, y = check_training_data(X, y, minimum=self.minimum_samples)

        self._model = self._build()
        self._model.fit(X, y)
        self._fitted = True

        diagnostics: dict[str, Any] = {}
        if hasattr(self._model, "feature_importances_"):
            diagnostics["feature_importances"] = [
                float(v) for v in self._model.feature_importances_
            ]
        if hasattr(self._model, "coef_"):
            diagnostics["coefficients"] = np.asarray(self._model.coef_).ravel().tolist()
        if self._classes is not None:
            diagnostics["classes"] = [str(c) for c in self._classes]

        return FitReport(
            predictor=self.name,
            family=self.family,
            task=self.task,
            n_samples=int(X.shape[0]),
            n_features=int(X.shape[1]),
            params=dict(self.params),
            diagnostics=diagnostics,
        )

    def predict(self, X: np.ndarray) -> Prediction:
        self._require_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return Prediction(values=np.asarray(self._model.predict(X)))


# ── regression ──────────────────────────────────────────────────────────────


class RidgeRegression(SklearnPredictor):
    """Linear regression with L2 shrinkage. The safe default when features correlate."""

    name = "ridge"
    task = Task.REGRESSION

    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__(alpha=alpha)

    def _build(self):
        from sklearn.linear_model import Ridge

        return Ridge(alpha=self.params["alpha"])


class LassoRegression(SklearnPredictor):
    """Linear regression with L1 shrinkage. Drives weak coefficients to exactly zero,
    so it doubles as feature selection."""

    name = "lasso"
    task = Task.REGRESSION

    def __init__(self, alpha: float = 0.1) -> None:
        super().__init__(alpha=alpha)

    def _build(self):
        from sklearn.linear_model import Lasso

        return Lasso(alpha=self.params["alpha"], max_iter=10_000)


class ElasticNetRegression(SklearnPredictor):
    """L1 and L2 combined. Lasso's selection without its instability when features
    are correlated."""

    name = "elastic_net"
    task = Task.REGRESSION

    def __init__(self, alpha: float = 0.1, l1_ratio: float = 0.5) -> None:
        super().__init__(alpha=alpha, l1_ratio=l1_ratio)

    def _build(self):
        from sklearn.linear_model import ElasticNet

        return ElasticNet(
            alpha=self.params["alpha"],
            l1_ratio=self.params["l1_ratio"],
            max_iter=10_000,
        )


class RandomForestRegression(SklearnPredictor):
    """Averaged decision trees. Handles non-linearity and interactions without
    feature engineering, and reports which columns mattered."""

    name = "random_forest"
    task = Task.REGRESSION

    def __init__(self, n_estimators: int = 200, max_depth: int | None = None) -> None:
        super().__init__(n_estimators=n_estimators, max_depth=max_depth)

    def _build(self):
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(
            n_estimators=self.params["n_estimators"],
            max_depth=self.params["max_depth"],
            random_state=DEFAULT_RANDOM_STATE,
            n_jobs=-1,
        )


class GradientBoostingRegression(SklearnPredictor):
    """Sequentially corrected trees. Usually the strongest tabular regressor here,
    and the slowest to fit."""

    name = "gradient_boosting"
    task = Task.REGRESSION

    def __init__(self, n_estimators: int = 200, learning_rate: float = 0.1) -> None:
        super().__init__(n_estimators=n_estimators, learning_rate=learning_rate)

    def _build(self):
        from sklearn.ensemble import GradientBoostingRegressor

        return GradientBoostingRegressor(
            n_estimators=self.params["n_estimators"],
            learning_rate=self.params["learning_rate"],
            random_state=DEFAULT_RANDOM_STATE,
        )


class SupportVectorRegression(SklearnPredictor):
    """Kernel regression with an insensitivity margin. Good on small, smooth,
    non-linear data; poor above a few thousand rows."""

    name = "svr"
    task = Task.REGRESSION

    def __init__(self, kernel: str = "rbf", C: float = 1.0) -> None:
        super().__init__(kernel=kernel, C=C)

    def _build(self):
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVR

        # SVR is scale-sensitive to the point of uselessness on raw features.
        # Bundling the scaler means a caller cannot forget it.
        return make_pipeline(
            StandardScaler(), SVR(kernel=self.params["kernel"], C=self.params["C"])
        )


class KNearestRegression(SklearnPredictor):
    """Averages the k nearest points. No training, no assumptions, and no
    extrapolation beyond the observed neighbourhood."""

    name = "knn"
    task = Task.REGRESSION

    def __init__(self, n_neighbors: int = 5) -> None:
        super().__init__(n_neighbors=n_neighbors)

    def _build(self):
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        # Distance-based, so unscaled features let whichever column has the
        # largest units decide every neighbour.
        return make_pipeline(
            StandardScaler(),
            KNeighborsRegressor(n_neighbors=self.params["n_neighbors"]),
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> FitReport:
        neighbours = self.params["n_neighbors"]
        rows = np.asarray(X).shape[0]
        if rows < neighbours:
            raise ValueError(
                f"knn with n_neighbors={neighbours} needs at least that many rows; got {rows}."
            )
        return super().fit(X, y)


# ── classification ──────────────────────────────────────────────────────────


class LogisticClassification(SklearnPredictor):
    """Linear classifier. Coefficients are readable, which matters when a person
    has to defend the decision."""

    name = "logistic"
    task = Task.CLASSIFICATION

    def __init__(self, C: float = 1.0) -> None:
        super().__init__(C=C)

    def _build(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=self.params["C"], max_iter=1000),
        )


class RandomForestClassification(SklearnPredictor):
    """Averaged decision trees for classification. Robust to unscaled and mixed
    features."""

    name = "random_forest_classifier"
    task = Task.CLASSIFICATION

    def __init__(self, n_estimators: int = 200, max_depth: int | None = None) -> None:
        super().__init__(n_estimators=n_estimators, max_depth=max_depth)

    def _build(self):
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=self.params["n_estimators"],
            max_depth=self.params["max_depth"],
            random_state=DEFAULT_RANDOM_STATE,
            n_jobs=-1,
        )


class GradientBoostingClassification(SklearnPredictor):
    """Sequentially corrected trees for classification."""

    name = "gradient_boosting_classifier"
    task = Task.CLASSIFICATION

    def __init__(self, n_estimators: int = 200, learning_rate: float = 0.1) -> None:
        super().__init__(n_estimators=n_estimators, learning_rate=learning_rate)

    def _build(self):
        from sklearn.ensemble import GradientBoostingClassifier

        return GradientBoostingClassifier(
            n_estimators=self.params["n_estimators"],
            learning_rate=self.params["learning_rate"],
            random_state=DEFAULT_RANDOM_STATE,
        )
