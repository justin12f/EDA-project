"""Prediction contracts.

One deliberate departure from the rest of the engine: predictors are **not**
keyed by backend.

Everywhere else — readers, cleaning steps, statistics — the three backends have
genuinely different native implementations, because null handling and group-by
differ between pandas, polars and Spark. Ridge regression does not. There is one
implementation, it operates on numpy arrays, and registering the same class three
times would be cargo-culting a pattern rather than applying it.

The backend axis lives instead in `extract.py`, which turns any backend's frame
into arrays. That is where the difference actually is.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np


class Task(StrEnum):
    """What a predictor is for. Determines which metrics make sense."""

    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    FORECAST = "forecast"


class Family(StrEnum):
    """How a predictor works — the axis a person chooses along.

    NUMERICAL   closed-form or deterministic fits: least squares, polynomial,
                interpolation. Cheap, explainable, no training loop.
    TIMESERIES  methods that assume the row order is time and extrapolate it.
    ML          learned models with hyperparameters and a fit/predict cycle.
    """

    NUMERICAL = "numerical"
    TIMESERIES = "timeseries"
    ML = "ml"


@dataclass(frozen=True)
class FitReport:
    """What happened during fitting. Returned, never printed."""

    predictor: str
    family: Family
    task: Task
    n_samples: int
    n_features: int
    params: dict[str, Any] = field(default_factory=dict)
    # Fitted coefficients, chosen smoothing constants, feature importances —
    # whatever this method can say about itself. Empty is honest for a KNN.
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Prediction:
    values: np.ndarray
    # Present only for methods that produce one honestly. A model that cannot
    # quantify its uncertainty says None rather than inventing a band.
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None

    def as_list(self) -> list[float]:
        return [float(v) for v in np.asarray(self.values).ravel()]


class Predictor(ABC):
    """Fit on arrays, predict arrays.

    Implementations must not mutate their inputs and must raise before fitting
    rather than producing a model from data that cannot support one — a
    silently-fitted model on three points is worse than a refusal.
    """

    name: str
    family: Family
    task: Task
    # Forecasters take a single series and a horizon; the rest take a design
    # matrix. The distinction changes the call signature, so it is declared
    # rather than inferred.
    univariate: bool = False

    def __init__(self, **params: Any) -> None:
        self.params = params
        self._fitted = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> FitReport:
        """Fit on `X` (n_samples, n_features) and `y` (n_samples,)."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> Prediction:
        """Predict for `X`. For univariate forecasters `X` carries the horizon."""

    def forecast(self, horizon: int) -> Prediction:
        """Extrapolate `horizon` steps beyond the fitted series."""
        if not self.univariate:
            raise NotImplementedError(
                f"{self.name} needs a feature matrix — call predict(X). "
                "forecast(horizon) is for univariate time-series methods."
            )
        return self.predict(np.arange(horizon, dtype=float).reshape(-1, 1))

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(f"{self.name} has not been fitted — call fit() first.")


def check_training_data(X: np.ndarray, y: np.ndarray, minimum: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Validate and normalise a training set.

    Rows with a NaN in the target are dropped rather than imputed: inventing a
    target value is fabricating the thing being learned. NaNs in features are
    left to the caller's cleaning pipeline, which is where that decision belongs
    and where a human approved it.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()

    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X has {X.shape[0]} rows but y has {y.shape[0]}")

    keep = ~np.isnan(y)
    X, y = X[keep], y[keep]

    if X.shape[0] < minimum:
        raise ValueError(
            f"Need at least {minimum} rows with a non-null target to fit; got {X.shape[0]}. "
            "A model fitted on fewer is not a model."
        )
    return X, y
