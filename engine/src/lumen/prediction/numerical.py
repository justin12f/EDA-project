"""Numerical methods: closed-form fits and deterministic extrapolation.

No training loop, no hyperparameter search, no randomness. Given the same
points these return the same coefficients, which is why they belong in a product
where a person has to approve what the agent proposes: a least-squares line can
be checked by hand, and a gradient-boosted ensemble cannot.
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


class LeastSquares(Predictor):
    """Ordinary least squares via SVD.

    `lstsq` rather than the normal equations: inverting XᵀX squares the condition
    number, so a mildly collinear design that SVD handles cleanly comes back as
    numerical noise. The cost difference is irrelevant at any size where a person
    is waiting.
    """

    name = "least_squares"
    family = Family.NUMERICAL
    task = Task.REGRESSION

    def __init__(self, fit_intercept: bool = True) -> None:
        super().__init__(fit_intercept=fit_intercept)
        self._coefficients: np.ndarray | None = None

    def _design(self, X: np.ndarray) -> np.ndarray:
        if not self.params["fit_intercept"]:
            return X
        return np.hstack([np.ones((X.shape[0], 1)), X])

    def fit(self, X: np.ndarray, y: np.ndarray) -> FitReport:
        X, y = check_training_data(X, y, minimum=2)
        design = self._design(X)
        coefficients, residuals, rank, singular = np.linalg.lstsq(design, y, rcond=None)
        self._coefficients = coefficients
        self._fitted = True

        condition = (
            float(singular.max() / singular.min())
            if singular.size and singular.min() > 0
            else float("inf")
        )
        return FitReport(
            predictor=self.name,
            family=self.family,
            task=self.task,
            n_samples=X.shape[0],
            n_features=X.shape[1],
            params=dict(self.params),
            diagnostics={
                "coefficients": [float(c) for c in coefficients],
                "rank": int(rank),
                # Above ~1e10 the fit is numerically meaningless and the caller
                # should know, rather than reading confident nonsense.
                "condition_number": condition,
                "rank_deficient": bool(rank < design.shape[1]),
            },
        )

    def predict(self, X: np.ndarray) -> Prediction:
        self._require_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return Prediction(values=self._design(X) @ self._coefficients)


class PolynomialFit(Predictor):
    """Least-squares polynomial of a chosen degree, on one feature."""

    name = "polynomial"
    family = Family.NUMERICAL
    task = Task.REGRESSION

    def __init__(self, degree: int = 2) -> None:
        if degree < 1:
            raise ValueError("degree must be at least 1")
        super().__init__(degree=degree)
        self._coefficients: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> FitReport:
        # A low minimum here on purpose: this call is for the shape check and
        # for dropping null targets. The degree-aware guard below owns the
        # "too few points" case, because it can say why rather than just how many.
        X, y = check_training_data(X, y, minimum=2)
        if X.shape[1] != 1:
            raise ValueError(
                f"polynomial fits one feature; got {X.shape[1]}. "
                "Use least_squares or an ML predictor for several."
            )

        degree = self.params["degree"]
        if X.shape[0] <= degree + 1:
            raise ValueError(
                f"degree {degree} needs more than {degree + 1} points; got {X.shape[0]}. "
                "A polynomial through every point is memorisation, not a fit."
            )

        self._coefficients = np.polyfit(X.ravel(), y, degree)
        self._fitted = True
        return FitReport(
            predictor=self.name,
            family=self.family,
            task=self.task,
            n_samples=X.shape[0],
            n_features=1,
            params=dict(self.params),
            diagnostics={"coefficients": [float(c) for c in self._coefficients]},
        )

    def predict(self, X: np.ndarray) -> Prediction:
        self._require_fitted()
        values = np.polyval(self._coefficients, np.asarray(X, dtype=float).ravel())
        return Prediction(values=values)


class LinearInterpolation(Predictor):
    """Piecewise-linear interpolation between observed points.

    Interpolation, not extrapolation: outside the observed range this holds the
    endpoint value rather than continuing the last slope. Extending a local
    gradient into unobserved territory is how interpolators produce confident
    nonsense, and the flat answer at least looks like the non-answer it is.
    """

    name = "linear_interpolation"
    family = Family.NUMERICAL
    task = Task.REGRESSION

    def __init__(self) -> None:
        super().__init__()
        self._x: np.ndarray | None = None
        self._y: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> FitReport:
        X, y = check_training_data(X, y, minimum=2)
        if X.shape[1] != 1:
            raise ValueError(f"interpolation takes one feature; got {X.shape[1]}")

        order = np.argsort(X.ravel())
        self._x, self._y = X.ravel()[order], y[order]
        self._fitted = True
        return FitReport(
            predictor=self.name,
            family=self.family,
            task=self.task,
            n_samples=X.shape[0],
            n_features=1,
            params={},
            diagnostics={"x_min": float(self._x[0]), "x_max": float(self._x[-1])},
        )

    def predict(self, X: np.ndarray) -> Prediction:
        self._require_fitted()
        values = np.interp(np.asarray(X, dtype=float).ravel(), self._x, self._y)
        return Prediction(values=values)


class CubicSpline(Predictor):
    """Natural cubic spline. Smoother than linear where the truth is smooth."""

    name = "cubic_spline"
    family = Family.NUMERICAL
    task = Task.REGRESSION

    def __init__(self) -> None:
        super().__init__()
        self._spline: Any = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> FitReport:
        from scipy.interpolate import CubicSpline as SciPyCubicSpline

        X, y = check_training_data(X, y, minimum=4)
        if X.shape[1] != 1:
            raise ValueError(f"spline takes one feature; got {X.shape[1]}")

        order = np.argsort(X.ravel())
        x_sorted, y_sorted = X.ravel()[order], y[order]
        # Duplicate abscissae make the system singular; averaging their targets
        # is the standard, and the only, sane resolution.
        unique_x, inverse = np.unique(x_sorted, return_inverse=True)
        if unique_x.size != x_sorted.size:
            y_sorted = np.bincount(inverse, weights=y_sorted) / np.bincount(inverse)
            x_sorted = unique_x
        if x_sorted.size < 4:
            raise ValueError("a cubic spline needs at least 4 distinct x values")

        self._spline = SciPyCubicSpline(x_sorted, y_sorted, bc_type="natural")
        self._fitted = True
        return FitReport(
            predictor=self.name,
            family=self.family,
            task=self.task,
            n_samples=x_sorted.size,
            n_features=1,
            params={},
            diagnostics={"knots": int(x_sorted.size)},
        )

    def predict(self, X: np.ndarray) -> Prediction:
        self._require_fitted()
        return Prediction(values=self._spline(np.asarray(X, dtype=float).ravel()))


class MovingAverage(Predictor):
    """Trailing mean of the last `window` observations.

    The baseline every other method should have to beat. It is here because a
    forecast that cannot outperform "roughly what it was recently" is not
    earning its complexity, and without the baseline on the table nobody checks.
    """

    name = "moving_average"
    family = Family.NUMERICAL
    task = Task.FORECAST
    univariate = True

    def __init__(self, window: int = 3) -> None:
        if window < 1:
            raise ValueError("window must be at least 1")
        super().__init__(window=window)
        self._level: float | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> FitReport:
        X, y = check_training_data(X, y, minimum=1)
        window = min(self.params["window"], y.size)
        self._level = float(np.mean(y[-window:]))
        self._fitted = True
        return FitReport(
            predictor=self.name,
            family=self.family,
            task=self.task,
            n_samples=y.size,
            n_features=0,
            params=dict(self.params),
            diagnostics={"level": self._level, "effective_window": window},
        )

    def predict(self, X: np.ndarray) -> Prediction:
        self._require_fitted()
        steps = np.asarray(X).reshape(-1).size
        return Prediction(values=np.full(steps, self._level, dtype=float))
