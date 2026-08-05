"""Backtesting and model comparison.

The point of this module is that a number without a holdout is not evidence.
Every predictor here is scored on data it did not see, and the split respects
what kind of data it is: shuffling a time series to make a "fair" split leaks the
future into the training set and produces a model that looks excellent and
forecasts nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from lumen.prediction.base import FitReport, Predictor, Task
from lumen.prediction.metrics import Metric, primary_metric, score
from lumen.prediction.registry import PredictorRegistry


@dataclass(frozen=True)
class Evaluation:
    predictor: str
    task: Task
    metrics: dict[str, Metric]
    n_train: int
    n_test: int
    fit_report: FitReport | None = None
    error: str | None = None

    @property
    def failed(self) -> bool:
        return self.error is not None

    def primary(self) -> float:
        metric = self.metrics.get(primary_metric(self.task))
        return metric.value if metric else float("nan")


@dataclass
class Comparison:
    task: Task
    ranked: list[Evaluation] = field(default_factory=list)

    @property
    def best(self) -> Evaluation | None:
        return self.ranked[0] if self.ranked and not self.ranked[0].failed else None

    def summary(self) -> list[dict[str, Any]]:
        return [
            {
                "predictor": evaluation.predictor,
                "error": evaluation.error,
                **{name: round(m.value, 6) for name, m in evaluation.metrics.items()},
            }
            for evaluation in self.ranked
        ]


def split(
    X: np.ndarray, y: np.ndarray, test_fraction: float = 0.2, *, ordered: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Hold out a test set.

    `ordered=True` takes the tail rather than a random sample. For anything where
    row order is time, a random split trains on the future and tests on the past,
    which inflates every metric and is the single most common way a forecast
    evaluation lies.
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"test_fraction must be in (0, 1); got {test_fraction}")

    n = X.shape[0]
    n_test = max(1, int(round(n * test_fraction)))
    if n - n_test < 1:
        raise ValueError(f"{n} rows cannot be split with test_fraction={test_fraction}")

    if ordered:
        return X[:-n_test], X[-n_test:], y[:-n_test], y[-n_test:]

    rng = np.random.default_rng(20260804)
    order = rng.permutation(n)
    test_index, train_index = order[:n_test], order[n_test:]
    return X[train_index], X[test_index], y[train_index], y[test_index]


def evaluate(
    predictor: Predictor,
    X: np.ndarray,
    y: np.ndarray,
    *,
    test_fraction: float = 0.2,
) -> Evaluation:
    """Fit on a training split, score on the holdout.

    A predictor that raises produces a failed Evaluation rather than an
    exception: in a comparison of nine methods, one that cannot fit this data is
    a result about that method, not a reason to abandon the other eight.
    """
    ordered = predictor.univariate
    try:
        X_train, X_test, y_train, y_test = split(
            X, y, test_fraction, ordered=ordered
        )
        report = predictor.fit(X_train, y_train)
        prediction = (
            predictor.forecast(len(y_test))
            if predictor.univariate
            else predictor.predict(X_test)
        )
        metrics = score(y_test, prediction.values, predictor.task)
        return Evaluation(
            predictor=predictor.name,
            task=predictor.task,
            metrics=metrics,
            n_train=int(X_train.shape[0]),
            n_test=int(X_test.shape[0]),
            fit_report=report,
        )
    except Exception as exc:  # noqa: BLE001 — see docstring
        return Evaluation(
            predictor=predictor.name,
            task=predictor.task,
            metrics={},
            n_train=0,
            n_test=0,
            error=f"{type(exc).__name__}: {exc}",
        )


def backtest(
    predictor: Predictor, y: np.ndarray, *, folds: int = 3, horizon: int = 1
) -> Evaluation:
    """Rolling-origin evaluation for a forecaster.

    Each fold trains on everything up to a cut and forecasts the next `horizon`
    steps — the only honest way to ask "how would this have done", because it is
    the only one where the model never sees a value it is about to predict.
    """
    if not predictor.univariate:
        raise ValueError(f"{predictor.name} is not a univariate forecaster")

    y = np.asarray(y, dtype=float).ravel()
    y = y[~np.isnan(y)]

    minimum = folds * horizon + 2
    if y.size < minimum:
        raise ValueError(
            f"backtesting {folds} folds at horizon {horizon} needs at least "
            f"{minimum} points; got {y.size}."
        )

    truths: list[float] = []
    forecasts: list[float] = []
    for fold in range(folds):
        cut = y.size - (folds - fold) * horizon
        train, actual = y[:cut], y[cut : cut + horizon]
        predictor.fit(np.arange(cut, dtype=float).reshape(-1, 1), train)
        predicted = predictor.forecast(horizon).values
        truths.extend(actual.tolist())
        forecasts.extend(np.asarray(predicted).ravel().tolist()[: actual.size])

    metrics = score(np.array(truths), np.array(forecasts), predictor.task)
    return Evaluation(
        predictor=predictor.name,
        task=predictor.task,
        metrics=metrics,
        n_train=int(y.size - folds * horizon),
        n_test=len(truths),
    )


def compare(
    X: np.ndarray,
    y: np.ndarray,
    *,
    task: Task,
    candidates: list[str] | None = None,
    test_fraction: float = 0.2,
) -> Comparison:
    """Score every candidate on the same holdout and rank them.

    The ranking direction comes from the metric, not from a hardcoded list —
    lower RMSE wins, higher F1 wins, and a table sorted the wrong way is worse
    than no table.
    """
    names = candidates or PredictorRegistry.names(task=task)
    if not names:
        raise ValueError(f"no registered predictors for task '{task}'")

    evaluations = [
        evaluate(PredictorRegistry.create(name), X, y, test_fraction=test_fraction)
        for name in names
    ]

    key = primary_metric(task)
    higher_is_better = any(
        m.higher_is_better
        for evaluation in evaluations
        for name, m in evaluation.metrics.items()
        if name == key
    )

    def sort_key(evaluation: Evaluation) -> tuple[int, float]:
        metric = evaluation.metrics.get(key)
        if metric is None or np.isnan(metric.value):
            # Failures and undefined scores sort last, never first by accident.
            return (1, 0.0)
        return (0, -metric.value if higher_is_better else metric.value)

    return Comparison(task=task, ranked=sorted(evaluations, key=sort_key))
