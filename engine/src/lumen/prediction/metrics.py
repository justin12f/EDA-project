"""Scoring.

Every metric returns a plain float and states its direction, because "0.83 is
good" depends entirely on which number it is — and a comparison table that sorts
the wrong way is worse than no table.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lumen.prediction.base import Task


@dataclass(frozen=True)
class Metric:
    name: str
    value: float
    # True when a larger value is better. The comparison layer sorts on this
    # rather than on a hardcoded list of metric names.
    higher_is_better: bool


def _clean_pair(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: y_true {y_true.shape}, y_pred {y_pred.shape}")
    keep = ~(np.isnan(y_true) | np.isnan(y_pred))
    return y_true[keep], y_pred[keep]


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    t, p = _clean_pair(y_true, y_pred)
    return float(np.mean(np.abs(t - p))) if t.size else float("nan")


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    t, p = _clean_pair(y_true, y_pred)
    return float(np.sqrt(np.mean((t - p) ** 2))) if t.size else float("nan")


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute percentage error, as a fraction.

    Undefined where the truth is zero, so those rows are excluded rather than
    contributing an infinity that silently poisons the mean. Returns NaN when
    every row is excluded — which is the honest answer for a series of zeros.
    """
    t, p = _clean_pair(y_true, y_pred)
    nonzero = t != 0
    if not nonzero.any():
        return float("nan")
    return float(np.mean(np.abs((t[nonzero] - p[nonzero]) / t[nonzero])))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination.

    Zero variance in the truth makes R² undefined, not 1.0 — a constant series
    predicted perfectly explains no variance because there is none to explain.
    """
    t, p = _clean_pair(y_true, y_pred)
    if t.size < 2:
        return float("nan")
    total = float(np.sum((t - np.mean(t)) ** 2))
    if total == 0.0:
        return float("nan")
    residual = float(np.sum((t - p) ** 2))
    return 1.0 - residual / total


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    t, p = _clean_pair(y_true, y_pred)
    return float(np.mean(t == p)) if t.size else float("nan")


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Unweighted mean F1 across classes.

    Macro rather than micro on purpose: with imbalanced classes, micro-F1
    flatters a model that only ever predicts the majority, which is exactly the
    failure a person needs to see.
    """
    t, p = _clean_pair(y_true, y_pred)
    if not t.size:
        return float("nan")

    scores: list[float] = []
    for label in np.unique(t):
        tp = float(np.sum((p == label) & (t == label)))
        fp = float(np.sum((p == label) & (t != label)))
        fn = float(np.sum((p != label) & (t == label)))
        if tp == 0:
            scores.append(0.0)
            continue
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        scores.append(2 * precision * recall / (precision + recall))
    return float(np.mean(scores))


REGRESSION_METRICS = {
    "mae": (mae, False),
    "rmse": (rmse, False),
    "mape": (mape, False),
    "r2": (r2, True),
}

CLASSIFICATION_METRICS = {
    "accuracy": (accuracy, True),
    "macro_f1": (macro_f1, True),
}


def score(y_true: np.ndarray, y_pred: np.ndarray, task: Task) -> dict[str, Metric]:
    """Every metric appropriate to the task, keyed by name."""
    table = (
        CLASSIFICATION_METRICS if task is Task.CLASSIFICATION else REGRESSION_METRICS
    )
    return {
        name: Metric(name=name, value=fn(y_true, y_pred), higher_is_better=higher)
        for name, (fn, higher) in table.items()
    }


def primary_metric(task: Task) -> str:
    """The one to rank on when nobody said which.

    RMSE for regression because it punishes the large miss that a person
    actually notices; macro-F1 for classification because accuracy lies on
    imbalanced data.
    """
    return "macro_f1" if task is Task.CLASSIFICATION else "rmse"
