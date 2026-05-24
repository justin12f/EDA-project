"""Evaluation metrics factory by backend."""

from __future__ import annotations

from typing import Any

import numpy as np

from core.abstract_factory import RegistryFactory
from evaluation.score import MeanSquareError, SquaredR


class _PandasMSE:
    def score(self, y_true: Any, y_pred: Any) -> float:
        return MeanSquareError().mean_square_error(
            np.asarray(y_true), np.asarray(y_pred)
        )


class _PandasMAE:
    def score(self, y_true: Any, y_pred: Any) -> float:
        yt = np.ravel(np.asarray(y_true))
        yp = np.ravel(np.asarray(y_pred))
        return float(np.mean(np.abs(yt - yp)))


class _PandasR2:
    def score(self, y_true: Any, y_pred: Any) -> float:
        return SquaredR().squared_r(np.asarray(y_true), np.asarray(y_pred))


class EvaluationScoreFactory(RegistryFactory[str, Any]):
    """Maps (metric_name, backend) → scorer with .score(y_true, y_pred)."""


def _register() -> None:
    for backend in ("pandas", "polars", "spark"):
        EvaluationScoreFactory.register("mse", backend, _PandasMSE)
        EvaluationScoreFactory.register("mae", backend, _PandasMAE)
        EvaluationScoreFactory.register("r2", backend, _PandasR2)


_register()
