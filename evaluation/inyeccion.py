"""Evaluation metrics dependency injection."""

from __future__ import annotations

from typing import Any

from core.backend import DEFAULT_BACKEND
from core.inyeccion import BackendInyeccionDependency
from evaluation.score_factory import EvaluationScoreFactory


class EvaluationInyeccionDependency(BackendInyeccionDependency):
    def __init__(self, backend: str = DEFAULT_BACKEND) -> None:
        super().__init__(backend)

    def score(self, metric: str, y_true: Any, y_pred: Any) -> float:
        scorer = EvaluationScoreFactory.create(metric, self._backend)
        return float(scorer.score(y_true, y_pred))
