"""Prediction dependency injection.

Follows the naming of the other domain layers (`ReadersInyeccionDependency`,
`StatisticsInyeccionDependency`) so `AgentMasterFactory` stays uniform. It holds
a backend because *extraction* needs one — the predictors themselves do not.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from lumen.core.backend import DEFAULT_BACKEND
from lumen.core.inyeccion import BackendInyeccionDependency
from lumen.prediction.base import Family, FitReport, Prediction, Predictor, Task
from lumen.prediction.evaluate import Comparison, Evaluation, backtest, compare, evaluate
from lumen.prediction.extract import to_series, to_supervised
from lumen.prediction.registry import PredictorRegistry


class PredictionInyeccionDependency(BackendInyeccionDependency):
    def __init__(self, backend: str = DEFAULT_BACKEND) -> None:
        super().__init__(backend)

    # ── discovery ───────────────────────────────────────────────────────────

    def available(self, *, task: Task | None = None, family: Family | None = None) -> list[str]:
        return PredictorRegistry.names(task=task, family=family)

    def catalogue(self) -> list[dict[str, Any]]:
        return PredictorRegistry.describe()

    def create(self, name: str, **params: Any) -> Predictor:
        return PredictorRegistry.create(name, **params)

    # ── supervised ──────────────────────────────────────────────────────────

    def fit(
        self,
        frame: Any,
        name: str,
        target: str,
        features: list[str] | None = None,
        *,
        order_by: str | None = None,
        **params: Any,
    ) -> tuple[Predictor, FitReport, list[str]]:
        X, y, used = to_supervised(
            frame, self._backend, target, features, order_by=order_by
        )
        predictor = self.create(name, **params)
        return predictor, predictor.fit(X, y), used

    def evaluate(
        self,
        frame: Any,
        name: str,
        target: str,
        features: list[str] | None = None,
        *,
        order_by: str | None = None,
        test_fraction: float = 0.2,
        **params: Any,
    ) -> Evaluation:
        X, y, _ = to_supervised(frame, self._backend, target, features, order_by=order_by)
        return evaluate(self.create(name, **params), X, y, test_fraction=test_fraction)

    def compare(
        self,
        frame: Any,
        target: str,
        features: list[str] | None = None,
        *,
        task: Task = Task.REGRESSION,
        candidates: list[str] | None = None,
        order_by: str | None = None,
        test_fraction: float = 0.2,
    ) -> Comparison:
        X, y, _ = to_supervised(frame, self._backend, target, features, order_by=order_by)
        return compare(
            X, y, task=task, candidates=candidates, test_fraction=test_fraction
        )

    # ── forecasting ─────────────────────────────────────────────────────────

    def forecast(
        self,
        frame: Any,
        name: str,
        column: str,
        horizon: int,
        *,
        order_by: str | None = None,
        **params: Any,
    ) -> tuple[Prediction, FitReport]:
        series = to_series(frame, self._backend, column, order_by=order_by)
        series = series[~np.isnan(series)]

        predictor = self.create(name, **params)
        if not predictor.univariate:
            raise ValueError(
                f"'{name}' is not a forecaster. Univariate methods: "
                f"{', '.join(self.available(task=Task.FORECAST))}"
            )
        report = predictor.fit(np.arange(series.size, dtype=float).reshape(-1, 1), series)
        return predictor.forecast(horizon), report

    def backtest(
        self,
        frame: Any,
        name: str,
        column: str,
        *,
        folds: int = 3,
        horizon: int = 1,
        order_by: str | None = None,
        **params: Any,
    ) -> Evaluation:
        series = to_series(frame, self._backend, column, order_by=order_by)
        return backtest(self.create(name, **params), series, folds=folds, horizon=horizon)

    def compare_forecasters(
        self,
        frame: Any,
        column: str,
        *,
        folds: int = 3,
        horizon: int = 1,
        candidates: list[str] | None = None,
        order_by: str | None = None,
    ) -> Comparison:
        """Rolling-origin backtest for every forecaster, ranked.

        Includes `naive` unless the caller excluded it deliberately: a forecast
        that cannot beat "tomorrow equals today" has not earned its complexity,
        and without the baseline in the table nobody checks.
        """
        series = to_series(frame, self._backend, column, order_by=order_by)
        names = candidates or self.available(task=Task.FORECAST)

        results: list[Evaluation] = []
        for name in names:
            try:
                results.append(
                    backtest(self.create(name), series, folds=folds, horizon=horizon)
                )
            except Exception as exc:  # noqa: BLE001 — one bad method is a result
                results.append(
                    Evaluation(
                        predictor=name,
                        task=Task.FORECAST,
                        metrics={},
                        n_train=0,
                        n_test=0,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )

        def sort_key(evaluation: Evaluation) -> tuple[int, float]:
            metric = evaluation.metrics.get("rmse")
            if metric is None or np.isnan(metric.value):
                return (1, 0.0)
            return (0, metric.value)

        return Comparison(task=Task.FORECAST, ranked=sorted(results, key=sort_key))
