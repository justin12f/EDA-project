"""The predictor registry.

Keyed by name alone, not by `(name, backend)` like the rest of the engine's
factories. That is a considered departure, not an oversight: readers and cleaning
steps differ per backend because null handling and group-by genuinely differ,
whereas ridge regression does not change when the frame came from polars. The
backend axis lives in `extract.py`, which is where the difference is.

Adding a method is a `register()` call. Nothing else in the system needs to know
it exists — the agent tool reads its vocabulary from here, the same way the
cleaning tool reads its step names from the cleaning factory.
"""

from __future__ import annotations

from typing import Any

from lumen.prediction import ml, numerical, timeseries
from lumen.prediction.base import Family, Predictor, Task


class PredictorRegistry:
    """name → predictor class."""

    _registry: dict[str, type[Predictor]] = {}

    @classmethod
    def register(cls, predictor: type[Predictor]) -> type[Predictor]:
        cls._registry[predictor.name] = predictor
        return predictor

    @classmethod
    def get_class(cls, name: str) -> type[Predictor]:
        predictor = cls._registry.get(name)
        if predictor is None:
            raise ValueError(
                f"Unknown predictor '{name}'. Available: {', '.join(sorted(cls._registry))}"
            )
        return predictor

    @classmethod
    def create(cls, name: str, **params: Any) -> Predictor:
        return cls.get_class(name)(**params)

    @classmethod
    def names(
        cls, *, task: Task | None = None, family: Family | None = None
    ) -> list[str]:
        return sorted(
            name
            for name, predictor in cls._registry.items()
            if (task is None or predictor.task is task)
            and (family is None or predictor.family is family)
        )

    @classmethod
    def describe(cls) -> list[dict[str, Any]]:
        """Every registered method with what it is for.

        Consumed by the agent tool description, so a model never has to guess a
        predictor name — the same fix applied to cleaning steps after an agent
        confidently proposed one that had never existed.
        """
        import inspect

        rows: list[dict[str, Any]] = []
        for name in sorted(cls._registry):
            predictor = cls._registry[name]
            signature = inspect.signature(predictor.__init__)
            rows.append(
                {
                    "name": name,
                    "family": str(predictor.family),
                    "task": str(predictor.task),
                    "univariate": predictor.univariate,
                    "params": {
                        parameter.name: (
                            parameter.default
                            if parameter.default is not inspect.Parameter.empty
                            else None
                        )
                        for parameter in signature.parameters.values()
                        if parameter.name not in ("self", "params", "kwargs")
                    },
                    "summary": (predictor.__doc__ or "").strip().split("\n")[0],
                }
            )
        return rows


for _predictor in (
    # numerical
    numerical.LeastSquares,
    numerical.PolynomialFit,
    numerical.LinearInterpolation,
    numerical.CubicSpline,
    numerical.MovingAverage,
    # time series
    timeseries.NaiveForecast,
    timeseries.SeasonalNaive,
    timeseries.DriftForecast,
    timeseries.ExponentialSmoothing,
    # machine learning — regression
    ml.RidgeRegression,
    ml.LassoRegression,
    ml.ElasticNetRegression,
    ml.RandomForestRegression,
    ml.GradientBoostingRegression,
    ml.SupportVectorRegression,
    ml.KNearestRegression,
    # machine learning — classification
    ml.LogisticClassification,
    ml.RandomForestClassification,
    ml.GradientBoostingClassification,
):
    PredictorRegistry.register(_predictor)
