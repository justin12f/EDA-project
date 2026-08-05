"""Prediction: numerical methods, time series, and machine learning.

    from lumen.agents.master_factory import AgentMasterFactory

    prediction = AgentMasterFactory("polars").prediction()

    prediction.available(task=Task.FORECAST)
    prediction.compare(frame, target="amount")
    prediction.forecast(frame, "exponential_smoothing", "signups", horizon=14)

Three families behind one registry. Adding a method is a `register()` call —
nothing else in the system needs to know it exists, because the agent tool reads
its vocabulary from the registry rather than from a list somebody has to
remember to update.
"""

from lumen.prediction.base import (
    Family,
    FitReport,
    Prediction,
    Predictor,
    Task,
)
from lumen.prediction.evaluate import (
    Comparison,
    Evaluation,
    backtest,
    compare,
    evaluate,
    split,
)
from lumen.prediction.extract import to_series, to_supervised
from lumen.prediction.inyeccion import PredictionInyeccionDependency
from lumen.prediction.metrics import Metric, primary_metric, score
from lumen.prediction.registry import PredictorRegistry

__all__ = [
    "Comparison",
    "Evaluation",
    "Family",
    "FitReport",
    "Metric",
    "Prediction",
    "PredictionInyeccionDependency",
    "Predictor",
    "PredictorRegistry",
    "Task",
    "backtest",
    "compare",
    "evaluate",
    "primary_metric",
    "score",
    "split",
    "to_series",
    "to_supervised",
]
