"""Polars statistics backends — `survival`."""
from __future__ import annotations
from typing import Any
import polars as pl
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F
from statistics.core.polars_frame import eager, numeric_series
from statistics.survival.backends import pandas_impl
from statistics.survival.backends.pandas_impl import *

from statistics.survival.abstract import *

class IntervalExtractorPolars(AbstractIntervalExtractor[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = IntervalExtractorPandas()

    def extract(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.extract(data, column, **kwargs)

class RegularityClassifierPolars(AbstractRegularityClassifier[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = RegularityClassifierPandas()

    def classify(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        frame = eager(data)
        s = numeric_series(frame, column)
        if s.len() < 8:
            raise ValueError('Need at least 8 samples for classify')
        skew = float(s.skew())
        kurt = float(s.kurtosis())
        label = 'symmetric' if abs(skew) < 0.5 else 'skewed'
        return {
            "classification_label": label,
            "skewness": skew,
            "kurtosis": kurt,
            "is_bimodal": False,
            "recommended_transformation": "log1p" if skew > 1 else "none",
        }

class TemporalBurstinessCalculatorPolars(AbstractTemporalBurstinessCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = TemporalBurstinessCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class RollingEventRateCalculatorPolars(AbstractRollingEventRateCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = RollingEventRateCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class EventDensityCalculatorPolars(AbstractEventDensityCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = EventDensityCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class NelsonAalenEstimatorPolars(AbstractNelsonAalenEstimator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = NelsonAalenEstimatorPandas()

    def estimate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.estimate(data, column, **kwargs)

class KernelHazardSmootherPolars(AbstractKernelHazardSmoother[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = KernelHazardSmootherPandas()

    def smooth(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.smooth(data, column, **kwargs)

class HazardRateCalculatorPolars(AbstractHazardRateCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = HazardRateCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class KaplanMeierEstimatorPolars(AbstractKaplanMeierEstimator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = KaplanMeierEstimatorPandas()

    def estimate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.estimate(data, column, **kwargs)

class MedianSurvivalEstimatorPolars(AbstractMedianSurvivalEstimator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = MedianSurvivalEstimatorPandas()

    def estimate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.estimate(data, column, **kwargs)

class KaplanMeierCalculatorPolars(AbstractKaplanMeierCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = KaplanMeierCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class TimeToEventSummaryCalculatorPolars(AbstractTimeToEventSummaryCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = TimeToEventSummaryCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class ThresholdSurvivalAnalyserPolars(AbstractThresholdSurvivalAnalyser[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = ThresholdSurvivalAnalyserPandas()

    def analyse(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.analyse(data, column, **kwargs)

class ExponentialFitterPolars(AbstractExponentialFitter[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = ExponentialFitterPandas()

    def fit(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.fit(data, column, **kwargs)

class TimeToEventCalculatorPolars(AbstractTimeToEventCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = TimeToEventCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)
