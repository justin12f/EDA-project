"""Spark statistics backends — `survival`."""
from __future__ import annotations
from typing import Any
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F
from statistics.survival.abstract import *

from statistics.survival.backends import pandas_impl
from statistics.survival.backends.pandas_impl import *

class IntervalExtractorSpark(AbstractIntervalExtractor[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = IntervalExtractorPandas()

    def extract(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.extract(data, column, **kwargs)

class RegularityClassifierSpark(AbstractRegularityClassifier[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = RegularityClassifierPandas()

    def classify(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.classify(data, column, **kwargs)

class TemporalBurstinessCalculatorSpark(AbstractTemporalBurstinessCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = TemporalBurstinessCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class RollingEventRateCalculatorSpark(AbstractRollingEventRateCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = RollingEventRateCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class EventDensityCalculatorSpark(AbstractEventDensityCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = EventDensityCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class NelsonAalenEstimatorSpark(AbstractNelsonAalenEstimator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = NelsonAalenEstimatorPandas()

    def estimate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.estimate(data, column, **kwargs)

class KernelHazardSmootherSpark(AbstractKernelHazardSmoother[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = KernelHazardSmootherPandas()

    def smooth(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.smooth(data, column, **kwargs)

class HazardRateCalculatorSpark(AbstractHazardRateCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = HazardRateCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class KaplanMeierEstimatorSpark(AbstractKaplanMeierEstimator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = KaplanMeierEstimatorPandas()

    def estimate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.estimate(data, column, **kwargs)

class MedianSurvivalEstimatorSpark(AbstractMedianSurvivalEstimator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = MedianSurvivalEstimatorPandas()

    def estimate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.estimate(data, column, **kwargs)

class KaplanMeierCalculatorSpark(AbstractKaplanMeierCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = KaplanMeierCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class TimeToEventSummaryCalculatorSpark(AbstractTimeToEventSummaryCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = TimeToEventSummaryCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class ThresholdSurvivalAnalyserSpark(AbstractThresholdSurvivalAnalyser[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = ThresholdSurvivalAnalyserPandas()

    def analyse(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.analyse(data, column, **kwargs)

class ExponentialFitterSpark(AbstractExponentialFitter[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = ExponentialFitterPandas()

    def fit(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.fit(data, column, **kwargs)

class TimeToEventCalculatorSpark(AbstractTimeToEventCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = TimeToEventCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)
