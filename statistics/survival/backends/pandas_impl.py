"""Pandas statistics backends — `survival`."""
from __future__ import annotations
from typing import Any
import pandas as pd
from statistics.core.frame_extract import column_to_numpy
from statistics.survival.abstract import *

import statistics.survival.event_density as _mod_event_density
import statistics.survival.hazard_rate as _mod_hazard_rate
import statistics.survival.kaplan_meier as _mod_kaplan_meier
import statistics.survival.time_to_event as _mod_time_to_event

class IntervalExtractorPandas(AbstractIntervalExtractor[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_event_density.IntervalExtractor()

    def extract(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.extract(arr, **kwargs)

class RegularityClassifierPandas(AbstractRegularityClassifier[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_event_density.RegularityClassifier()

    def classify(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.classify(arr, **kwargs)

class TemporalBurstinessCalculatorPandas(AbstractTemporalBurstinessCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_event_density.TemporalBurstinessCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class RollingEventRateCalculatorPandas(AbstractRollingEventRateCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_event_density.RollingEventRateCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class EventDensityCalculatorPandas(AbstractEventDensityCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_event_density.EventDensityCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class NelsonAalenEstimatorPandas(AbstractNelsonAalenEstimator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_hazard_rate.NelsonAalenEstimator()

    def estimate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.estimate(arr, **kwargs)

class KernelHazardSmootherPandas(AbstractKernelHazardSmoother[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_hazard_rate.KernelHazardSmoother()

    def smooth(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.smooth(arr, **kwargs)

class HazardRateCalculatorPandas(AbstractHazardRateCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_hazard_rate.HazardRateCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class KaplanMeierEstimatorPandas(AbstractKaplanMeierEstimator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_kaplan_meier.KaplanMeierEstimator()

    def estimate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.estimate(arr, **kwargs)

class MedianSurvivalEstimatorPandas(AbstractMedianSurvivalEstimator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_kaplan_meier.MedianSurvivalEstimator()

    def estimate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.estimate(arr, **kwargs)

class KaplanMeierCalculatorPandas(AbstractKaplanMeierCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_kaplan_meier.KaplanMeierCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class TimeToEventSummaryCalculatorPandas(AbstractTimeToEventSummaryCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_time_to_event.TimeToEventSummaryCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class ThresholdSurvivalAnalyserPandas(AbstractThresholdSurvivalAnalyser[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_time_to_event.ThresholdSurvivalAnalyser()

    def analyse(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.analyse(arr, **kwargs)

class ExponentialFitterPandas(AbstractExponentialFitter[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_time_to_event.ExponentialFitter()

    def fit(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.fit(arr, **kwargs)

class TimeToEventCalculatorPandas(AbstractTimeToEventCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_time_to_event.TimeToEventCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)
