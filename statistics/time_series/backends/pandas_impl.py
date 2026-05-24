"""Pandas statistics backends — `time_series`."""
from __future__ import annotations
from typing import Any
import pandas as pd
from statistics.core.frame_extract import column_to_numpy
from statistics.time_series.abstract import *

import statistics.time_series.change_points as _mod_change_points
import statistics.time_series.cyclical_patterns as _mod_cyclical_patterns
import statistics.time_series.forecast_accuracy as _mod_forecast_accuracy
import statistics.time_series.lag_features as _mod_lag_features
import statistics.time_series.momentum as _mod_momentum
import statistics.time_series.moving_averages as _mod_moving_averages
import statistics.time_series.rolling_statistics as _mod_rolling_statistics
import statistics.time_series.seasonal as _mod_seasonal
import statistics.time_series.stationarity as _mod_stationarity
import statistics.time_series.volatility as _mod_volatility

class CUSUMDetectorPandas(AbstractCUSUMDetector[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_change_points.CUSUMDetector()

    def detect(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.detect(arr, **kwargs)

class VarianceShiftDetectorPandas(AbstractVarianceShiftDetector[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_change_points.VarianceShiftDetector()

    def detect(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.detect(arr, **kwargs)

class ChangePointDetectorPandas(AbstractChangePointDetector[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_change_points.ChangePointDetector()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class TrendRemoverPandas(AbstractTrendRemover[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_cyclical_patterns.TrendRemover()

    def remove(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.remove(arr, **kwargs)

class HanningWindowApplierPandas(AbstractHanningWindowApplier[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_cyclical_patterns.HanningWindowApplier()

    def apply(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.apply(arr, **kwargs)

class FFTPowerSpectrumCalculatorPandas(AbstractFFTPowerSpectrumCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_cyclical_patterns.FFTPowerSpectrumCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class DominantCycleExtractorPandas(AbstractDominantCycleExtractor[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_cyclical_patterns.DominantCycleExtractor()

    def extract(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.extract(arr, **kwargs)

class CyclicalPatternsCalculatorPandas(AbstractCyclicalPatternsCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_cyclical_patterns.CyclicalPatternsCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class BaseAccuracyMetricPandas(AbstractBaseAccuracyMetric[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_forecast_accuracy.BaseAccuracyMetric()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class MAEMetricPandas(AbstractMAEMetric[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_forecast_accuracy.MAEMetric()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class RMSEMetricPandas(AbstractRMSEMetric[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_forecast_accuracy.RMSEMetric()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class MAPEMetricPandas(AbstractMAPEMetric[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_forecast_accuracy.MAPEMetric()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class MASEMetricPandas(AbstractMASEMetric[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_forecast_accuracy.MASEMetric()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class ForecastAccuracyCalculatorPandas(AbstractForecastAccuracyCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_forecast_accuracy.ForecastAccuracyCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class AutocovarianceCalculatorPandas(AbstractAutocovarianceCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_lag_features.AutocovarianceCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class ACFCalculatorPandas(AbstractACFCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_lag_features.ACFCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class PACFCalculatorPandas(AbstractPACFCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_lag_features.PACFCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class LagFeatureBuilderPandas(AbstractLagFeatureBuilder[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_lag_features.LagFeatureBuilder()

    def build(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.build(arr, **kwargs)

class LagFeaturesCalculatorPandas(AbstractLagFeaturesCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_lag_features.LagFeaturesCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class RateOfChangeCalculatorPandas(AbstractRateOfChangeCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_momentum.RateOfChangeCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class AccelerationCalculatorPandas(AbstractAccelerationCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_momentum.AccelerationCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class MomentumSignalClassifierPandas(AbstractMomentumSignalClassifier[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_momentum.MomentumSignalClassifier()

    def classify(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.classify(arr, **kwargs)

class MomentumCalculatorPandas(AbstractMomentumCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_momentum.MomentumCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class BaseMovingAveragePandas(AbstractBaseMovingAverage[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_moving_averages.BaseMovingAverage()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class SimpleMovingAveragePandas(AbstractSimpleMovingAverage[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_moving_averages.SimpleMovingAverage()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class ExponentialMovingAveragePandas(AbstractExponentialMovingAverage[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_moving_averages.ExponentialMovingAverage()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class WeightedMovingAveragePandas(AbstractWeightedMovingAverage[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_moving_averages.WeightedMovingAverage()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class CrossoverDetectorPandas(AbstractCrossoverDetector[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_moving_averages.CrossoverDetector()

    def detect(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.detect(arr, **kwargs)

class MovingAveragesCalculatorPandas(AbstractMovingAveragesCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_moving_averages.MovingAveragesCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class BaseRollingStatisticPandas(AbstractBaseRollingStatistic[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_rolling_statistics.BaseRollingStatistic()

    def compute(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.compute(arr, **kwargs)

    def name(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.name(arr, **kwargs)

class RollingMeanPandas(AbstractRollingMean[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_rolling_statistics.RollingMean()

    def name(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.name(arr, **kwargs)

    def compute(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.compute(arr, **kwargs)

class RollingStdPandas(AbstractRollingStd[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_rolling_statistics.RollingStd()

    def name(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.name(arr, **kwargs)

    def compute(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.compute(arr, **kwargs)

class RollingMinPandas(AbstractRollingMin[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_rolling_statistics.RollingMin()

    def name(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.name(arr, **kwargs)

    def compute(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.compute(arr, **kwargs)

class RollingMaxPandas(AbstractRollingMax[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_rolling_statistics.RollingMax()

    def name(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.name(arr, **kwargs)

    def compute(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.compute(arr, **kwargs)

class RollingMedianPandas(AbstractRollingMedian[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_rolling_statistics.RollingMedian()

    def name(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.name(arr, **kwargs)

    def compute(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.compute(arr, **kwargs)

class RollingSkewnessPandas(AbstractRollingSkewness[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_rolling_statistics.RollingSkewness()

    def name(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.name(arr, **kwargs)

    def compute(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.compute(arr, **kwargs)

class RollingStatisticsCalculatorPandas(AbstractRollingStatisticsCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_rolling_statistics.RollingStatisticsCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class CenteredMovingAveragePandas(AbstractCenteredMovingAverage[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_seasonal.CenteredMovingAverage()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class EstacionalComponentPandas(AbstractEstacionalComponent[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_seasonal.EstacionalComponent()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class SeasonalDecompositionPandas(AbstractSeasonalDecomposition[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_seasonal.SeasonalDecomposition()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class AugmentedDickeyFullerTestPandas(AbstractAugmentedDickeyFullerTest[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_stationarity.AugmentedDickeyFullerTest()

    def test(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.test(arr, **kwargs)

class KPSSTestPandas(AbstractKPSSTest[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_stationarity.KPSSTest()

    def test(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.test(arr, **kwargs)

class StationarityVerdictInterpreterPandas(AbstractStationarityVerdictInterpreter[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_stationarity.StationarityVerdictInterpreter()

    def interpret(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.interpret(arr, **kwargs)

class StationarityCalculatorPandas(AbstractStationarityCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_stationarity.StationarityCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class RollingStdCalculatorPandas(AbstractRollingStdCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_volatility.RollingStdCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class EWMAVolatilityCalculatorPandas(AbstractEWMAVolatilityCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_volatility.EWMAVolatilityCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class CoefficientOfVariationCalculatorPandas(AbstractCoefficientOfVariationCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_volatility.CoefficientOfVariationCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class VolatilityRegimeDetectorPandas(AbstractVolatilityRegimeDetector[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_volatility.VolatilityRegimeDetector()

    def detect(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.detect(arr, **kwargs)

class VolatilityCalculatorPandas(AbstractVolatilityCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_volatility.VolatilityCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)
