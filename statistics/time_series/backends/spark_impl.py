"""Spark statistics backends — `time_series`."""
from __future__ import annotations
from typing import Any
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F
from statistics.time_series.abstract import *

from statistics.time_series.backends import pandas_impl
from statistics.time_series.backends.pandas_impl import *

class CUSUMDetectorSpark(AbstractCUSUMDetector[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = CUSUMDetectorPandas()

    def detect(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.detect(data, column, **kwargs)

class VarianceShiftDetectorSpark(AbstractVarianceShiftDetector[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = VarianceShiftDetectorPandas()

    def detect(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.detect(data, column, **kwargs)

class ChangePointDetectorSpark(AbstractChangePointDetector[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = ChangePointDetectorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class TrendRemoverSpark(AbstractTrendRemover[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = TrendRemoverPandas()

    def remove(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.remove(data, column, **kwargs)

class HanningWindowApplierSpark(AbstractHanningWindowApplier[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = HanningWindowApplierPandas()

    def apply(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.apply(data, column, **kwargs)

class FFTPowerSpectrumCalculatorSpark(AbstractFFTPowerSpectrumCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = FFTPowerSpectrumCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class DominantCycleExtractorSpark(AbstractDominantCycleExtractor[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = DominantCycleExtractorPandas()

    def extract(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.extract(data, column, **kwargs)

class CyclicalPatternsCalculatorSpark(AbstractCyclicalPatternsCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = CyclicalPatternsCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class BaseAccuracyMetricSpark(AbstractBaseAccuracyMetric[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = BaseAccuracyMetricPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class MAEMetricSpark(AbstractMAEMetric[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = MAEMetricPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class RMSEMetricSpark(AbstractRMSEMetric[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = RMSEMetricPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class MAPEMetricSpark(AbstractMAPEMetric[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = MAPEMetricPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class MASEMetricSpark(AbstractMASEMetric[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = MASEMetricPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class ForecastAccuracyCalculatorSpark(AbstractForecastAccuracyCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = ForecastAccuracyCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class AutocovarianceCalculatorSpark(AbstractAutocovarianceCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = AutocovarianceCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class ACFCalculatorSpark(AbstractACFCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = ACFCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class PACFCalculatorSpark(AbstractPACFCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = PACFCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class LagFeatureBuilderSpark(AbstractLagFeatureBuilder[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = LagFeatureBuilderPandas()

    def build(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.build(data, column, **kwargs)

class LagFeaturesCalculatorSpark(AbstractLagFeaturesCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = LagFeaturesCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class RateOfChangeCalculatorSpark(AbstractRateOfChangeCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = RateOfChangeCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class AccelerationCalculatorSpark(AbstractAccelerationCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = AccelerationCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class MomentumSignalClassifierSpark(AbstractMomentumSignalClassifier[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = MomentumSignalClassifierPandas()

    def classify(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.classify(data, column, **kwargs)

class MomentumCalculatorSpark(AbstractMomentumCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = MomentumCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class BaseMovingAverageSpark(AbstractBaseMovingAverage[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = BaseMovingAveragePandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class SimpleMovingAverageSpark(AbstractSimpleMovingAverage[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = SimpleMovingAveragePandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class ExponentialMovingAverageSpark(AbstractExponentialMovingAverage[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = ExponentialMovingAveragePandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class WeightedMovingAverageSpark(AbstractWeightedMovingAverage[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = WeightedMovingAveragePandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class CrossoverDetectorSpark(AbstractCrossoverDetector[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = CrossoverDetectorPandas()

    def detect(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.detect(data, column, **kwargs)

class MovingAveragesCalculatorSpark(AbstractMovingAveragesCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = MovingAveragesCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class BaseRollingStatisticSpark(AbstractBaseRollingStatistic[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = BaseRollingStatisticPandas()

    def compute(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.compute(data, column, **kwargs)

    def name(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.name(data, column, **kwargs)

class RollingMeanSpark(AbstractRollingMean[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = RollingMeanPandas()

    def name(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.name(data, column, **kwargs)

    def compute(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.compute(data, column, **kwargs)

class RollingStdSpark(AbstractRollingStd[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = RollingStdPandas()

    def name(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.name(data, column, **kwargs)

    def compute(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.compute(data, column, **kwargs)

class RollingMinSpark(AbstractRollingMin[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = RollingMinPandas()

    def name(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.name(data, column, **kwargs)

    def compute(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.compute(data, column, **kwargs)

class RollingMaxSpark(AbstractRollingMax[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = RollingMaxPandas()

    def name(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.name(data, column, **kwargs)

    def compute(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.compute(data, column, **kwargs)

class RollingMedianSpark(AbstractRollingMedian[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = RollingMedianPandas()

    def name(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.name(data, column, **kwargs)

    def compute(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.compute(data, column, **kwargs)

class RollingSkewnessSpark(AbstractRollingSkewness[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = RollingSkewnessPandas()

    def name(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.name(data, column, **kwargs)

    def compute(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.compute(data, column, **kwargs)

class RollingStatisticsCalculatorSpark(AbstractRollingStatisticsCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = RollingStatisticsCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class CenteredMovingAverageSpark(AbstractCenteredMovingAverage[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = CenteredMovingAveragePandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class EstacionalComponentSpark(AbstractEstacionalComponent[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = EstacionalComponentPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class SeasonalDecompositionSpark(AbstractSeasonalDecomposition[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = SeasonalDecompositionPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class AugmentedDickeyFullerTestSpark(AbstractAugmentedDickeyFullerTest[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = AugmentedDickeyFullerTestPandas()

    def test(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.test(data, column, **kwargs)

class KPSSTestSpark(AbstractKPSSTest[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = KPSSTestPandas()

    def test(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.test(data, column, **kwargs)

class StationarityVerdictInterpreterSpark(AbstractStationarityVerdictInterpreter[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = StationarityVerdictInterpreterPandas()

    def interpret(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.interpret(data, column, **kwargs)

class StationarityCalculatorSpark(AbstractStationarityCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = StationarityCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class RollingStdCalculatorSpark(AbstractRollingStdCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = RollingStdCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class EWMAVolatilityCalculatorSpark(AbstractEWMAVolatilityCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = EWMAVolatilityCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class CoefficientOfVariationCalculatorSpark(AbstractCoefficientOfVariationCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = CoefficientOfVariationCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class VolatilityRegimeDetectorSpark(AbstractVolatilityRegimeDetector[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = VolatilityRegimeDetectorPandas()

    def detect(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.detect(data, column, **kwargs)

class VolatilityCalculatorSpark(AbstractVolatilityCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = VolatilityCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)
