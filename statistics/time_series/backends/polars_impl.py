"""Polars statistics backends — `time_series`."""
from __future__ import annotations
from typing import Any
import polars as pl
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F
from statistics.core.polars_frame import eager, numeric_series
from statistics.time_series.backends import pandas_impl
from statistics.time_series.backends.pandas_impl import *

from statistics.time_series.abstract import *

class CUSUMDetectorPolars(AbstractCUSUMDetector[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = CUSUMDetectorPandas()

    def detect(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.detect(data, column, **kwargs)

class VarianceShiftDetectorPolars(AbstractVarianceShiftDetector[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = VarianceShiftDetectorPandas()

    def detect(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.detect(data, column, **kwargs)

class ChangePointDetectorPolars(AbstractChangePointDetector[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = ChangePointDetectorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class TrendRemoverPolars(AbstractTrendRemover[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = TrendRemoverPandas()

    def remove(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.remove(data, column, **kwargs)

class HanningWindowApplierPolars(AbstractHanningWindowApplier[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = HanningWindowApplierPandas()

    def apply(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.apply(data, column, **kwargs)

class FFTPowerSpectrumCalculatorPolars(AbstractFFTPowerSpectrumCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = FFTPowerSpectrumCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class DominantCycleExtractorPolars(AbstractDominantCycleExtractor[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = DominantCycleExtractorPandas()

    def extract(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.extract(data, column, **kwargs)

class CyclicalPatternsCalculatorPolars(AbstractCyclicalPatternsCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = CyclicalPatternsCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class BaseAccuracyMetricPolars(AbstractBaseAccuracyMetric[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = BaseAccuracyMetricPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class MAEMetricPolars(AbstractMAEMetric[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = MAEMetricPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class RMSEMetricPolars(AbstractRMSEMetric[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = RMSEMetricPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class MAPEMetricPolars(AbstractMAPEMetric[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = MAPEMetricPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class MASEMetricPolars(AbstractMASEMetric[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = MASEMetricPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class ForecastAccuracyCalculatorPolars(AbstractForecastAccuracyCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = ForecastAccuracyCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class AutocovarianceCalculatorPolars(AbstractAutocovarianceCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = AutocovarianceCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class ACFCalculatorPolars(AbstractACFCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = ACFCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class PACFCalculatorPolars(AbstractPACFCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = PACFCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class LagFeatureBuilderPolars(AbstractLagFeatureBuilder[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = LagFeatureBuilderPandas()

    def build(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.build(data, column, **kwargs)

class LagFeaturesCalculatorPolars(AbstractLagFeaturesCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = LagFeaturesCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class RateOfChangeCalculatorPolars(AbstractRateOfChangeCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = RateOfChangeCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class AccelerationCalculatorPolars(AbstractAccelerationCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = AccelerationCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class MomentumSignalClassifierPolars(AbstractMomentumSignalClassifier[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = MomentumSignalClassifierPandas()

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

class MomentumCalculatorPolars(AbstractMomentumCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = MomentumCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class BaseMovingAveragePolars(AbstractBaseMovingAverage[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = BaseMovingAveragePandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class SimpleMovingAveragePolars(AbstractSimpleMovingAverage[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = SimpleMovingAveragePandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class ExponentialMovingAveragePolars(AbstractExponentialMovingAverage[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = ExponentialMovingAveragePandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class WeightedMovingAveragePolars(AbstractWeightedMovingAverage[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = WeightedMovingAveragePandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class CrossoverDetectorPolars(AbstractCrossoverDetector[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = CrossoverDetectorPandas()

    def detect(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.detect(data, column, **kwargs)

class MovingAveragesCalculatorPolars(AbstractMovingAveragesCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = MovingAveragesCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class BaseRollingStatisticPolars(AbstractBaseRollingStatistic[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = BaseRollingStatisticPandas()

    def compute(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.compute(data, column, **kwargs)

    def name(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.name(data, column, **kwargs)

class RollingMeanPolars(AbstractRollingMean[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = RollingMeanPandas()

    def name(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.name(data, column, **kwargs)

    def compute(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.compute(data, column, **kwargs)

class RollingStdPolars(AbstractRollingStd[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = RollingStdPandas()

    def name(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.name(data, column, **kwargs)

    def compute(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.compute(data, column, **kwargs)

class RollingMinPolars(AbstractRollingMin[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = RollingMinPandas()

    def name(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.name(data, column, **kwargs)

    def compute(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.compute(data, column, **kwargs)

class RollingMaxPolars(AbstractRollingMax[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = RollingMaxPandas()

    def name(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.name(data, column, **kwargs)

    def compute(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.compute(data, column, **kwargs)

class RollingMedianPolars(AbstractRollingMedian[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = RollingMedianPandas()

    def name(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.name(data, column, **kwargs)

    def compute(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.compute(data, column, **kwargs)

class RollingSkewnessPolars(AbstractRollingSkewness[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = RollingSkewnessPandas()

    def name(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.name(data, column, **kwargs)

    def compute(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.compute(data, column, **kwargs)

class RollingStatisticsCalculatorPolars(AbstractRollingStatisticsCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = RollingStatisticsCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class CenteredMovingAveragePolars(AbstractCenteredMovingAverage[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = CenteredMovingAveragePandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class EstacionalComponentPolars(AbstractEstacionalComponent[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = EstacionalComponentPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class SeasonalDecompositionPolars(AbstractSeasonalDecomposition[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = SeasonalDecompositionPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class AugmentedDickeyFullerTestPolars(AbstractAugmentedDickeyFullerTest[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = AugmentedDickeyFullerTestPandas()

    def test(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.test(data, column, **kwargs)

class KPSSTestPolars(AbstractKPSSTest[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = KPSSTestPandas()

    def test(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.test(data, column, **kwargs)

class StationarityVerdictInterpreterPolars(AbstractStationarityVerdictInterpreter[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = StationarityVerdictInterpreterPandas()

    def interpret(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.interpret(data, column, **kwargs)

class StationarityCalculatorPolars(AbstractStationarityCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = StationarityCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class RollingStdCalculatorPolars(AbstractRollingStdCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = RollingStdCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class EWMAVolatilityCalculatorPolars(AbstractEWMAVolatilityCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = EWMAVolatilityCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class CoefficientOfVariationCalculatorPolars(AbstractCoefficientOfVariationCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = CoefficientOfVariationCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class VolatilityRegimeDetectorPolars(AbstractVolatilityRegimeDetector[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = VolatilityRegimeDetectorPandas()

    def detect(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.detect(data, column, **kwargs)

class VolatilityCalculatorPolars(AbstractVolatilityCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = VolatilityCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)
