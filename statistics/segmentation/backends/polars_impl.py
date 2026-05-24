"""Polars statistics backends — `segmentation`."""
from __future__ import annotations
from typing import Any
import polars as pl
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F
from statistics.core.polars_frame import eager, numeric_series
from statistics.segmentation.backends import pandas_impl
from statistics.segmentation.backends.pandas_impl import *

from statistics.segmentation.abstract import *

class CohortAssignerPolars(AbstractCohortAssigner[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = CohortAssignerPandas()

    def assign(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.assign(data, column, **kwargs)

class CohortPeriodOffsetCalculatorPolars(AbstractCohortPeriodOffsetCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = CohortPeriodOffsetCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class RetentionMatrixBuilderPolars(AbstractRetentionMatrixBuilder[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = RetentionMatrixBuilderPandas()

    def build(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.build(data, column, **kwargs)

class RetentionRateNormalizerPolars(AbstractRetentionRateNormalizer[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = RetentionRateNormalizerPandas()

    def normalize(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.normalize(data, column, **kwargs)

class CohortAnalysisCalculatorPolars(AbstractCohortAnalysisCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = CohortAnalysisCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class EpsilonEstimatorPolars(AbstractEpsilonEstimator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = EpsilonEstimatorPandas()

    def estimate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.estimate(data, column, **kwargs)

class DBSCANClusterProfileBuilderPolars(AbstractDBSCANClusterProfileBuilder[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = DBSCANClusterProfileBuilderPandas()

    def build(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.build(data, column, **kwargs)

class DBSCANClusterCalculatorPolars(AbstractDBSCANClusterCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = DBSCANClusterCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class LinkageMatrixBuilderPolars(AbstractLinkageMatrixBuilder[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = LinkageMatrixBuilderPandas()

    def build(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.build(data, column, **kwargs)

class CopheneticCorrelationCalculatorPolars(AbstractCopheneticCorrelationCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = CopheneticCorrelationCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class OptimalCutoffSelectorPolars(AbstractOptimalCutoffSelector[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = OptimalCutoffSelectorPandas()

    def select(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.select(data, column, **kwargs)

class DendrogramDataExtractorPolars(AbstractDendrogramDataExtractor[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = DendrogramDataExtractorPandas()

    def extract(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.extract(data, column, **kwargs)

class HierarchicalClusterProfileBuilderPolars(AbstractHierarchicalClusterProfileBuilder[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = HierarchicalClusterProfileBuilderPandas()

    def build(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.build(data, column, **kwargs)

class HierarchicalClusterCalculatorPolars(AbstractHierarchicalClusterCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = HierarchicalClusterCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class ElbowMethodCalculatorPolars(AbstractElbowMethodCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = ElbowMethodCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class SilhouetteScoreCalculatorPolars(AbstractSilhouetteScoreCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = SilhouetteScoreCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class OptimalKSelectorPolars(AbstractOptimalKSelector[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = OptimalKSelectorPandas()

    def select(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.select(data, column, **kwargs)

class ClusterProfileBuilderPolars(AbstractClusterProfileBuilder[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = ClusterProfileBuilderPandas()

    def build(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.build(data, column, **kwargs)

class KMeansClusterCalculatorPolars(AbstractKMeansClusterCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = KMeansClusterCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class WelchTTestComparatorPolars(AbstractWelchTTestComparator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = WelchTTestComparatorPandas()

    def compare(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.compare(data, column, **kwargs)

class CohensDComputerPolars(AbstractCohensDComputer[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = CohensDComputerPandas()

    def compute(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.compute(data, column, **kwargs)

class EffectMagnitudeClassifierPolars(AbstractEffectMagnitudeClassifier[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = EffectMagnitudeClassifierPandas()

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

class CategoricalDistributionComparatorPolars(AbstractCategoricalDistributionComparator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = CategoricalDistributionComparatorPandas()

    def compare(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.compare(data, column, **kwargs)

class PopulationSplitsCalculatorPolars(AbstractPopulationSplitsCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = PopulationSplitsCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class RFMMetricsComputerPolars(AbstractRFMMetricsComputer[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = RFMMetricsComputerPandas()

    def compute(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.compute(data, column, **kwargs)

class QuantileRFMScorerPolars(AbstractQuantileRFMScorer[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = QuantileRFMScorerPandas()

    def score(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.score(data, column, **kwargs)

class RFMSegmentAssignerPolars(AbstractRFMSegmentAssigner[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = RFMSegmentAssignerPandas()

    def assign(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.assign(data, column, **kwargs)

class RFMSegmentationCalculatorPolars(AbstractRFMSegmentationCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = RFMSegmentationCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)
