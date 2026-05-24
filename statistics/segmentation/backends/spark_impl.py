"""Spark statistics backends — `segmentation`."""
from __future__ import annotations
from typing import Any
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F
from statistics.segmentation.abstract import *

from statistics.segmentation.backends import pandas_impl
from statistics.segmentation.backends.pandas_impl import *

class CohortAssignerSpark(AbstractCohortAssigner[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = CohortAssignerPandas()

    def assign(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.assign(data, column, **kwargs)

class CohortPeriodOffsetCalculatorSpark(AbstractCohortPeriodOffsetCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = CohortPeriodOffsetCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class RetentionMatrixBuilderSpark(AbstractRetentionMatrixBuilder[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = RetentionMatrixBuilderPandas()

    def build(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.build(data, column, **kwargs)

class RetentionRateNormalizerSpark(AbstractRetentionRateNormalizer[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = RetentionRateNormalizerPandas()

    def normalize(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.normalize(data, column, **kwargs)

class CohortAnalysisCalculatorSpark(AbstractCohortAnalysisCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = CohortAnalysisCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class EpsilonEstimatorSpark(AbstractEpsilonEstimator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = EpsilonEstimatorPandas()

    def estimate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.estimate(data, column, **kwargs)

class DBSCANClusterProfileBuilderSpark(AbstractDBSCANClusterProfileBuilder[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = DBSCANClusterProfileBuilderPandas()

    def build(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.build(data, column, **kwargs)

class DBSCANClusterCalculatorSpark(AbstractDBSCANClusterCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = DBSCANClusterCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class LinkageMatrixBuilderSpark(AbstractLinkageMatrixBuilder[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = LinkageMatrixBuilderPandas()

    def build(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.build(data, column, **kwargs)

class CopheneticCorrelationCalculatorSpark(AbstractCopheneticCorrelationCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = CopheneticCorrelationCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class OptimalCutoffSelectorSpark(AbstractOptimalCutoffSelector[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = OptimalCutoffSelectorPandas()

    def select(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.select(data, column, **kwargs)

class DendrogramDataExtractorSpark(AbstractDendrogramDataExtractor[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = DendrogramDataExtractorPandas()

    def extract(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.extract(data, column, **kwargs)

class HierarchicalClusterProfileBuilderSpark(AbstractHierarchicalClusterProfileBuilder[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = HierarchicalClusterProfileBuilderPandas()

    def build(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.build(data, column, **kwargs)

class HierarchicalClusterCalculatorSpark(AbstractHierarchicalClusterCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = HierarchicalClusterCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class ElbowMethodCalculatorSpark(AbstractElbowMethodCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = ElbowMethodCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class SilhouetteScoreCalculatorSpark(AbstractSilhouetteScoreCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = SilhouetteScoreCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class OptimalKSelectorSpark(AbstractOptimalKSelector[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = OptimalKSelectorPandas()

    def select(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.select(data, column, **kwargs)

class ClusterProfileBuilderSpark(AbstractClusterProfileBuilder[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = ClusterProfileBuilderPandas()

    def build(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.build(data, column, **kwargs)

class KMeansClusterCalculatorSpark(AbstractKMeansClusterCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = KMeansClusterCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class WelchTTestComparatorSpark(AbstractWelchTTestComparator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = WelchTTestComparatorPandas()

    def compare(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.compare(data, column, **kwargs)

class CohensDComputerSpark(AbstractCohensDComputer[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = CohensDComputerPandas()

    def compute(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.compute(data, column, **kwargs)

class EffectMagnitudeClassifierSpark(AbstractEffectMagnitudeClassifier[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = EffectMagnitudeClassifierPandas()

    def classify(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.classify(data, column, **kwargs)

class CategoricalDistributionComparatorSpark(AbstractCategoricalDistributionComparator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = CategoricalDistributionComparatorPandas()

    def compare(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.compare(data, column, **kwargs)

class PopulationSplitsCalculatorSpark(AbstractPopulationSplitsCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = PopulationSplitsCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class RFMMetricsComputerSpark(AbstractRFMMetricsComputer[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = RFMMetricsComputerPandas()

    def compute(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.compute(data, column, **kwargs)

class QuantileRFMScorerSpark(AbstractQuantileRFMScorer[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = QuantileRFMScorerPandas()

    def score(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.score(data, column, **kwargs)

class RFMSegmentAssignerSpark(AbstractRFMSegmentAssigner[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = RFMSegmentAssignerPandas()

    def assign(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.assign(data, column, **kwargs)

class RFMSegmentationCalculatorSpark(AbstractRFMSegmentationCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = RFMSegmentationCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)
