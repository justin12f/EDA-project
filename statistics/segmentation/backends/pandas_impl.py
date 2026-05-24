"""Pandas statistics backends — `segmentation`."""
from __future__ import annotations
from typing import Any
import pandas as pd
from statistics.core.frame_extract import column_to_numpy
from statistics.segmentation.abstract import *

import statistics.segmentation.cohort_analysis as _mod_cohort_analysis
import statistics.segmentation.dbscan_clusters as _mod_dbscan_clusters
import statistics.segmentation.hierarchical_clusters as _mod_hierarchical_clusters
import statistics.segmentation.kmeans_clusters as _mod_kmeans_clusters
import statistics.segmentation.population_splits as _mod_population_splits
import statistics.segmentation.rfm_segmentation as _mod_rfm_segmentation

class CohortAssignerPandas(AbstractCohortAssigner[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_cohort_analysis.CohortAssigner()

    def assign(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.assign(arr, **kwargs)

class CohortPeriodOffsetCalculatorPandas(AbstractCohortPeriodOffsetCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_cohort_analysis.CohortPeriodOffsetCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class RetentionMatrixBuilderPandas(AbstractRetentionMatrixBuilder[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_cohort_analysis.RetentionMatrixBuilder()

    def build(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.build(arr, **kwargs)

class RetentionRateNormalizerPandas(AbstractRetentionRateNormalizer[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_cohort_analysis.RetentionRateNormalizer()

    def normalize(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.normalize(arr, **kwargs)

class CohortAnalysisCalculatorPandas(AbstractCohortAnalysisCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_cohort_analysis.CohortAnalysisCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class EpsilonEstimatorPandas(AbstractEpsilonEstimator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_dbscan_clusters.EpsilonEstimator()

    def estimate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.estimate(arr, **kwargs)

class DBSCANClusterProfileBuilderPandas(AbstractDBSCANClusterProfileBuilder[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_dbscan_clusters.DBSCANClusterProfileBuilder()

    def build(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.build(arr, **kwargs)

class DBSCANClusterCalculatorPandas(AbstractDBSCANClusterCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_dbscan_clusters.DBSCANClusterCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class LinkageMatrixBuilderPandas(AbstractLinkageMatrixBuilder[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_hierarchical_clusters.LinkageMatrixBuilder()

    def build(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.build(arr, **kwargs)

class CopheneticCorrelationCalculatorPandas(AbstractCopheneticCorrelationCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_hierarchical_clusters.CopheneticCorrelationCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class OptimalCutoffSelectorPandas(AbstractOptimalCutoffSelector[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_hierarchical_clusters.OptimalCutoffSelector()

    def select(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.select(arr, **kwargs)

class DendrogramDataExtractorPandas(AbstractDendrogramDataExtractor[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_hierarchical_clusters.DendrogramDataExtractor()

    def extract(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.extract(arr, **kwargs)

class HierarchicalClusterProfileBuilderPandas(AbstractHierarchicalClusterProfileBuilder[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_hierarchical_clusters.HierarchicalClusterProfileBuilder()

    def build(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.build(arr, **kwargs)

class HierarchicalClusterCalculatorPandas(AbstractHierarchicalClusterCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_hierarchical_clusters.HierarchicalClusterCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class ElbowMethodCalculatorPandas(AbstractElbowMethodCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_kmeans_clusters.ElbowMethodCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class SilhouetteScoreCalculatorPandas(AbstractSilhouetteScoreCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_kmeans_clusters.SilhouetteScoreCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class OptimalKSelectorPandas(AbstractOptimalKSelector[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_kmeans_clusters.OptimalKSelector()

    def select(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.select(arr, **kwargs)

class ClusterProfileBuilderPandas(AbstractClusterProfileBuilder[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_kmeans_clusters.ClusterProfileBuilder()

    def build(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.build(arr, **kwargs)

class KMeansClusterCalculatorPandas(AbstractKMeansClusterCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_kmeans_clusters.KMeansClusterCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class WelchTTestComparatorPandas(AbstractWelchTTestComparator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_population_splits.WelchTTestComparator()

    def compare(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.compare(arr, **kwargs)

class CohensDComputerPandas(AbstractCohensDComputer[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_population_splits.CohensDComputer()

    def compute(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.compute(arr, **kwargs)

class EffectMagnitudeClassifierPandas(AbstractEffectMagnitudeClassifier[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_population_splits.EffectMagnitudeClassifier()

    def classify(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.classify(arr, **kwargs)

class CategoricalDistributionComparatorPandas(AbstractCategoricalDistributionComparator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_population_splits.CategoricalDistributionComparator()

    def compare(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.compare(arr, **kwargs)

class PopulationSplitsCalculatorPandas(AbstractPopulationSplitsCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_population_splits.PopulationSplitsCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class RFMMetricsComputerPandas(AbstractRFMMetricsComputer[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_rfm_segmentation.RFMMetricsComputer()

    def compute(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.compute(arr, **kwargs)

class QuantileRFMScorerPandas(AbstractQuantileRFMScorer[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_rfm_segmentation.QuantileRFMScorer()

    def score(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.score(arr, **kwargs)

class RFMSegmentAssignerPandas(AbstractRFMSegmentAssigner[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_rfm_segmentation.RFMSegmentAssigner()

    def assign(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.assign(arr, **kwargs)

class RFMSegmentationCalculatorPandas(AbstractRFMSegmentationCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_rfm_segmentation.RFMSegmentationCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)
