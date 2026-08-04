"""Pandas adapter backend for the segmentation statistics domain."""
from __future__ import annotations

from typing import Any
import pandas as pd

from segmentation.abstract.cohort_analysis import AbstractCohortAnalysisCalculator
from segmentation.abstract.dbscan_clusters import AbstractDBSCANClustersCalculator
from segmentation.abstract.hierarchical_clusters import AbstractHierarchicalClustersCalculator
from segmentation.abstract.kmeans_clusters import AbstractKMeansClustersCalculator
from segmentation.abstract.population_splits import AbstractPopulationSplitsCalculator
from segmentation.abstract.rfm_segmentation import AbstractRFMSegmentationCalculator

from segmentation.cohort_analysis import CohortAnalysisCalculator
from segmentation.dbscan_clusters import DBSCANClustersCalculator
from segmentation.hierarchical_clusters import HierarchicalClustersCalculator
from segmentation.kmeans_clusters import KMeansClustersCalculator
from segmentation.population_splits import PopulationSplitsCalculator
from segmentation.rfm_segmentation import RFMSegmentationCalculator


class CohortAnalysisCalculatorPandas(AbstractCohortAnalysisCalculator):
    def __init__(self) -> None:
        self._impl = CohortAnalysisCalculator()

    def calculate(
        self,
        data: Any,
        user_column: str,
        date_column: str,
        period: str = "month",
    ) -> dict[str, Any]:
        df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        return self._impl.calculate(df, user_column, date_column, period)


class DBSCANClustersCalculatorPandas(AbstractDBSCANClustersCalculator):
    def __init__(self) -> None:
        self._impl = DBSCANClustersCalculator()

    def calculate(
        self,
        data: Any,
        features: list[str],
        eps: float = 0.5,
        min_samples: int = 5,
    ) -> dict[str, Any]:
        df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        return self._impl.calculate(df, features, eps, min_samples)


class HierarchicalClustersCalculatorPandas(AbstractHierarchicalClustersCalculator):
    def __init__(self) -> None:
        self._impl = HierarchicalClustersCalculator()

    def calculate(
        self,
        data: Any,
        features: list[str],
        n_clusters: int = 3,
        linkage: str = "ward",
    ) -> dict[str, Any]:
        df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        return self._impl.calculate(df, features, n_clusters, linkage)


class KMeansClustersCalculatorPandas(AbstractKMeansClustersCalculator):
    def __init__(self) -> None:
        self._impl = KMeansClustersCalculator()

    def calculate(
        self,
        data: Any,
        features: list[str],
        n_clusters: int = 3,
        random_state: int = 42,
    ) -> dict[str, Any]:
        df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        return self._impl.calculate(df, features, n_clusters, random_state)


class PopulationSplitsCalculatorPandas(AbstractPopulationSplitsCalculator):
    def __init__(self) -> None:
        self._impl = PopulationSplitsCalculator()

    def calculate(
        self,
        data: Any,
        column: str,
        method: str = "quantiles",
        n_bins: int = 4,
    ) -> dict[str, Any]:
        df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        return self._impl.calculate(df, column, method, n_bins)


class RFMSegmentationCalculatorPandas(AbstractRFMSegmentationCalculator):
    def __init__(self) -> None:
        self._impl = RFMSegmentationCalculator()

    def calculate(
        self,
        data: Any,
        customer_column: str,
        date_column: str,
        amount_column: str,
        reference_date: str | None = None,
    ) -> dict[str, Any]:
        df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        return self._impl.calculate(df, customer_column, date_column, amount_column, reference_date)
