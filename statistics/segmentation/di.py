"""Dependency injection container for the segmentation statistics domain."""
from __future__ import annotations

from typing import Literal

from segmentation.abstract.cohort_analysis import AbstractCohortAnalysisCalculator
from segmentation.abstract.dbscan_clusters import AbstractDBSCANClustersCalculator
from segmentation.abstract.hierarchical_clusters import AbstractHierarchicalClustersCalculator
from segmentation.abstract.kmeans_clusters import AbstractKMeansClustersCalculator
from segmentation.abstract.population_splits import AbstractPopulationSplitsCalculator
from segmentation.abstract.rfm_segmentation import AbstractRFMSegmentationCalculator
from segmentation.factory import SegmentationStatisticsFactory

Backend = Literal["polars", "spark", "pandas"]


class SegmentationDependencyContainer:
    def __init__(self, backend: Backend) -> None:
        self._backend: Backend = backend
        self._factory = SegmentationStatisticsFactory

        self._cohort_analysis: AbstractCohortAnalysisCalculator | None = None
        self._dbscan_clusters: AbstractDBSCANClustersCalculator | None = None
        self._hierarchical_clusters: AbstractHierarchicalClustersCalculator | None = None
        self._kmeans_clusters: AbstractKMeansClustersCalculator | None = None
        self._population_splits: AbstractPopulationSplitsCalculator | None = None
        self._rfm_segmentation: AbstractRFMSegmentationCalculator | None = None

    @property
    def backend(self) -> Backend:
        return self._backend

    def cohort_analysis_calculator(self) -> AbstractCohortAnalysisCalculator:
        if self._cohort_analysis is None:
            self._cohort_analysis = self._factory.create("cohort_analysis_calculator", self._backend)
        return self._cohort_analysis

    def dbscan_clusters_calculator(self) -> AbstractDBSCANClustersCalculator:
        if self._dbscan_clusters is None:
            self._dbscan_clusters = self._factory.create("dbscan_clusters_calculator", self._backend)
        return self._dbscan_clusters

    def hierarchical_clusters_calculator(self) -> AbstractHierarchicalClustersCalculator:
        if self._hierarchical_clusters is None:
            self._hierarchical_clusters = self._factory.create("hierarchical_clusters_calculator", self._backend)
        return self._hierarchical_clusters

    def kmeans_clusters_calculator(self) -> AbstractKMeansClustersCalculator:
        if self._kmeans_clusters is None:
            self._kmeans_clusters = self._factory.create("kmeans_clusters_calculator", self._backend)
        return self._kmeans_clusters

    def population_splits_calculator(self) -> AbstractPopulationSplitsCalculator:
        if self._population_splits is None:
            self._population_splits = self._factory.create("population_splits_calculator", self._backend)
        return self._population_splits

    def rfm_segmentation_calculator(self) -> AbstractRFMSegmentationCalculator:
        if self._rfm_segmentation is None:
            self._rfm_segmentation = self._factory.create("rfm_segmentation_calculator", self._backend)
        return self._rfm_segmentation
