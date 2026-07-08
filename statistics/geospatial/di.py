"""Dependency injection container for the geospatial statistics domain."""
from __future__ import annotations

from typing import Literal

from geospatial.abstract.geo_bounding_box import AbstractGeoBoundingBoxCalculator
from geospatial.abstract.geo_clustering import AbstractGeoClusteringCalculator
from geospatial.abstract.geo_distribution import AbstractGeoDistributionCalculator
from geospatial.abstract.geo_heatmap import AbstractGeoHeatmapCalculator
from geospatial.abstract.proximity_analysis import AbstractProximityAnalysisCalculator
from geospatial.factory import GeospatialStatisticsFactory

Backend = Literal["polars", "spark", "pandas"]


class GeospatialDependencyContainer:
    def __init__(self, backend: Backend) -> None:
        self._backend: Backend = backend
        self._factory = GeospatialStatisticsFactory

        self._geo_bounding_box: AbstractGeoBoundingBoxCalculator | None = None
        self._geo_clustering: AbstractGeoClusteringCalculator | None = None
        self._geo_distribution: AbstractGeoDistributionCalculator | None = None
        self._geo_heatmap: AbstractGeoHeatmapCalculator | None = None
        self._proximity_analysis: AbstractProximityAnalysisCalculator | None = None

    @property
    def backend(self) -> Backend:
        return self._backend

    def geo_bounding_box_calculator(self) -> AbstractGeoBoundingBoxCalculator:
        if self._geo_bounding_box is None:
            self._geo_bounding_box = self._factory.create("geo_bounding_box_calculator", self._backend)
        return self._geo_bounding_box

    def geo_clustering_calculator(self) -> AbstractGeoClusteringCalculator:
        if self._geo_clustering is None:
            self._geo_clustering = self._factory.create("geo_clustering_calculator", self._backend)
        return self._geo_clustering

    def geo_distribution_calculator(self) -> AbstractGeoDistributionCalculator:
        if self._geo_distribution is None:
            self._geo_distribution = self._factory.create("geo_distribution_calculator", self._backend)
        return self._geo_distribution

    def geo_heatmap_calculator(self) -> AbstractGeoHeatmapCalculator:
        if self._geo_heatmap is None:
            self._geo_heatmap = self._factory.create("geo_heatmap_calculator", self._backend)
        return self._geo_heatmap

    def proximity_analysis_calculator(self) -> AbstractProximityAnalysisCalculator:
        if self._proximity_analysis is None:
            self._proximity_analysis = self._factory.create("proximity_analysis_calculator", self._backend)
        return self._proximity_analysis
