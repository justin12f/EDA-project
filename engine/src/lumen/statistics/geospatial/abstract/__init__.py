"""Abstract contracts for the geospatial statistics domain."""
from geospatial.abstract.geo_bounding_box import AbstractGeoBoundingBoxCalculator
from geospatial.abstract.geo_clustering import AbstractGeoClusteringCalculator
from geospatial.abstract.geo_distribution import AbstractGeoDistributionCalculator
from geospatial.abstract.geo_heatmap import AbstractGeoHeatmapCalculator
from geospatial.abstract.proximity_analysis import AbstractProximityAnalysisCalculator

__all__ = [
    "AbstractGeoBoundingBoxCalculator",
    "AbstractGeoClusteringCalculator",
    "AbstractGeoDistributionCalculator",
    "AbstractGeoHeatmapCalculator",
    "AbstractProximityAnalysisCalculator",
]
