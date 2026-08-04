"""Factory for the geospatial statistics domain."""
from __future__ import annotations

from lumen.core.abstract_factory import RegistryFactory


class GeospatialStatisticsFactory(RegistryFactory):
    """Registry factory scoped to the geospatial statistics domain."""


def _register() -> None:
    from geospatial.backends import polars_impl as pl_impl
    from geospatial.backends import spark_impl as sp_impl
    from geospatial.backends import pandas_impl as pd_impl

    # --- Polars ---
    GeospatialStatisticsFactory.register("geo_bounding_box_calculator", "polars", pl_impl.GeoBoundingBoxCalculatorPolars)
    GeospatialStatisticsFactory.register("geo_clustering_calculator", "polars", pl_impl.GeoClusteringCalculatorPolars)
    GeospatialStatisticsFactory.register("geo_distribution_calculator", "polars", pl_impl.GeoDistributionCalculatorPolars)
    GeospatialStatisticsFactory.register("geo_heatmap_calculator", "polars", pl_impl.GeoHeatmapCalculatorPolars)
    GeospatialStatisticsFactory.register("proximity_analysis_calculator", "polars", pl_impl.ProximityAnalysisCalculatorPolars)

    # --- Spark ---
    GeospatialStatisticsFactory.register("geo_bounding_box_calculator", "spark", sp_impl.GeoBoundingBoxCalculatorSpark)
    GeospatialStatisticsFactory.register("geo_clustering_calculator", "spark", sp_impl.GeoClusteringCalculatorSpark)
    GeospatialStatisticsFactory.register("geo_distribution_calculator", "spark", sp_impl.GeoDistributionCalculatorSpark)
    GeospatialStatisticsFactory.register("geo_heatmap_calculator", "spark", sp_impl.GeoHeatmapCalculatorSpark)
    GeospatialStatisticsFactory.register("proximity_analysis_calculator", "spark", sp_impl.ProximityAnalysisCalculatorSpark)

    # --- Pandas ---
    GeospatialStatisticsFactory.register("geo_bounding_box_calculator", "pandas", pd_impl.GeoBoundingBoxCalculatorPandas)
    GeospatialStatisticsFactory.register("geo_clustering_calculator", "pandas", pd_impl.GeoClusteringCalculatorPandas)
    GeospatialStatisticsFactory.register("geo_distribution_calculator", "pandas", pd_impl.GeoDistributionCalculatorPandas)
    GeospatialStatisticsFactory.register("geo_heatmap_calculator", "pandas", pd_impl.GeoHeatmapCalculatorPandas)
    GeospatialStatisticsFactory.register("proximity_analysis_calculator", "pandas", pd_impl.ProximityAnalysisCalculatorPandas)


_register()
