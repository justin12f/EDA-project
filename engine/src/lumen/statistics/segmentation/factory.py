"""Factory for the segmentation statistics domain."""
from __future__ import annotations

from lumen.core.abstract_factory import RegistryFactory


class SegmentationStatisticsFactory(RegistryFactory):
    """Registry factory scoped to the segmentation statistics domain."""


def _register() -> None:
    from segmentation.backends import polars_impl as pl_impl
    from segmentation.backends import spark_impl as sp_impl
    from segmentation.backends import pandas_impl as pd_impl

    # --- Polars ---
    SegmentationStatisticsFactory.register("cohort_analysis_calculator", "polars", pl_impl.CohortAnalysisCalculatorPolars)
    SegmentationStatisticsFactory.register("dbscan_clusters_calculator", "polars", pl_impl.DBSCANClustersCalculatorPolars)
    SegmentationStatisticsFactory.register("hierarchical_clusters_calculator", "polars", pl_impl.HierarchicalClustersCalculatorPolars)
    SegmentationStatisticsFactory.register("kmeans_clusters_calculator", "polars", pl_impl.KMeansClustersCalculatorPolars)
    SegmentationStatisticsFactory.register("population_splits_calculator", "polars", pl_impl.PopulationSplitsCalculatorPolars)
    SegmentationStatisticsFactory.register("rfm_segmentation_calculator", "polars", pl_impl.RFMSegmentationCalculatorPolars)

    # --- Spark ---
    SegmentationStatisticsFactory.register("cohort_analysis_calculator", "spark", sp_impl.CohortAnalysisCalculatorSpark)
    SegmentationStatisticsFactory.register("dbscan_clusters_calculator", "spark", sp_impl.DBSCANClustersCalculatorSpark)
    SegmentationStatisticsFactory.register("hierarchical_clusters_calculator", "spark", sp_impl.HierarchicalClustersCalculatorSpark)
    SegmentationStatisticsFactory.register("kmeans_clusters_calculator", "spark", sp_impl.KMeansClustersCalculatorSpark)
    SegmentationStatisticsFactory.register("population_splits_calculator", "spark", sp_impl.PopulationSplitsCalculatorSpark)
    SegmentationStatisticsFactory.register("rfm_segmentation_calculator", "spark", sp_impl.RFMSegmentationCalculatorSpark)

    # --- Pandas ---
    SegmentationStatisticsFactory.register("cohort_analysis_calculator", "pandas", pd_impl.CohortAnalysisCalculatorPandas)
    SegmentationStatisticsFactory.register("dbscan_clusters_calculator", "pandas", pd_impl.DBSCANClustersCalculatorPandas)
    SegmentationStatisticsFactory.register("hierarchical_clusters_calculator", "pandas", pd_impl.HierarchicalClustersCalculatorPandas)
    SegmentationStatisticsFactory.register("kmeans_clusters_calculator", "pandas", pd_impl.KMeansClustersCalculatorPandas)
    SegmentationStatisticsFactory.register("population_splits_calculator", "pandas", pd_impl.PopulationSplitsCalculatorPandas)
    SegmentationStatisticsFactory.register("rfm_segmentation_calculator", "pandas", pd_impl.RFMSegmentationCalculatorPandas)


_register()
