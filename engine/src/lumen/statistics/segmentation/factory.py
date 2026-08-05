"""Factory for the segmentation statistics domain."""
from __future__ import annotations

from lumen.core.abstract_factory import RegistryFactory


class SegmentationStatisticsFactory(RegistryFactory):
    """Registry factory scoped to the segmentation statistics domain."""


def _register() -> None:
    from lumen.statistics.segmentation.backends import polars_impl as pl_impl
    try:
        from lumen.statistics.segmentation.backends import spark_impl as sp_impl
    except ImportError:  # PySpark is the optional `spark` extra
        sp_impl = None
    from lumen.statistics.segmentation.backends import pandas_impl as pd_impl

    # --- Polars ---
    SegmentationStatisticsFactory.register("cohort_analysis_calculator", "polars", pl_impl.CohortAnalysisCalculatorPolars)
    SegmentationStatisticsFactory.register("dbscan_clusters_calculator", "polars", pl_impl.DBSCANClustersCalculatorPolars)
    SegmentationStatisticsFactory.register("hierarchical_clusters_calculator", "polars", pl_impl.HierarchicalClustersCalculatorPolars)
    SegmentationStatisticsFactory.register("kmeans_clusters_calculator", "polars", pl_impl.KMeansClustersCalculatorPolars)
    SegmentationStatisticsFactory.register("population_splits_calculator", "polars", pl_impl.PopulationSplitsCalculatorPolars)
    SegmentationStatisticsFactory.register("rfm_segmentation_calculator", "polars", pl_impl.RFMSegmentationCalculatorPolars)

    # --- Spark ---
    if sp_impl is not None:
        SegmentationStatisticsFactory.register("cohort_analysis_calculator", "spark", sp_impl.CohortAnalysisCalculatorSpark)
    if sp_impl is not None:
        SegmentationStatisticsFactory.register("dbscan_clusters_calculator", "spark", sp_impl.DBSCANClustersCalculatorSpark)
    if sp_impl is not None:
        SegmentationStatisticsFactory.register("hierarchical_clusters_calculator", "spark", sp_impl.HierarchicalClustersCalculatorSpark)
    if sp_impl is not None:
        SegmentationStatisticsFactory.register("kmeans_clusters_calculator", "spark", sp_impl.KMeansClustersCalculatorSpark)
    if sp_impl is not None:
        SegmentationStatisticsFactory.register("population_splits_calculator", "spark", sp_impl.PopulationSplitsCalculatorSpark)
    if sp_impl is not None:
        SegmentationStatisticsFactory.register("rfm_segmentation_calculator", "spark", sp_impl.RFMSegmentationCalculatorSpark)

    # --- Pandas ---
    SegmentationStatisticsFactory.register("cohort_analysis_calculator", "pandas", pd_impl.CohortAnalysisCalculatorPandas)
    SegmentationStatisticsFactory.register("dbscan_clusters_calculator", "pandas", pd_impl.DBSCANClustersCalculatorPandas)
    SegmentationStatisticsFactory.register("hierarchical_clusters_calculator", "pandas", pd_impl.HierarchicalClustersCalculatorPandas)
    SegmentationStatisticsFactory.register("kmeans_clusters_calculator", "pandas", pd_impl.KMeansClustersCalculatorPandas)
    SegmentationStatisticsFactory.register("population_splits_calculator", "pandas", pd_impl.PopulationSplitsCalculatorPandas)
    SegmentationStatisticsFactory.register("rfm_segmentation_calculator", "pandas", pd_impl.RFMSegmentationCalculatorPandas)


_register()
