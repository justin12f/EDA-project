"""Abstract contracts for the segmentation statistics domain."""
from segmentation.abstract.cohort_analysis import AbstractCohortAnalysisCalculator
from segmentation.abstract.dbscan_clusters import AbstractDBSCANClustersCalculator
from segmentation.abstract.hierarchical_clusters import AbstractHierarchicalClustersCalculator
from segmentation.abstract.kmeans_clusters import AbstractKMeansClustersCalculator
from segmentation.abstract.population_splits import AbstractPopulationSplitsCalculator
from segmentation.abstract.rfm_segmentation import AbstractRFMSegmentationCalculator

__all__ = [
    "AbstractCohortAnalysisCalculator",
    "AbstractDBSCANClustersCalculator",
    "AbstractHierarchicalClustersCalculator",
    "AbstractKMeansClustersCalculator",
    "AbstractPopulationSplitsCalculator",
    "AbstractRFMSegmentationCalculator",
]
