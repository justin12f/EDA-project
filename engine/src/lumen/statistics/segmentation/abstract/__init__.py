"""Abstract contracts for the segmentation statistics domain."""
from lumen.statistics.segmentation.abstract.cohort_analysis import AbstractCohortAnalysisCalculator
from lumen.statistics.segmentation.abstract.dbscan_clusters import AbstractDBSCANClustersCalculator
from lumen.statistics.segmentation.abstract.hierarchical_clusters import AbstractHierarchicalClustersCalculator
from lumen.statistics.segmentation.abstract.kmeans_clusters import AbstractKMeansClustersCalculator
from lumen.statistics.segmentation.abstract.population_splits import AbstractPopulationSplitsCalculator
from lumen.statistics.segmentation.abstract.rfm_segmentation import AbstractRFMSegmentationCalculator

__all__ = [
    "AbstractCohortAnalysisCalculator",
    "AbstractDBSCANClustersCalculator",
    "AbstractHierarchicalClustersCalculator",
    "AbstractKMeansClustersCalculator",
    "AbstractPopulationSplitsCalculator",
    "AbstractRFMSegmentationCalculator",
]
