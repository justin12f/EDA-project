"""Abstract contract for Hierarchical Clustering."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class AbstractHierarchicalClustersCalculator(ABC):
    """Contract for performing Agglomerative (Hierarchical) Clustering.

    Parameters
    ----------
    data:
        Backend-native dataframe.
    features:
        List of numeric features to cluster on.
    n_clusters:
        Number of clusters to find.
    linkage:
        Linkage criterion ('ward', 'complete', 'average', 'single').

    Returns
    -------
    dict[str, Any]
        Dictionary with cluster assignments and centroids.
    """

    @abstractmethod
    def calculate(
        self,
        data: Any,
        features: list[str],
        n_clusters: int = 3,
        linkage: str = "ward",
    ) -> dict[str, Any]:
        """Perform hierarchical clustering."""
