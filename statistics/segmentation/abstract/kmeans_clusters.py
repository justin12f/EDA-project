"""Abstract contract for KMeans Clustering."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class AbstractKMeansClustersCalculator(ABC):
    """Contract for performing K-Means clustering.

    Parameters
    ----------
    data:
        Backend-native dataframe.
    features:
        List of numeric features to cluster on.
    n_clusters:
        Number of clusters (k).
    random_state:
        Seed for reproducibility.

    Returns
    -------
    dict[str, Any]
        Dictionary with cluster assignments, centroids, and inertia.
    """

    @abstractmethod
    def calculate(
        self,
        data: Any,
        features: list[str],
        n_clusters: int = 3,
        random_state: int = 42,
    ) -> dict[str, Any]:
        """Perform K-Means clustering."""
