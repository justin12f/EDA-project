"""Abstract contract for DBSCAN Clustering."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class AbstractDBSCANClustersCalculator(ABC):
    """Contract for calculating DBSCAN clusters on arbitrary features.

    Parameters
    ----------
    data:
        Backend-native dataframe.
    features:
        List of numeric features to cluster on.
    eps:
        The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    min_samples:
        The number of samples in a neighborhood for a point to be considered as a core point.

    Returns
    -------
    dict[str, Any]
        Dictionary with cluster assignments, noise points, and centroids.
    """

    @abstractmethod
    def calculate(
        self,
        data: Any,
        features: list[str],
        eps: float = 0.5,
        min_samples: int = 5,
    ) -> dict[str, Any]:
        """Perform DBSCAN clustering."""
