"""Abstract contract for Geo Clustering."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class AbstractGeoClusteringCalculator(ABC):
    """Contract for geospatial clustering (e.g. DBSCAN).

    Parameters
    ----------
    data:
        Backend-native dataframe containing coordinates.
    lat_column:
        Latitude column name.
    lon_column:
        Longitude column name.
    eps_km:
        Maximum distance between two samples for one to be considered in the neighborhood of the other (in km).
    min_samples:
        The number of samples in a neighborhood for a point to be considered as a core point.

    Returns
    -------
    dict[str, Any]
        Dictionary with cluster assignments and centroids.
    """

    @abstractmethod
    def calculate(
        self,
        data: Any,
        lat_column: str,
        lon_column: str,
        eps_km: float = 1.0,
        min_samples: int = 5,
    ) -> dict[str, Any]:
        """Perform geospatial clustering."""
