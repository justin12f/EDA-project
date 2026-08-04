"""Abstract contract for Proximity Analysis."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class AbstractProximityAnalysisCalculator(ABC):
    """Contract for proximity and distance analysis (nearest neighbor, distance matrix).

    Parameters
    ----------
    data:
        Backend-native dataframe containing target coordinates.
    lat_column:
        Latitude column name.
    lon_column:
        Longitude column name.
    reference_lat:
        Latitude of the reference point.
    reference_lon:
        Longitude of the reference point.
    max_distance_km:
        Maximum radius to consider for neighbors (optional).

    Returns
    -------
    dict[str, Any]
        Dictionary with proximity metrics (distances to reference, nearest count within radius).
    """

    @abstractmethod
    def calculate(
        self,
        data: Any,
        lat_column: str,
        lon_column: str,
        reference_lat: float,
        reference_lon: float,
        max_distance_km: float | None = None,
    ) -> dict[str, Any]:
        """Calculate distance metrics relative to a reference point."""
