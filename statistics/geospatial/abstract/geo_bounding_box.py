"""Abstract contract for Geo Bounding Box."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class AbstractGeoBoundingBoxCalculator(ABC):
    """Contract for calculating geospatial bounding boxes and centroids.

    Parameters
    ----------
    data:
        Backend-native dataframe containing coordinates.
    lat_column:
        Latitude column name.
    lon_column:
        Longitude column name.

    Returns
    -------
    dict[str, Any]
        Dictionary with bounding box coordinates and centroid.
    """

    @abstractmethod
    def calculate(
        self,
        data: Any,
        lat_column: str,
        lon_column: str,
    ) -> dict[str, Any]:
        """Calculate bounding box and centroid."""
