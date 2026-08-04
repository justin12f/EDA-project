"""Abstract contract for Geo Heatmap."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class AbstractGeoHeatmapCalculator(ABC):
    """Contract for generating geospatial heatmap grid data (density or weighted density).

    Parameters
    ----------
    data:
        Backend-native dataframe containing coordinates.
    lat_column:
        Latitude column name.
    lon_column:
        Longitude column name.
    weight_column:
        Optional weighting column.
    grid_size_lat:
        Number of bins in latitude.
    grid_size_lon:
        Number of bins in longitude.

    Returns
    -------
    dict[str, Any]
        Dictionary with gridded heatmap data.
    """

    @abstractmethod
    def calculate(
        self,
        data: Any,
        lat_column: str,
        lon_column: str,
        weight_column: str | None = None,
        grid_size_lat: int = 50,
        grid_size_lon: int = 50,
    ) -> dict[str, Any]:
        """Calculate heatmap density grid."""
