"""Abstract contract for Geo Distribution."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class AbstractGeoDistributionCalculator(ABC):
    """Contract for analyzing geospatial distribution metrics (e.g. spatial variance, standard distance).

    Parameters
    ----------
    data:
        Backend-native dataframe containing coordinates.
    lat_column:
        Latitude column name.
    lon_column:
        Longitude column name.
    weight_column:
        Optional column for weighted distribution.

    Returns
    -------
    dict[str, Any]
        Dictionary with spatial distribution metrics.
    """

    @abstractmethod
    def calculate(
        self,
        data: Any,
        lat_column: str,
        lon_column: str,
        weight_column: str | None = None,
    ) -> dict[str, Any]:
        """Calculate spatial distribution metrics."""
