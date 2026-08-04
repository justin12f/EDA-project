"""Pandas adapter backend for the geospatial statistics domain."""
from __future__ import annotations

from typing import Any
import pandas as pd

from geospatial.abstract.geo_bounding_box import AbstractGeoBoundingBoxCalculator
from geospatial.abstract.geo_clustering import AbstractGeoClusteringCalculator
from geospatial.abstract.geo_distribution import AbstractGeoDistributionCalculator
from geospatial.abstract.geo_heatmap import AbstractGeoHeatmapCalculator
from geospatial.abstract.proximity_analysis import AbstractProximityAnalysisCalculator

from geospatial.geo_bounding_box import GeoBoundingBoxCalculator
from geospatial.geo_clustering import GeoClusteringCalculator
from geospatial.geo_distribution import GeoDistributionCalculator
from geospatial.geo_heatmap import GeoHeatmapCalculator
from geospatial.proximity_analysis import ProximityAnalysisCalculator
from lumen.core.frame_extract import column_to_numpy


class GeoBoundingBoxCalculatorPandas(AbstractGeoBoundingBoxCalculator):
    def __init__(self) -> None:
        self._impl = GeoBoundingBoxCalculator()

    def calculate(
        self,
        data: Any,
        lat_column: str,
        lon_column: str,
    ) -> dict[str, Any]:
        df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        coords = df[[lat_column, lon_column]].dropna().values
        return self._impl.calculate(coords)


class GeoClusteringCalculatorPandas(AbstractGeoClusteringCalculator):
    def __init__(self) -> None:
        self._impl = GeoClusteringCalculator()

    def calculate(
        self,
        data: Any,
        lat_column: str,
        lon_column: str,
        eps_km: float = 1.0,
        min_samples: int = 5,
    ) -> dict[str, Any]:
        df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        coords = df[[lat_column, lon_column]].dropna().values
        return self._impl.calculate(
            coords, eps_km=eps_km, min_samples=min_samples
        )


class GeoDistributionCalculatorPandas(AbstractGeoDistributionCalculator):
    def __init__(self) -> None:
        self._impl = GeoDistributionCalculator()

    def calculate(
        self,
        data: Any,
        lat_column: str,
        lon_column: str,
        weight_column: str | None = None,
    ) -> dict[str, Any]:
        df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        clean = df[[lat_column, lon_column] + ([weight_column] if weight_column else [])].dropna()
        coords = clean[[lat_column, lon_column]].values
        weights = clean[weight_column].values if weight_column else None
        return self._impl.calculate(coords, weights=weights)


class GeoHeatmapCalculatorPandas(AbstractGeoHeatmapCalculator):
    def __init__(self) -> None:
        self._impl = GeoHeatmapCalculator()

    def calculate(
        self,
        data: Any,
        lat_column: str,
        lon_column: str,
        weight_column: str | None = None,
        grid_size_lat: int = 50,
        grid_size_lon: int = 50,
    ) -> dict[str, Any]:
        df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        clean = df[[lat_column, lon_column] + ([weight_column] if weight_column else [])].dropna()
        coords = clean[[lat_column, lon_column]].values
        weights = clean[weight_column].values if weight_column else None
        return self._impl.calculate(
            coords, weights=weights,
            grid_size_lat=grid_size_lat, grid_size_lon=grid_size_lon
        )


class ProximityAnalysisCalculatorPandas(AbstractProximityAnalysisCalculator):
    def __init__(self) -> None:
        self._impl = ProximityAnalysisCalculator()

    def calculate(
        self,
        data: Any,
        lat_column: str,
        lon_column: str,
        reference_lat: float,
        reference_lon: float,
        max_distance_km: float | None = None,
    ) -> dict[str, Any]:
        df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        coords = df[[lat_column, lon_column]].dropna().values
        reference_point = (reference_lat, reference_lon)
        return self._impl.calculate(
            coords, reference_point, max_distance_km=max_distance_km
        )
