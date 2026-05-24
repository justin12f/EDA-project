"""Geographic bounding box, centroid, and spatial dispersion metrics."""

# #[AI_CONTEXT_START]
# - CONFIGURACIÓN DE FACTORY: Registrar este calculator en `GeospatialStatisticsFactory` (backends pandas | polars | spark) y exponerlo mediante `StatisticsInyeccionDependency`, inyectada por la Factory Maestra de Agentes junto a las demás fábricas de dominio.
# - ABSTRACCIÓN DEL DATO: Mutar constructores y `analyze`/`compute` para recibir el contenedor abstracto del backend (`pd.DataFrame`, `pl.DataFrame`/`pl.LazyFrame`, `pyspark.sql.DataFrame`) inyectado por la factory; eliminar `np.ndarray`/`pd.Series` sueltos en firmas públicas.
# - REFACTOR NATIVO: Resolver métricas con expresiones 100 % nativas del backend activo (Polars: `.select`/`.group_by`/`.agg` sin `.collect()` salvo materialización acordada; PySpark: `pyspark.sql.functions` y ventanas distribuidas; Pandas: operaciones vectorizadas). No convertir a NumPy/Pandas desde backends no-pandas.
# #[AI_CONTEXT_END]
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

@dataclass(frozen=True)
class BoundingBoxResult:
    """Immutable bounding box and centroid result."""

    centroid_lat: float
    centroid_lon: float
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    lat_range_deg: float
    lon_range_deg: float
    diagonal_km: float
    mean_radius_km: float
    dispersion_label: str

class CentroidCalculator:
    """Computes the geographic centroid of a point set.

    Uses arithmetic mean of lat/lon — valid approximation for
    point sets within a single continent or country. For global
    datasets spanning the antimeridian (180°/-180°), arithmetic
    mean produces incorrect results.
    """

    def calculate(
        self, lat: np.ndarray, lon: np.ndarray
    ) -> tuple[float, float]:
        """Compute geographic centroid.

        Args:
            lat: Latitude array in decimal degrees.
            lon: Longitude array in decimal degrees.

        Returns:
            Tuple (centroid_lat, centroid_lon).
        """
        return float(lat.mean()), float(lon.mean())

class BoundingBoxCalculator:
    """Computes bounding box extents for a coordinate set."""

    def calculate(
        self, lat: np.ndarray, lon: np.ndarray
    ) -> tuple[float, float, float, float]:
        """Compute bounding box.

        Args:
            lat: Latitude array.
            lon: Longitude array.

        Returns:
            Tuple (lat_min, lat_max, lon_min, lon_max).
        """
        return (
            float(lat.min()), float(lat.max()),
            float(lon.min()), float(lon.max()),
        )

class DiagonalDistanceCalculator:
    """Computes the Haversine diagonal of a bounding box.

    The diagonal represents the worst-case distance within the box —
    a useful proxy for the spatial extent of the dataset.
    """

    _EARTH_RADIUS_KM: float = 6371.0

    def calculate(
        self,
        lat_min: float, lat_max: float,
        lon_min: float, lon_max: float,
    ) -> float:
        """Compute Haversine distance between bounding box corners.

        Args:
            lat_min: Southern boundary.
            lat_max: Northern boundary.
            lon_min: Western boundary.
            lon_max: Eastern boundary.

        Returns:
            Diagonal distance in kilometers.
        """
        lat1, lon1 = np.radians(lat_min), np.radians(lon_min)
        lat2, lon2 = np.radians(lat_max), np.radians(lon_max)

        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        )
        return float(2 * self._EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1))))

class MeanRadiusCalculator:
    """Computes mean Haversine distance from the centroid to all points.

    Mean radius characterizes the average spread of points around their
    geographic center — independent of bounding box shape.
    """

    _EARTH_RADIUS_KM: float = 6371.0

    def calculate(
        self,
        lat: np.ndarray, lon: np.ndarray,
        centroid_lat: float, centroid_lon: float,
    ) -> float:
        """Compute mean Haversine radius from centroid.

        Args:
            lat: Latitude array.
            lon: Longitude array.
            centroid_lat: Centroid latitude.
            centroid_lon: Centroid longitude.

        Returns:
            Mean radius in kilometers.
        """
        lat_r = np.radians(lat)
        lon_r = np.radians(lon)
        c_lat = np.radians(centroid_lat)
        c_lon = np.radians(centroid_lon)

        dlat = lat_r - c_lat
        dlon = lon_r - c_lon
        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(c_lat) * np.cos(lat_r) * np.sin(dlon / 2) ** 2
        )
        distances = 2 * self._EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        return float(distances.mean())

class DispersionLabelAssigner:
    """Assigns a human-readable dispersion label based on mean radius.

    Thresholds (approximate geographic scales):
        < 5 km:     hyper_local  (neighborhood/district level)
        < 50 km:    local        (city level)
        < 500 km:   regional     (country/state level)
        < 2000 km:  continental  (multi-country)
        >= 2000 km: global
    """

    _THRESHOLDS: list[tuple[float, str]] = [
        (2000.0, "global"),
        (500.0,  "continental"),
        (50.0,   "regional"),
        (5.0,    "local"),
        (0.0,    "hyper_local"),
    ]

    def assign(self, mean_radius_km: float) -> str:
        """Assign dispersion label.

        Args:
            mean_radius_km: Mean radius from centroid in km.

        Returns:
            Dispersion label string.
        """
        for threshold, label in self._THRESHOLDS:
            if mean_radius_km >= threshold:
                return label
        return "hyper_local"

class GeoBoundingBoxCalculator:
    """Bounding box, centroid, diagonal, and dispersion for a coordinate set.

    Workflow:
        calculator = GeoBoundingBoxCalculator()
        result = calculator.calculate(
            data_frame=df,
            lat_column="latitude",
            lon_column="longitude",
        )
    """

    _MINIMUM_POINTS: int = 2

    def __init__(self) -> None:
        self._centroid_calc = CentroidCalculator()
        self._bbox_calc = BoundingBoxCalculator()
        self._diagonal_calc = DiagonalDistanceCalculator()
        self._radius_calc = MeanRadiusCalculator()
        self._label_assigner = DispersionLabelAssigner()

    def calculate(
        self,
        data_frame: pd.DataFrame,
        lat_column: str,
        lon_column: str,
    ) -> dict:
        """Compute bounding box and spatial dispersion metrics.

        Args:
            data_frame: Source DataFrame.
            lat_column: Latitude column (decimal degrees).
            lon_column: Longitude column (decimal degrees).

        Returns:
            Dict with centroid, bounding box, diagonal, mean radius, and label.

        Raises:
            KeyError: If coordinate columns are not found.
            ValueError: If fewer than 2 valid coordinate pairs.
        """
        for col in (lat_column, lon_column):
            if col not in data_frame.columns:
                raise KeyError(f"Column '{col}' not found in DataFrame.")

        clean = data_frame[[lat_column, lon_column]].dropna()

        if len(clean) < self._MINIMUM_POINTS:
            raise ValueError(
                f"At least {self._MINIMUM_POINTS} coordinate pairs required. "
                f"Got {len(clean)}."
            )

        lat = clean[lat_column].to_numpy(dtype=float)
        lon = clean[lon_column].to_numpy(dtype=float)

        centroid_lat, centroid_lon = self._centroid_calc.calculate(lat, lon)
        lat_min, lat_max, lon_min, lon_max = self._bbox_calc.calculate(lat, lon)
        diagonal_km = self._diagonal_calc.calculate(lat_min, lat_max, lon_min, lon_max)
        mean_radius_km = self._radius_calc.calculate(lat, lon, centroid_lat, centroid_lon)
        dispersion_label = self._label_assigner.assign(mean_radius_km)

        result = BoundingBoxResult(
            centroid_lat=round(centroid_lat, 6),
            centroid_lon=round(centroid_lon, 6),
            lat_min=round(lat_min, 6),
            lat_max=round(lat_max, 6),
            lon_min=round(lon_min, 6),
            lon_max=round(lon_max, 6),
            lat_range_deg=round(lat_max - lat_min, 6),
            lon_range_deg=round(lon_max - lon_min, 6),
            diagonal_km=round(diagonal_km, 2),
            mean_radius_km=round(mean_radius_km, 2),
            dispersion_label=dispersion_label,
        )

        return {
            "centroid": {
                "lat": result.centroid_lat,
                "lon": result.centroid_lon,
            },
            "bounding_box": {
                "lat_min": result.lat_min,
                "lat_max": result.lat_max,
                "lon_min": result.lon_min,
                "lon_max": result.lon_max,
                "lat_range_deg": result.lat_range_deg,
                "lon_range_deg": result.lon_range_deg,
            },
            "spatial_metrics": {
                "diagonal_km": result.diagonal_km,
                "mean_radius_km": result.mean_radius_km,
                "dispersion_label": result.dispersion_label,
            },
            "n_points": len(clean),
        }
