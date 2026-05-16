"""Proximity analysis: nearest neighbor distances and spatial statistics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class NearestNeighborResult:
    """Immutable nearest neighbor result for a single point."""

    point_index: int
    nearest_neighbor_index: int
    distance_km: float


class HaversineVectorizedCalculator:
    """Vectorized Haversine distance from one reference point to all others.

    Vectorized over a single reference point for efficiency — used
    when finding the nearest neighbor for each point in O(n²).
    """

    _EARTH_RADIUS_KM: float = 6371.0

    def distances_from_point(
        self,
        ref_lat: float,
        ref_lon: float,
        lat_array: np.ndarray,
        lon_array: np.ndarray,
    ) -> np.ndarray:
        """Compute Haversine distances from one point to an array.

        Args:
            ref_lat: Reference point latitude (degrees).
            ref_lon: Reference point longitude (degrees).
            lat_array: Target latitude array (degrees).
            lon_array: Target longitude array (degrees).

        Returns:
            Distance array in kilometers.
        """
        ref_lat_r = np.radians(ref_lat)
        ref_lon_r = np.radians(ref_lon)
        lat_r = np.radians(lat_array)
        lon_r = np.radians(lon_array)

        dlat = lat_r - ref_lat_r
        dlon = lon_r - ref_lon_r

        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(ref_lat_r) * np.cos(lat_r) * np.sin(dlon / 2) ** 2
        )
        return 2 * self._EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


class NearestNeighborFinder:
    """Finds the nearest neighbor for each point in a coordinate set.

    Time complexity: O(n²) — acceptable for n < 10,000.
    For larger datasets, a KD-tree or BallTree on radians should be used.
    """

    def __init__(self, haversine_calculator: HaversineVectorizedCalculator) -> None:
        self._haversine = haversine_calculator

    def find_all(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
    ) -> list[NearestNeighborResult]:
        """Find nearest neighbor for every point.

        Args:
            lat: Latitude array.
            lon: Longitude array.

        Returns:
            List of NearestNeighborResult, one per point.
        """
        results: list[NearestNeighborResult] = []

        for i in range(len(lat)):
            distances = self._haversine.distances_from_point(
                lat[i], lon[i], lat, lon
            )
            distances[i] = np.inf  # exclude self
            nn_idx = int(np.argmin(distances))
            results.append(
                NearestNeighborResult(
                    point_index=i,
                    nearest_neighbor_index=nn_idx,
                    distance_km=round(float(distances[nn_idx]), 4),
                )
            )

        return results


class AverageNearestNeighborIndexCalculator:
    """Computes the Average Nearest Neighbor (ANN) index.

    ANN = observed_mean_distance / expected_mean_distance

    Expected mean distance for a random (Poisson) process:
        E[d] = 1 / (2 × √(n/A))

    where A = area of the bounding box in km² and n = point count.

    ANN < 1: clustered pattern (observed distances < expected random).
    ANN ≈ 1: random pattern.
    ANN > 1: dispersed/uniform pattern.
    """

    _EARTH_RADIUS_KM: float = 6371.0

    def calculate(
        self,
        nn_distances: np.ndarray,
        lat: np.ndarray,
        lon: np.ndarray,
    ) -> dict:
        """Compute ANN index and spatial pattern label.

        Args:
            nn_distances: Array of nearest neighbor distances in km.
            lat: Latitude array.
            lon: Longitude array.

        Returns:
            Dict with observed/expected distances, ANN index, and pattern label.
        """
        observed_mean = float(nn_distances.mean())
        n = len(lat)

        # Approximate bounding box area using equirectangular projection
        lat_range_km = (float(lat.max()) - float(lat.min())) * np.pi / 180 * self._EARTH_RADIUS_KM
        mean_lat = float(lat.mean())
        lon_range_km = (
            (float(lon.max()) - float(lon.min()))
            * np.pi / 180
            * self._EARTH_RADIUS_KM
            * np.cos(np.radians(mean_lat))
        )
        area_km2 = lat_range_km * lon_range_km if lat_range_km > 0 and lon_range_km > 0 else 1.0

        density = n / area_km2
        expected_mean = 1.0 / (2.0 * np.sqrt(density)) if density > 0 else float("inf")
        ann_index = observed_mean / expected_mean if expected_mean > 0 else float("inf")

        if ann_index < 0.85:
            pattern = "clustered"
        elif ann_index > 1.15:
            pattern = "dispersed"
        else:
            pattern = "random"

        return {
            "observed_mean_distance_km": round(observed_mean, 4),
            "expected_mean_distance_km": round(expected_mean, 4),
            "ann_index": round(ann_index, 4),
            "spatial_pattern": pattern,
            "area_km2": round(area_km2, 2),
        }


class ProximityAnalysisCalculator:
    """Nearest neighbor distances, mean proximity, and spatial pattern analysis.

    Workflow:
        calculator = ProximityAnalysisCalculator()
        result = calculator.calculate(
            data_frame=df,
            lat_column="latitude",
            lon_column="longitude",
            include_all_nn=False,    # True = return per-point NN results
            max_points=2000,         # safety cap for O(n²) computation
        )
    """

    _MINIMUM_POINTS: int = 3
    _DEFAULT_MAX_POINTS: int = 2_000

    def __init__(self) -> None:
        self._haversine = HaversineVectorizedCalculator()
        self._nn_finder = NearestNeighborFinder(self._haversine)
        self._ann_calculator = AverageNearestNeighborIndexCalculator()

    def calculate(
        self,
        data_frame: pd.DataFrame,
        lat_column: str,
        lon_column: str,
        include_all_nn: bool = False,
        max_points: int = _DEFAULT_MAX_POINTS,
    ) -> dict:
        """Run proximity analysis on coordinate set.

        Args:
            data_frame: Source DataFrame.
            lat_column: Latitude column (decimal degrees).
            lon_column: Longitude column (decimal degrees).
            include_all_nn: Whether to return per-point nearest neighbor results.
            max_points: Maximum points to process (O(n²) safeguard).

        Returns:
            Dict with NN statistics, ANN index, and spatial pattern label.

        Raises:
            KeyError: If coordinate columns are not found.
            ValueError: If fewer than 3 valid coordinate pairs.
        """
        for col in (lat_column, lon_column):
            if col not in data_frame.columns:
                raise KeyError(f"Column '{col}' not found in DataFrame.")
        if max_points < self._MINIMUM_POINTS:
            raise ValueError(f"max_points must be >= {self._MINIMUM_POINTS}. Got {max_points}.")

        clean = data_frame[[lat_column, lon_column]].dropna()

        if len(clean) < self._MINIMUM_POINTS:
            raise ValueError(
                f"At least {self._MINIMUM_POINTS} coordinate pairs required. "
                f"Got {len(clean)}."
            )

        if len(clean) > max_points:
            clean = clean.sample(n=max_points, random_state=42)
            sampled = True
        else:
            sampled = False

        lat = clean[lat_column].to_numpy(dtype=float)
        lon = clean[lon_column].to_numpy(dtype=float)

        nn_results = self._nn_finder.find_all(lat, lon)
        nn_distances = np.array([r.distance_km for r in nn_results])
        ann_result = self._ann_calculator.calculate(nn_distances, lat, lon)

        return {
            "nearest_neighbor_stats": {
                "mean_nn_distance_km": round(float(nn_distances.mean()), 4),
                "std_nn_distance_km": round(float(nn_distances.std()), 4),
                "min_nn_distance_km": round(float(nn_distances.min()), 4),
                "max_nn_distance_km": round(float(nn_distances.max()), 4),
                "median_nn_distance_km": round(float(np.median(nn_distances)), 4),
                "p5_nn_distance_km": round(float(np.percentile(nn_distances, 5)), 4),
                "p95_nn_distance_km": round(float(np.percentile(nn_distances, 95)), 4),
            },
            "spatial_pattern": ann_result,
            "nearest_neighbors": [
                {
                    "point_index": r.point_index,
                    "nearest_neighbor_index": r.nearest_neighbor_index,
                    "distance_km": r.distance_km,
                }
                for r in nn_results
            ] if include_all_nn else None,
            "n_points": len(clean),
            "sampled": sampled,
            "max_points": max_points,
        }
