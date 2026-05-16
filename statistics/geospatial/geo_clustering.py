"""Geographic DBSCAN clustering on latitude/longitude coordinates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN


@dataclass(frozen=True)
class GeoCluster:
    """Immutable geographic cluster record."""

    cluster_id: int
    n_points: int
    proportion: float
    centroid_lat: float
    centroid_lon: float
    bbox_lat_min: float
    bbox_lat_max: float
    bbox_lon_min: float
    bbox_lon_max: float
    is_noise: bool


class HaversineDistanceMatrix:
    """Computes pairwise Haversine distances between coordinate pairs.

    Haversine distance accounts for Earth's curvature — essential for
    geographic clustering where Euclidean distance introduces significant
    error for distances > ~100km.

    Haversine(lat1,lon1,lat2,lon2) = 2R × arcsin(√(sin²(Δlat/2)
        + cos(lat1)×cos(lat2)×sin²(Δlon/2)))

    where R = 6371 km (Earth's mean radius).
    """

    _EARTH_RADIUS_KM: float = 6371.0

    def compute(self, coords: np.ndarray) -> np.ndarray:
        """Compute pairwise Haversine distance matrix.

        Args:
            coords: Array of shape (n, 2) with [lat, lon] in degrees.

        Returns:
            Symmetric distance matrix of shape (n, n) in kilometers.
        """
        lat = np.radians(coords[:, 0])
        lon = np.radians(coords[:, 1])

        n = len(coords)
        dist_matrix = np.zeros((n, n))

        for i in range(n):
            dlat = lat - lat[i]
            dlon = lon - lon[i]
            a = (
                np.sin(dlat / 2) ** 2
                + np.cos(lat[i]) * np.cos(lat) * np.sin(dlon / 2) ** 2
            )
            dist_matrix[i] = 2 * self._EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

        return dist_matrix


class GeoClusterProfileBuilder:
    """Builds geographic cluster profiles from labels and coordinate data."""

    def build(
        self,
        coords: np.ndarray,
        labels: np.ndarray,
    ) -> list[GeoCluster]:
        """Build profile for each detected cluster including noise group.

        Args:
            coords: Array of shape (n, 2) with [lat, lon].
            labels: DBSCAN label array (-1 = noise).

        Returns:
            List of GeoCluster objects.
        """
        n_total = len(labels)
        unique_labels = sorted(set(labels))
        profiles: list[GeoCluster] = []

        for label in unique_labels:
            mask = labels == label
            cluster_coords = coords[mask]
            n_points = int(mask.sum())

            profiles.append(
                GeoCluster(
                    cluster_id=int(label),
                    n_points=n_points,
                    proportion=round(n_points / n_total, 4),
                    centroid_lat=round(float(cluster_coords[:, 0].mean()), 6),
                    centroid_lon=round(float(cluster_coords[:, 1].mean()), 6),
                    bbox_lat_min=round(float(cluster_coords[:, 0].min()), 6),
                    bbox_lat_max=round(float(cluster_coords[:, 0].max()), 6),
                    bbox_lon_min=round(float(cluster_coords[:, 1].min()), 6),
                    bbox_lon_max=round(float(cluster_coords[:, 1].max()), 6),
                    is_noise=label == -1,
                )
            )

        return profiles


class GeoClusteringCalculator:
    """Haversine-DBSCAN geographic point clustering.

    Uses the Haversine metric for DBSCAN so epsilon is specified in
    kilometers rather than degrees — making the threshold directly
    interpretable.

    Workflow:
        calculator = GeoClusteringCalculator()
        result = calculator.calculate(
            data_frame=df,
            lat_column="latitude",
            lon_column="longitude",
            epsilon_km=5.0,     # neighborhood radius in kilometers
            min_samples=5,
        )
    """

    _MINIMUM_POINTS: int = 10

    def __init__(self) -> None:
        self._distance_computer = HaversineDistanceMatrix()
        self._profile_builder = GeoClusterProfileBuilder()

    def calculate(
        self,
        data_frame: pd.DataFrame,
        lat_column: str,
        lon_column: str,
        epsilon_km: float = 5.0,
        min_samples: int = 5,
    ) -> dict:
        """Run Haversine-DBSCAN geographic clustering.

        Args:
            data_frame: Source DataFrame with coordinate columns.
            lat_column: Latitude column name (decimal degrees).
            lon_column: Longitude column name (decimal degrees).
            epsilon_km: Neighborhood radius in kilometers.
            min_samples: Minimum points to form a core point.

        Returns:
            Dict with cluster profiles, noise stats, and geographic summary.

        Raises:
            KeyError: If coordinate columns are not found.
            ValueError: If parameters are invalid or data is insufficient.
        """
        for col in (lat_column, lon_column):
            if col not in data_frame.columns:
                raise KeyError(f"Column '{col}' not found in DataFrame.")
        if epsilon_km <= 0:
            raise ValueError(f"epsilon_km must be > 0. Got {epsilon_km}.")
        if min_samples < 2:
            raise ValueError(f"min_samples must be >= 2. Got {min_samples}.")

        clean = data_frame[[lat_column, lon_column]].dropna()

        if len(clean) < self._MINIMUM_POINTS:
            raise ValueError(
                f"At least {self._MINIMUM_POINTS} coordinate pairs required. "
                f"Got {len(clean)}."
            )

        lat_vals = clean[lat_column].to_numpy(dtype=float)
        lon_vals = clean[lon_column].to_numpy(dtype=float)

        invalid_lat = (lat_vals < -90) | (lat_vals > 90)
        invalid_lon = (lon_vals < -180) | (lon_vals > 180)
        if invalid_lat.any() or invalid_lon.any():
            raise ValueError(
                "Coordinate values out of range. "
                "Latitude must be in [-90, 90], longitude in [-180, 180]."
            )

        coords = np.column_stack([lat_vals, lon_vals])
        dist_matrix = self._distance_computer.compute(coords)

        model = DBSCAN(
            eps=epsilon_km,
            min_samples=min_samples,
            metric="precomputed",
        )
        labels = model.fit_predict(dist_matrix)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int(np.sum(labels == -1))
        profiles = self._profile_builder.build(coords, labels)

        non_noise_profiles = [p for p in profiles if not p.is_noise]
        largest_cluster = max(non_noise_profiles, key=lambda p: p.n_points) if non_noise_profiles else None

        return {
            "labels": labels.tolist(),
            "n_clusters": n_clusters,
            "n_noise_points": n_noise,
            "noise_ratio": round(n_noise / len(labels), 4),
            "cluster_profiles": [
                {
                    "cluster_id": p.cluster_id,
                    "n_points": p.n_points,
                    "proportion": p.proportion,
                    "centroid": {
                        "lat": p.centroid_lat,
                        "lon": p.centroid_lon,
                    },
                    "bounding_box": {
                        "lat_min": p.bbox_lat_min,
                        "lat_max": p.bbox_lat_max,
                        "lon_min": p.bbox_lon_min,
                        "lon_max": p.bbox_lon_max,
                    },
                    "is_noise": p.is_noise,
                }
                for p in profiles
            ],
            "largest_cluster": {
                "cluster_id": largest_cluster.cluster_id,
                "n_points": largest_cluster.n_points,
                "centroid_lat": largest_cluster.centroid_lat,
                "centroid_lon": largest_cluster.centroid_lon,
            } if largest_cluster else None,
            "parameters": {
                "epsilon_km": epsilon_km,
                "min_samples": min_samples,
            },
            "n_observations": len(clean),
        }
