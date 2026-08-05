"""Polars-native backend implementations for the geospatial statistics domain."""
from __future__ import annotations

from typing import Any
import math
import polars as pl
import numpy as np

from lumen.statistics.geospatial.abstract.geo_bounding_box import AbstractGeoBoundingBoxCalculator
from lumen.statistics.geospatial.abstract.geo_clustering import AbstractGeoClusteringCalculator
from lumen.statistics.geospatial.abstract.geo_distribution import AbstractGeoDistributionCalculator
from lumen.statistics.geospatial.abstract.geo_heatmap import AbstractGeoHeatmapCalculator
from lumen.statistics.geospatial.abstract.proximity_analysis import AbstractProximityAnalysisCalculator

def _eager(data: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    return data.collect() if isinstance(data, pl.LazyFrame) else data


# Haversine distance expression (returns km)
def _haversine_expr(lat1: float, lon1: float, lat2_col: str, lon2_col: str) -> pl.Expr:
    R = 6371.0 # Earth radius in km
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    
    lat2_rad = pl.col(lat2_col) * (math.pi / 180.0)
    lon2_rad = pl.col(lon2_col) * (math.pi / 180.0)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = (dlat / 2.0).sin()**2 + math.cos(lat1_rad) * lat2_rad.cos() * (dlon / 2.0).sin()**2
    # Use clip for safety
    a = a.clip(0.0, 1.0)
    c = 2.0 * a.sqrt().arcsin()
    return R * c


class GeoBoundingBoxCalculatorPolars(AbstractGeoBoundingBoxCalculator):
    def calculate(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        lat_column: str,
        lon_column: str,
    ) -> dict[str, Any]:
        frame = _eager(data).select([lat_column, lon_column]).drop_nulls()
        
        res = frame.select([
            pl.col(lat_column).min().alias("min_lat"),
            pl.col(lat_column).max().alias("max_lat"),
            pl.col(lon_column).min().alias("min_lon"),
            pl.col(lon_column).max().alias("max_lon"),
            pl.col(lat_column).mean().alias("mean_lat"),
            pl.col(lon_column).mean().alias("mean_lon")
        ]).row(0)

        min_lat, max_lat, min_lon, max_lon, mean_lat, mean_lon = res
        
        return {
            "bounding_box": {
                "min_latitude": float(min_lat),
                "max_latitude": float(max_lat),
                "min_longitude": float(min_lon),
                "max_longitude": float(max_lon),
            },
            "centroid": {
                "latitude": float(mean_lat),
                "longitude": float(mean_lon)
            }
        }


class GeoClusteringCalculatorPolars(AbstractGeoClusteringCalculator):
    def calculate(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        lat_column: str,
        lon_column: str,
        eps_km: float = 1.0,
        min_samples: int = 5,
    ) -> dict[str, Any]:
        from sklearn.cluster import DBSCAN
        
        frame = _eager(data).select([lat_column, lon_column]).drop_nulls()
        coords = frame.to_numpy()
        
        if coords.shape[0] == 0:
            return {"clusters": {}, "n_clusters": 0, "n_noise": 0}

        # sklearn DBSCAN takes radians for haversine
        rad_coords = np.radians(coords)
        eps_rad = eps_km / 6371.0
        
        db = DBSCAN(eps=eps_rad, min_samples=min_samples, algorithm='ball_tree', metric='haversine')
        labels = db.fit_predict(rad_coords)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        unique_labels = set(labels)
        clusters = {}
        for k in unique_labels:
            if k == -1: continue
            mask = (labels == k)
            c_coords = coords[mask]
            clusters[str(k)] = {
                "size": int(np.sum(mask)),
                "centroid": {
                    "latitude": float(c_coords[:, 0].mean()),
                    "longitude": float(c_coords[:, 1].mean()),
                }
            }

        return {
            "clusters": clusters,
            "n_clusters": n_clusters,
            "n_noise": n_noise,
        }


class GeoDistributionCalculatorPolars(AbstractGeoDistributionCalculator):
    def calculate(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        lat_column: str,
        lon_column: str,
        weight_column: str | None = None,
    ) -> dict[str, Any]:
        cols = [lat_column, lon_column]
        if weight_column:
            cols.append(weight_column)
            
        frame = _eager(data).select(cols).drop_nulls()
        
        if weight_column:
            total_weight = frame[weight_column].sum()
            if total_weight == 0: total_weight = 1.0
            
            mean_lat = float((frame[lat_column] * frame[weight_column]).sum() / total_weight)
            mean_lon = float((frame[lon_column] * frame[weight_column]).sum() / total_weight)
            
            var_lat = float(((frame[lat_column] - mean_lat)**2 * frame[weight_column]).sum() / total_weight)
            var_lon = float(((frame[lon_column] - mean_lon)**2 * frame[weight_column]).sum() / total_weight)
        else:
            mean_lat = float(frame[lat_column].mean())
            mean_lon = float(frame[lon_column].mean())
            var_lat = float(frame[lat_column].var(ddof=0))
            var_lon = float(frame[lon_column].var(ddof=0))

        # Standard distance is roughly root of sum of variances
        standard_distance = math.sqrt(var_lat + var_lon)
        
        return {
            "center_of_mass": {"latitude": mean_lat, "longitude": mean_lon},
            "spatial_variance": {"latitude": var_lat, "longitude": var_lon},
            "standard_distance": standard_distance,
        }


class GeoHeatmapCalculatorPolars(AbstractGeoHeatmapCalculator):
    def calculate(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        lat_column: str,
        lon_column: str,
        weight_column: str | None = None,
        grid_size_lat: int = 50,
        grid_size_lon: int = 50,
    ) -> dict[str, Any]:
        cols = [lat_column, lon_column]
        if weight_column: cols.append(weight_column)
        
        frame = _eager(data).select(cols).drop_nulls()
        
        if frame.height == 0:
            return {"grid": [], "resolution": {"lat": grid_size_lat, "lon": grid_size_lon}}

        min_lat = float(frame[lat_column].min())
        max_lat = float(frame[lat_column].max())
        min_lon = float(frame[lon_column].min())
        max_lon = float(frame[lon_column].max())
        
        # Avoid zero division if points are identical
        if min_lat == max_lat: max_lat += 0.0001
        if min_lon == max_lon: max_lon += 0.0001

        step_lat = (max_lat - min_lat) / grid_size_lat
        step_lon = (max_lon - min_lon) / grid_size_lon

        # Assign indices
        frame = frame.with_columns([
            (((pl.col(lat_column) - min_lat) / step_lat).floor().cast(pl.Int32)).clip(0, grid_size_lat - 1).alias("lat_idx"),
            (((pl.col(lon_column) - min_lon) / step_lon).floor().cast(pl.Int32)).clip(0, grid_size_lon - 1).alias("lon_idx")
        ])

        if weight_column:
            agg_expr = pl.col(weight_column).sum().alias("intensity")
        else:
            agg_expr = pl.len().alias("intensity")

        grid = frame.group_by(["lat_idx", "lon_idx"]).agg(agg_expr).to_dicts()

        return {
            "grid": grid,
            "bounding_box": {
                "min_lat": min_lat, "max_lat": max_lat,
                "min_lon": min_lon, "max_lon": max_lon
            },
            "resolution": {"lat": grid_size_lat, "lon": grid_size_lon},
            "step_size": {"lat": step_lat, "lon": step_lon}
        }


class ProximityAnalysisCalculatorPolars(AbstractProximityAnalysisCalculator):
    def calculate(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        lat_column: str,
        lon_column: str,
        reference_lat: float,
        reference_lon: float,
        max_distance_km: float | None = None,
    ) -> dict[str, Any]:
        frame = _eager(data).select([lat_column, lon_column]).drop_nulls()
        
        frame = frame.with_columns([
            _haversine_expr(reference_lat, reference_lon, lat_column, lon_column).alias("distance_km")
        ])

        if max_distance_km is not None:
            frame = frame.filter(pl.col("distance_km") <= max_distance_km)

        if frame.height == 0:
            return {
                "reference_point": {"latitude": reference_lat, "longitude": reference_lon},
                "n_within_radius": 0,
                "min_distance_km": None,
                "mean_distance_km": None,
                "max_distance_km": None,
            }

        res = frame.select([
            pl.col("distance_km").min().alias("min_d"),
            pl.col("distance_km").mean().alias("mean_d"),
            pl.col("distance_km").max().alias("max_d"),
            pl.len().alias("n")
        ]).row(0)

        return {
            "reference_point": {"latitude": reference_lat, "longitude": reference_lon},
            "n_within_radius": int(res[3]),
            "min_distance_km": float(res[0]),
            "mean_distance_km": float(res[1]),
            "max_distance_km": float(res[2]),
        }
