"""PySpark-native backend implementations for the geospatial statistics domain."""
from __future__ import annotations

from typing import Any
import math
import numpy as np

from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F

from lumen.statistics.geospatial.abstract.geo_bounding_box import AbstractGeoBoundingBoxCalculator
from lumen.statistics.geospatial.abstract.geo_clustering import AbstractGeoClusteringCalculator
from lumen.statistics.geospatial.abstract.geo_distribution import AbstractGeoDistributionCalculator
from lumen.statistics.geospatial.abstract.geo_heatmap import AbstractGeoHeatmapCalculator
from lumen.statistics.geospatial.abstract.proximity_analysis import AbstractProximityAnalysisCalculator

def _haversine_expr(lat1: float, lon1: float, lat2_col: str, lon2_col: str) -> Any:
    R = 6371.0 # Earth radius in km
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    
    lat2_rad = F.radians(F.col(lat2_col))
    lon2_rad = F.radians(F.col(lon2_col))
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = F.sin(dlat / 2.0)**2 + math.cos(lat1_rad) * F.cos(lat2_rad) * F.sin(dlon / 2.0)**2
    # Spark doesn't have a direct clip, so we use when/otherwise or just assume a <= 1.0
    a = F.when(a > 1.0, 1.0).otherwise(F.when(a < 0.0, 0.0).otherwise(a))
    c = 2.0 * F.asin(F.sqrt(a))
    return R * c


class GeoBoundingBoxCalculatorSpark(AbstractGeoBoundingBoxCalculator):
    def calculate(
        self,
        data: SparkDataFrame,
        lat_column: str,
        lon_column: str,
    ) -> dict[str, Any]:
        clean = data.select(lat_column, lon_column).dropna()
        
        res = clean.agg(
            F.min(lat_column).alias("min_lat"),
            F.max(lat_column).alias("max_lat"),
            F.min(lon_column).alias("min_lon"),
            F.max(lon_column).alias("max_lon"),
            F.mean(lat_column).alias("mean_lat"),
            F.mean(lon_column).alias("mean_lon")
        ).collect()[0]

        return {
            "bounding_box": {
                "min_latitude": float(res["min_lat"] or 0),
                "max_latitude": float(res["max_lat"] or 0),
                "min_longitude": float(res["min_lon"] or 0),
                "max_longitude": float(res["max_lon"] or 0),
            },
            "centroid": {
                "latitude": float(res["mean_lat"] or 0),
                "longitude": float(res["mean_lon"] or 0)
            }
        }


class GeoClusteringCalculatorSpark(AbstractGeoClusteringCalculator):
    def calculate(
        self,
        data: SparkDataFrame,
        lat_column: str,
        lon_column: str,
        eps_km: float = 1.0,
        min_samples: int = 5,
    ) -> dict[str, Any]:
        from sklearn.cluster import DBSCAN
        
        # Scikit-learn DBSCAN is used here as Spark ML doesn't have native DBSCAN
        # We collect the data since DBSCAN is not easily distributed without custom libraries.
        clean = data.select(lat_column, lon_column).dropna()
        rows = clean.collect()
        
        if not rows:
            return {"clusters": {}, "n_clusters": 0, "n_noise": 0}

        coords = np.array([[r[0], r[1]] for r in rows], dtype=float)
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


class GeoDistributionCalculatorSpark(AbstractGeoDistributionCalculator):
    def calculate(
        self,
        data: SparkDataFrame,
        lat_column: str,
        lon_column: str,
        weight_column: str | None = None,
    ) -> dict[str, Any]:
        
        if weight_column:
            clean = data.select(lat_column, lon_column, weight_column).dropna()
            total_weight = float(clean.agg(F.sum(weight_column)).collect()[0][0] or 1.0)
            if total_weight == 0: total_weight = 1.0
            
            mean_df = clean.agg(
                (F.sum(F.col(lat_column) * F.col(weight_column)) / total_weight).alias("mean_lat"),
                (F.sum(F.col(lon_column) * F.col(weight_column)) / total_weight).alias("mean_lon")
            ).collect()[0]
            
            mean_lat = float(mean_df["mean_lat"])
            mean_lon = float(mean_df["mean_lon"])
            
            var_df = clean.agg(
                (F.sum((F.col(lat_column) - mean_lat)**2 * F.col(weight_column)) / total_weight).alias("var_lat"),
                (F.sum((F.col(lon_column) - mean_lon)**2 * F.col(weight_column)) / total_weight).alias("var_lon")
            ).collect()[0]
            
            var_lat = float(var_df["var_lat"])
            var_lon = float(var_df["var_lon"])
        else:
            clean = data.select(lat_column, lon_column).dropna()
            agg = clean.agg(
                F.mean(lat_column).alias("mean_lat"),
                F.mean(lon_column).alias("mean_lon"),
                F.var_pop(lat_column).alias("var_lat"),
                F.var_pop(lon_column).alias("var_lon")
            ).collect()[0]
            
            mean_lat = float(agg["mean_lat"] or 0.0)
            mean_lon = float(agg["mean_lon"] or 0.0)
            var_lat = float(agg["var_lat"] or 0.0)
            var_lon = float(agg["var_lon"] or 0.0)

        standard_distance = math.sqrt(var_lat + var_lon)
        
        return {
            "center_of_mass": {"latitude": mean_lat, "longitude": mean_lon},
            "spatial_variance": {"latitude": var_lat, "longitude": var_lon},
            "standard_distance": standard_distance,
        }


class GeoHeatmapCalculatorSpark(AbstractGeoHeatmapCalculator):
    def calculate(
        self,
        data: SparkDataFrame,
        lat_column: str,
        lon_column: str,
        weight_column: str | None = None,
        grid_size_lat: int = 50,
        grid_size_lon: int = 50,
    ) -> dict[str, Any]:
        cols = [lat_column, lon_column]
        if weight_column: cols.append(weight_column)
        
        clean = data.select(*cols).dropna()
        if clean.count() == 0:
            return {"grid": [], "resolution": {"lat": grid_size_lat, "lon": grid_size_lon}}

        bounds = clean.agg(
            F.min(lat_column).alias("min_lat"),
            F.max(lat_column).alias("max_lat"),
            F.min(lon_column).alias("min_lon"),
            F.max(lon_column).alias("max_lon")
        ).collect()[0]
        
        min_lat, max_lat = float(bounds["min_lat"]), float(bounds["max_lat"])
        min_lon, max_lon = float(bounds["min_lon"]), float(bounds["max_lon"])

        if min_lat == max_lat: max_lat += 0.0001
        if min_lon == max_lon: max_lon += 0.0001

        step_lat = (max_lat - min_lat) / grid_size_lat
        step_lon = (max_lon - min_lon) / grid_size_lon

        df = clean.withColumn("lat_idx", F.floor((F.col(lat_column) - min_lat) / step_lat).cast("int"))
        df = df.withColumn("lon_idx", F.floor((F.col(lon_column) - min_lon) / step_lon).cast("int"))
        
        # Clip indices
        df = df.withColumn("lat_idx", F.when(F.col("lat_idx") >= grid_size_lat, grid_size_lat - 1).otherwise(F.when(F.col("lat_idx") < 0, 0).otherwise(F.col("lat_idx"))))
        df = df.withColumn("lon_idx", F.when(F.col("lon_idx") >= grid_size_lon, grid_size_lon - 1).otherwise(F.when(F.col("lon_idx") < 0, 0).otherwise(F.col("lon_idx"))))

        if weight_column:
            grid_df = df.groupBy("lat_idx", "lon_idx").agg(F.sum(weight_column).alias("intensity"))
        else:
            grid_df = df.groupBy("lat_idx", "lon_idx").agg(F.count("*").alias("intensity"))

        grid_rows = grid_df.collect()
        grid = [{"lat_idx": r["lat_idx"], "lon_idx": r["lon_idx"], "intensity": float(r["intensity"])} for r in grid_rows]

        return {
            "grid": grid,
            "bounding_box": {
                "min_lat": min_lat, "max_lat": max_lat,
                "min_lon": min_lon, "max_lon": max_lon
            },
            "resolution": {"lat": grid_size_lat, "lon": grid_size_lon},
            "step_size": {"lat": step_lat, "lon": step_lon}
        }


class ProximityAnalysisCalculatorSpark(AbstractProximityAnalysisCalculator):
    def calculate(
        self,
        data: SparkDataFrame,
        lat_column: str,
        lon_column: str,
        reference_lat: float,
        reference_lon: float,
        max_distance_km: float | None = None,
    ) -> dict[str, Any]:
        
        clean = data.select(lat_column, lon_column).dropna()
        df = clean.withColumn("distance_km", _haversine_expr(reference_lat, reference_lon, lat_column, lon_column))

        if max_distance_km is not None:
            df = df.filter(F.col("distance_km") <= max_distance_km)

        count = df.count()
        if count == 0:
            return {
                "reference_point": {"latitude": reference_lat, "longitude": reference_lon},
                "n_within_radius": 0,
                "min_distance_km": None,
                "mean_distance_km": None,
                "max_distance_km": None,
            }

        res = df.agg(
            F.min("distance_km").alias("min_d"),
            F.mean("distance_km").alias("mean_d"),
            F.max("distance_km").alias("max_d")
        ).collect()[0]

        return {
            "reference_point": {"latitude": reference_lat, "longitude": reference_lon},
            "n_within_radius": count,
            "min_distance_km": float(res["min_d"]),
            "mean_distance_km": float(res["mean_d"]),
            "max_distance_km": float(res["max_d"]),
        }
