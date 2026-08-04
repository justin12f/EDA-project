"""PySpark-native backend implementations for the segmentation statistics domain."""
from __future__ import annotations

from typing import Any
import math
import numpy as np
from sklearn.cluster import DBSCAN, AgglomerativeClustering

from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler

from segmentation.abstract.cohort_analysis import AbstractCohortAnalysisCalculator
from segmentation.abstract.dbscan_clusters import AbstractDBSCANClustersCalculator
from segmentation.abstract.hierarchical_clusters import AbstractHierarchicalClustersCalculator
from segmentation.abstract.kmeans_clusters import AbstractKMeansClustersCalculator
from segmentation.abstract.population_splits import AbstractPopulationSplitsCalculator
from segmentation.abstract.rfm_segmentation import AbstractRFMSegmentationCalculator


class CohortAnalysisCalculatorSpark(AbstractCohortAnalysisCalculator):
    def calculate(
        self,
        data: SparkDataFrame,
        user_column: str,
        date_column: str,
        period: str = "month",
    ) -> dict[str, Any]:
        clean = data.select(user_column, date_column).dropna()
        
        if period == "month":
            trunc_expr = F.date_trunc("month", F.col(date_column))
        elif period == "week":
            trunc_expr = F.date_trunc("week", F.col(date_column))
        elif period == "day":
            trunc_expr = F.date_trunc("day", F.col(date_column))
        elif period == "year":
            trunc_expr = F.date_trunc("year", F.col(date_column))
        else:
            raise ValueError(f"Unsupported period {period}")

        df = clean.withColumn("event_period", F.to_date(trunc_expr))
        
        cohort_df = df.groupBy(user_column).agg(F.min("event_period").alias("cohort_period"))
        df = df.join(cohort_df, on=user_column)
        
        if period == "month":
            diff_expr = (F.year("event_period") - F.year("cohort_period")) * 12 + \
                        (F.month("event_period") - F.month("cohort_period"))
        elif period == "week":
            diff_expr = F.floor(F.datediff("event_period", "cohort_period") / 7)
        elif period == "day":
            diff_expr = F.datediff("event_period", "cohort_period")
        elif period == "year":
            diff_expr = F.year("event_period") - F.year("cohort_period")
            
        df = df.withColumn("cohort_age", diff_expr.cast("int"))
        
        cohort_counts = df.groupBy("cohort_period", "cohort_age").agg(F.countDistinct(user_column).alias("users"))
        
        rows = cohort_counts.collect()
        matrix = {}
        for r in rows:
            period_str = str(r["cohort_period"])
            age = r["cohort_age"]
            if period_str not in matrix:
                matrix[period_str] = {}
            matrix[period_str][str(age)] = int(r["users"])

        return {"retention_matrix": matrix, "period": period}


class DBSCANClustersCalculatorSpark(AbstractDBSCANClustersCalculator):
    def calculate(
        self,
        data: SparkDataFrame,
        features: list[str],
        eps: float = 0.5,
        min_samples: int = 5,
    ) -> dict[str, Any]:
        
        # Scikit-learn DBSCAN is used here as Spark ML doesn't have native DBSCAN
        clean = data.select(features).dropna()
        rows = clean.collect()
        
        if not rows:
            return {"clusters": {}, "n_clusters": 0, "n_noise": 0, "features": features}

        X = np.array([[r[f] for f in features] for r in rows], dtype=float)
        
        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(X)
        
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        clusters = {}
        for k in unique_labels:
            if k == -1: continue
            mask = (labels == k)
            c_coords = X[mask]
            clusters[str(k)] = {
                "size": int(np.sum(mask)),
                "centroid": c_coords.mean(axis=0).tolist(),
            }

        return {
            "clusters": clusters,
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "features": features,
        }


class HierarchicalClustersCalculatorSpark(AbstractHierarchicalClustersCalculator):
    def calculate(
        self,
        data: SparkDataFrame,
        features: list[str],
        n_clusters: int = 3,
        linkage: str = "ward",
    ) -> dict[str, Any]:
        
        clean = data.select(features).dropna()
        rows = clean.collect()
        
        if len(rows) < n_clusters:
            return {"clusters": {}, "n_clusters": 0, "features": features, "linkage": linkage}

        X = np.array([[r[f] for f in features] for r in rows], dtype=float)

        hc = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        labels = hc.fit_predict(X)
        
        unique_labels = set(labels)
        clusters = {}
        for k in unique_labels:
            mask = (labels == k)
            c_coords = X[mask]
            clusters[str(k)] = {
                "size": int(np.sum(mask)),
                "centroid": c_coords.mean(axis=0).tolist(),
            }

        return {
            "clusters": clusters,
            "n_clusters": n_clusters,
            "features": features,
            "linkage": linkage,
        }


class KMeansClustersCalculatorSpark(AbstractKMeansClustersCalculator):
    def calculate(
        self,
        data: SparkDataFrame,
        features: list[str],
        n_clusters: int = 3,
        random_state: int = 42,
    ) -> dict[str, Any]:
        clean = data.select(features).dropna()
        if clean.count() < n_clusters:
            return {"clusters": {}, "n_clusters": 0, "features": features}

        assembler = VectorAssembler(inputCols=features, outputCol="features_vec")
        df_vec = assembler.transform(clean)
        
        km = KMeans(featuresCol="features_vec", k=n_clusters, seed=random_state)
        model = km.fit(df_vec)
        
        predictions = model.transform(df_vec)
        
        centers = model.clusterCenters()
        counts = predictions.groupBy("prediction").count().collect()
        count_map = {r["prediction"]: r["count"] for r in counts}
        
        clusters = {}
        for i, center in enumerate(centers):
            clusters[str(i)] = {
                "size": int(count_map.get(i, 0)),
                "centroid": center.tolist()
            }
            
        inertia = model.summary.trainingCost

        return {
            "clusters": clusters,
            "n_clusters": n_clusters,
            "inertia": float(inertia),
            "features": features,
        }


class PopulationSplitsCalculatorSpark(AbstractPopulationSplitsCalculator):
    def calculate(
        self,
        data: SparkDataFrame,
        column: str,
        method: str = "quantiles",
        n_bins: int = 4,
    ) -> dict[str, Any]:
        clean = data.select(column).dropna()
        
        if method == "quantiles":
            qs = np.linspace(0, 1, n_bins + 1).tolist()
            edges = clean.approxQuantile(column, qs, 0.0)
            edges[0] -= 1e-6 # Include minimum
        elif method == "equal_width":
            agg = clean.agg(F.min(column).alias("min"), F.max(column).alias("max")).collect()[0]
            vmin, vmax = float(agg["min"]), float(agg["max"])
            edges = np.linspace(vmin, vmax, n_bins + 1).tolist()
        else:
            raise ValueError(f"Unknown method {method}")
            
        counts = []
        for i in range(n_bins):
            lower = edges[i]
            upper = edges[i+1]
            if i == n_bins - 1:
                cnt = clean.filter((F.col(column) > lower) & (F.col(column) <= upper)).count()
            else:
                cnt = clean.filter((F.col(column) > lower) & (F.col(column) <= upper)).count()
            counts.append(cnt)
            
        return {
            "method": method,
            "n_bins": n_bins,
            "bin_edges": edges,
            "bin_counts": counts,
        }


class RFMSegmentationCalculatorSpark(AbstractRFMSegmentationCalculator):
    def calculate(
        self,
        data: SparkDataFrame,
        customer_column: str,
        date_column: str,
        amount_column: str,
        reference_date: str | None = None,
    ) -> dict[str, Any]:
        clean = data.select(customer_column, date_column, amount_column).dropna()
        clean = clean.withColumn(date_column, F.to_date(F.col(date_column)))
        
        if reference_date:
            ref_dt = F.to_date(F.lit(reference_date))
        else:
            ref_dt = F.max(date_column).over(Window.partitionBy(F.lit(1)))
            
        rfm_df = clean.groupBy(customer_column).agg(
            F.max(date_column).alias("max_date"),
            F.count("*").alias("frequency"),
            F.sum(amount_column).alias("monetary")
        )
        
        if not reference_date:
            max_dt_val = clean.agg(F.max(date_column)).collect()[0][0]
            rfm_df = rfm_df.withColumn("recency", F.datediff(F.lit(max_dt_val), F.col("max_date")))
        else:
            rfm_df = rfm_df.withColumn("recency", F.datediff(ref_dt, F.col("max_date")))

        # Quintiles via percent_rank
        window_r = Window.orderBy(F.desc("recency"))
        window_f = Window.orderBy("frequency")
        window_m = Window.orderBy("monetary")
        
        def to_quintile(col_name: str, window: Any) -> Any:
            return F.floor(F.percent_rank().over(window) * 5).cast("int") + 1
            
        rfm_df = rfm_df.withColumn("r_score", to_quintile("recency", window_r))
        rfm_df = rfm_df.withColumn("f_score", to_quintile("frequency", window_f))
        rfm_df = rfm_df.withColumn("m_score", to_quintile("monetary", window_m))
        
        def segment_name(r, f, m):
            if r == 5 and f == 5 and m == 5: return "Champions"
            if f == 5: return "Loyal"
            if r == 5: return "Recent"
            if r == 1 and f == 1: return "Lost"
            return "Average"

        rows = rfm_df.collect()
        records = []
        for r in rows:
            seg = segment_name(r["r_score"], r["f_score"], r["m_score"])
            records.append({
                "entity_id": r[customer_column],
                "recency": float(r["recency"]),
                "frequency": float(r["frequency"]),
                "monetary": float(r["monetary"]),
                "r_score": r["r_score"],
                "f_score": r["f_score"],
                "m_score": r["m_score"],
                "segment_name": seg
            })

        summary = rfm_df.agg(
            F.mean("recency").alias("mean_r"),
            F.mean("frequency").alias("mean_f"),
            F.mean("monetary").alias("mean_m")
        ).collect()[0]

        return {
            "customers": records,
            "summary": {
                "mean_recency": float(summary["mean_r"]),
                "mean_frequency": float(summary["mean_f"]),
                "mean_monetary": float(summary["mean_m"]),
                "n_customers": len(records),
            }
        }
