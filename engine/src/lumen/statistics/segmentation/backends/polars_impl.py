"""Polars-native backend implementations for the segmentation statistics domain."""
from __future__ import annotations

from typing import Any
import math
import numpy as np
import polars as pl
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans

from segmentation.abstract.cohort_analysis import AbstractCohortAnalysisCalculator
from segmentation.abstract.dbscan_clusters import AbstractDBSCANClustersCalculator
from segmentation.abstract.hierarchical_clusters import AbstractHierarchicalClustersCalculator
from segmentation.abstract.kmeans_clusters import AbstractKMeansClustersCalculator
from segmentation.abstract.population_splits import AbstractPopulationSplitsCalculator
from segmentation.abstract.rfm_segmentation import AbstractRFMSegmentationCalculator


def _eager(data: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    return data.collect() if isinstance(data, pl.LazyFrame) else data


class CohortAnalysisCalculatorPolars(AbstractCohortAnalysisCalculator):
    def calculate(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        user_column: str,
        date_column: str,
        period: str = "month",
    ) -> dict[str, Any]:
        frame = _eager(data).select([user_column, date_column]).drop_nulls()
        
        # Cast to Date
        frame = frame.with_columns(pl.col(date_column).cast(pl.Date))
        
        if period == "month":
            trunc_expr = pl.col(date_column).dt.truncate("1mo")
        elif period == "week":
            trunc_expr = pl.col(date_column).dt.truncate("1w")
        elif period == "day":
            trunc_expr = pl.col(date_column).dt.truncate("1d")
        elif period == "year":
            trunc_expr = pl.col(date_column).dt.truncate("1y")
        else:
            raise ValueError(f"Unsupported period {period}")

        frame = frame.with_columns(trunc_expr.alias("event_period"))
        
        cohort_df = frame.group_by(user_column).agg(pl.col("event_period").min().alias("cohort_period"))
        
        frame = frame.join(cohort_df, on=user_column)
        
        # Calculate cohort age
        if period == "month":
            diff_expr = (pl.col("event_period").dt.year() - pl.col("cohort_period").dt.year()) * 12 + \
                        (pl.col("event_period").dt.month() - pl.col("cohort_period").dt.month())
        elif period == "week":
            diff_expr = (pl.col("event_period") - pl.col("cohort_period")).dt.total_days() // 7
        elif period == "day":
            diff_expr = (pl.col("event_period") - pl.col("cohort_period")).dt.total_days()
        elif period == "year":
            diff_expr = pl.col("event_period").dt.year() - pl.col("cohort_period").dt.year()
            
        frame = frame.with_columns(diff_expr.cast(pl.Int32).alias("cohort_age"))
        
        cohort_counts = frame.group_by(["cohort_period", "cohort_age"]).agg(
            pl.col(user_column).n_unique().alias("users")
        ).sort(["cohort_period", "cohort_age"])

        matrix = cohort_counts.pivot(
            values="users", index="cohort_period", columns="cohort_age", aggregate_function="sum"
        ).fill_null(0).to_dicts()
        
        # Convert date to string
        for r in matrix:
            r["cohort_period"] = str(r["cohort_period"])

        return {"retention_matrix": matrix, "period": period}


class DBSCANClustersCalculatorPolars(AbstractDBSCANClustersCalculator):
    def calculate(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        features: list[str],
        eps: float = 0.5,
        min_samples: int = 5,
    ) -> dict[str, Any]:
        frame = _eager(data).select(features).drop_nulls()
        X = frame.to_numpy()
        
        if len(X) == 0:
            return {"clusters": {}, "n_clusters": 0, "n_noise": 0}

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


class HierarchicalClustersCalculatorPolars(AbstractHierarchicalClustersCalculator):
    def calculate(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        features: list[str],
        n_clusters: int = 3,
        linkage: str = "ward",
    ) -> dict[str, Any]:
        frame = _eager(data).select(features).drop_nulls()
        X = frame.to_numpy()
        
        if len(X) < n_clusters:
            return {"clusters": {}, "n_clusters": 0, "features": features}

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


class KMeansClustersCalculatorPolars(AbstractKMeansClustersCalculator):
    def calculate(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        features: list[str],
        n_clusters: int = 3,
        random_state: int = 42,
    ) -> dict[str, Any]:
        frame = _eager(data).select(features).drop_nulls()
        X = frame.to_numpy()
        
        if len(X) < n_clusters:
            return {"clusters": {}, "n_clusters": 0, "features": features}

        km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
        labels = km.fit_predict(X)
        
        clusters = {}
        for k in range(n_clusters):
            mask = (labels == k)
            clusters[str(k)] = {
                "size": int(np.sum(mask)),
                "centroid": km.cluster_centers_[k].tolist(),
            }

        return {
            "clusters": clusters,
            "n_clusters": n_clusters,
            "inertia": float(km.inertia_),
            "features": features,
        }


class PopulationSplitsCalculatorPolars(AbstractPopulationSplitsCalculator):
    def calculate(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        column: str,
        method: str = "quantiles",
        n_bins: int = 4,
    ) -> dict[str, Any]:
        frame = _eager(data).select(pl.col(column).drop_nulls())
        
        if method == "quantiles":
            qs = np.linspace(0, 1, n_bins + 1)
            edges = [float(frame[column].quantile(q)) for q in qs]
            edges[0] -= 1e-6 # Include minimum
        elif method == "equal_width":
            vmin = float(frame[column].min())
            vmax = float(frame[column].max())
            edges = np.linspace(vmin, vmax, n_bins + 1).tolist()
        else:
            raise ValueError(f"Unknown method {method}")
            
        counts = []
        for i in range(n_bins):
            lower = edges[i]
            upper = edges[i+1]
            if i == n_bins - 1:
                cnt = frame.filter((pl.col(column) > lower) & (pl.col(column) <= upper)).height
            else:
                cnt = frame.filter((pl.col(column) > lower) & (pl.col(column) <= upper)).height
            counts.append(cnt)
            
        return {
            "method": method,
            "n_bins": n_bins,
            "bin_edges": edges,
            "bin_counts": counts,
        }


class RFMSegmentationCalculatorPolars(AbstractRFMSegmentationCalculator):
    def calculate(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        customer_column: str,
        date_column: str,
        amount_column: str,
        reference_date: str | None = None,
    ) -> dict[str, Any]:
        frame = _eager(data).select([customer_column, date_column, amount_column]).drop_nulls()
        frame = frame.with_columns(pl.col(date_column).cast(pl.Date))
        
        if reference_date:
            ref_dt = pl.lit(reference_date).cast(pl.Date)
        else:
            ref_dt = frame[date_column].max()
            
        rfm_df = frame.group_by(customer_column).agg([
            (ref_dt - pl.col(date_column).max()).dt.total_days().alias("recency"),
            pl.len().alias("frequency"),
            pl.col(amount_column).sum().alias("monetary")
        ])

        # Quintiles
        def get_quintile_expr(col_name: str, desc: bool) -> pl.Expr:
            # Polars rank is sufficient
            rank_col = pl.col(col_name).rank("average", descending=desc)
            n_obs = pl.len()
            return ((rank_col - 1) / n_obs * 5).floor().cast(pl.Int32) + 1

        rfm_df = rfm_df.with_columns([
            get_quintile_expr("recency", False).alias("r_score"),
            get_quintile_expr("frequency", True).alias("f_score"),
            get_quintile_expr("monetary", True).alias("m_score"),
        ]).with_columns(
            (pl.col("r_score").cast(pl.Utf8) + pl.col("f_score").cast(pl.Utf8) + pl.col("m_score").cast(pl.Utf8)).alias("rfm_segment")
        )
        
        # We can map some common segments
        def segment_name(r, f, m):
            if r == 5 and f == 5 and m == 5: return "Champions"
            if f == 5: return "Loyal"
            if r == 5: return "Recent"
            if r == 1 and f == 1: return "Lost"
            return "Average"

        records = rfm_df.to_dicts()
        for rec in records:
            rec["segment_name"] = segment_name(rec["r_score"], rec["f_score"], rec["m_score"])

        summary = rfm_df.select([
            pl.col("recency").mean().alias("mean_r"),
            pl.col("frequency").mean().alias("mean_f"),
            pl.col("monetary").mean().alias("mean_m")
        ]).row(0)

        return {
            "customers": records,
            "summary": {
                "mean_recency": float(summary[0]),
                "mean_frequency": float(summary[1]),
                "mean_monetary": float(summary[2]),
                "n_customers": len(records),
            }
        }
