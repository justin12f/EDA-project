"""DBSCAN density-based clustering with noise detection and cluster profiling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class DBSCANClusterProfile:
    """Immutable profile for a single DBSCAN cluster."""

    cluster_id: int
    n_members: int
    proportion: float
    feature_means: dict[str, float]
    feature_stds: dict[str, float]
    is_noise: bool


class EpsilonEstimator:
    """Estimates optimal epsilon using the k-distance elbow heuristic."""

    def estimate(self, x: np.ndarray, min_samples: int) -> float:
        """Estimate epsilon from k-distance curve elbow."""
        nbrs = NearestNeighbors(n_neighbors=min_samples).fit(x)
        distances, _ = nbrs.kneighbors(x)
        k_distances = np.sort(distances[:, -1])
        second_derivative = np.diff(np.diff(k_distances))
        elbow_idx = int(np.argmax(second_derivative)) + 2
        return round(float(k_distances[elbow_idx]), 4)


class DBSCANClusterProfileBuilder:
    """Builds statistical profiles for all DBSCAN clusters including noise."""

    def build(
        self,
        x_original: pd.DataFrame,
        labels: np.ndarray,
    ) -> list[DBSCANClusterProfile]:
        """Build profiles for all clusters and noise group."""
        n_total = len(labels)
        unique_labels = sorted(set(labels))
        feature_names = x_original.columns.tolist()
        profiles: list[DBSCANClusterProfile] = []

        for label in unique_labels:
            mask = labels == label
            cluster_data = x_original[mask]
            n_members = int(mask.sum())
            profiles.append(
                DBSCANClusterProfile(
                    cluster_id=int(label),
                    n_members=n_members,
                    proportion=round(n_members / n_total, 4),
                    feature_means={
                        name: round(float(cluster_data[name].mean()), 4)
                        for name in feature_names
                    },
                    feature_stds={
                        name: round(float(cluster_data[name].std(ddof=1)), 4)
                        for name in feature_names
                    },
                    is_noise=label == -1,
                )
            )
        return profiles


class DBSCANClusterCalculator:
    """DBSCAN clustering with auto epsilon estimation and noise quantification."""

    _MINIMUM_SAMPLES: int = 10
    _MINIMUM_FEATURES: int = 1

    def __init__(self) -> None:
        self._epsilon_estimator = EpsilonEstimator()
        self._profile_builder = DBSCANClusterProfileBuilder()

    def calculate(
        self,
        data_frame: pd.DataFrame,
        feature_columns: list[str] | None = None,
        epsilon: float | None = None,
        min_samples: int = 5,
        random_seed: int = 42,
    ) -> dict:
        """Run DBSCAN clustering."""
        if feature_columns is not None:
            missing = [c for c in feature_columns if c not in data_frame.columns]
            if missing:
                raise KeyError(f"Feature columns not found: {missing}")
            numeric_df = data_frame[feature_columns].select_dtypes(include=[np.number])
        else:
            numeric_df = data_frame.select_dtypes(include=[np.number])

        if numeric_df.shape[1] < self._MINIMUM_FEATURES:
            raise ValueError(
                f"At least {self._MINIMUM_FEATURES} numeric feature required. "
                f"Got {numeric_df.shape[1]}."
            )
        if min_samples < 2:
            raise ValueError(f"min_samples must be >= 2. Got {min_samples}.")

        clean = numeric_df.dropna()
        if len(clean) < self._MINIMUM_SAMPLES:
            raise ValueError(
                f"At least {self._MINIMUM_SAMPLES} observations required. "
                f"Got {len(clean)}."
            )

        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(clean.to_numpy(dtype=float))

        effective_epsilon = (
            epsilon
            if epsilon is not None
            else self._epsilon_estimator.estimate(x_scaled, min_samples)
        )
        if effective_epsilon <= 0:
            raise ValueError(f"epsilon must be > 0. Got {effective_epsilon}.")

        model = DBSCAN(eps=effective_epsilon, min_samples=min_samples)
        labels = model.fit_predict(x_scaled)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int(np.sum(labels == -1))
        noise_ratio = round(n_noise / len(labels), 4)

        sil_score: float | None = None
        non_noise_mask = labels != -1
        if n_clusters >= 2 and non_noise_mask.sum() >= 2:
            sil_score = round(
                float(silhouette_score(
                    x_scaled[non_noise_mask], labels[non_noise_mask]
                )),
                4,
            )

        profiles = self._profile_builder.build(clean, labels)

        return {
            "labels": labels.tolist(),
            "n_clusters": n_clusters,
            "n_noise_points": n_noise,
            "noise_ratio": noise_ratio,
            "epsilon_used": effective_epsilon,
            "epsilon_auto_estimated": epsilon is None,
            "min_samples": min_samples,
            "silhouette_score": sil_score,
            "cluster_profiles": [
                {
                    "cluster_id": p.cluster_id,
                    "n_members": p.n_members,
                    "proportion": p.proportion,
                    "feature_means": p.feature_means,
                    "feature_stds": p.feature_stds,
                    "is_noise": p.is_noise,
                }
                for p in profiles
            ],
            "feature_columns": clean.columns.tolist(),
            "n_observations": len(clean),
        }
