"""K-Means clustering with optimal K selection and cluster profiling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ClusterProfile:
    """Immutable profile for a single cluster."""

    cluster_id: int
    n_members: int
    proportion: float
    centroid: dict[str, float]
    feature_means: dict[str, float]
    feature_stds: dict[str, float]


class ElbowMethodCalculator:
    """Computes inertia across a range of K values for elbow detection.

    Inertia = Σ ||xᵢ - μₖ||² summed over all points and their
    assigned cluster centers. Decreases monotonically with K —
    the 'elbow' is where additional K yields diminishing returns.
    """

    def calculate(
        self,
        x: np.ndarray,
        k_range: range,
        random_seed: int,
    ) -> dict[int, float]:
        """Compute inertia for each K in range.

        Args:
            x: Scaled feature matrix.
            k_range: Range of K values to evaluate.
            random_seed: Seed for KMeans reproducibility.

        Returns:
            Dict mapping K → inertia.
        """
        return {
            k: float(
                KMeans(
                    n_clusters=k,
                    random_state=random_seed,
                    n_init=10,
                ).fit(x).inertia_
            )
            for k in k_range
        }


class SilhouetteScoreCalculator:
    """Computes silhouette score across a range of K values.

    Silhouette(i) = (b(i) - a(i)) / max(a(i), b(i))
    where a(i) = mean intra-cluster distance,
          b(i) = mean nearest-cluster distance.

    Score in [-1, 1]: higher = better separation.
    """

    def calculate(
        self,
        x: np.ndarray,
        k_range: range,
        random_seed: int,
    ) -> dict[int, float]:
        """Compute silhouette score for each K.

        Args:
            x: Scaled feature matrix.
            k_range: Range of K values (must be >= 2).
            random_seed: Seed for reproducibility.

        Returns:
            Dict mapping K → silhouette score.
        """
        scores: dict[int, float] = {}
        for k in k_range:
            if k < 2:
                continue
            labels = KMeans(
                n_clusters=k,
                random_state=random_seed,
                n_init=10,
            ).fit_predict(x)
            scores[k] = float(silhouette_score(x, labels))
        return scores


class OptimalKSelector:
    """Selects optimal K as the value maximizing silhouette score."""

    def select(self, silhouette_scores: dict[int, float]) -> int:
        """Return K with the highest silhouette score.

        Args:
            silhouette_scores: Dict mapping K → silhouette score.

        Returns:
            Optimal K value.

        Raises:
            ValueError: If silhouette_scores is empty.
        """
        if not silhouette_scores:
            raise ValueError(
                "silhouette_scores is empty. Cannot select optimal K."
            )
        return max(silhouette_scores, key=silhouette_scores.get)


class ClusterProfileBuilder:
    """Builds a statistical profile for each cluster."""

    def build(
        self,
        x_original: pd.DataFrame,
        labels: np.ndarray,
        centroids_scaled: np.ndarray,
        scaler: StandardScaler,
    ) -> list[ClusterProfile]:
        """Build profiles for all clusters.

        Args:
            x_original: Original (unscaled) feature DataFrame.
            labels: Cluster label array aligned with x_original.
            centroids_scaled: Cluster centroids in scaled space.
            scaler: Fitted scaler for inverse-transforming centroids.

        Returns:
            List of ClusterProfile sorted by cluster_id.
        """
        n_total = len(labels)
        centroids_original = scaler.inverse_transform(centroids_scaled)
        feature_names = x_original.columns.tolist()
        profiles: list[ClusterProfile] = []

        for cluster_id in range(len(centroids_scaled)):
            mask = labels == cluster_id
            cluster_data = x_original[mask]
            n_members = int(mask.sum())

            centroid = {
                name: round(float(centroids_original[cluster_id, j]), 4)
                for j, name in enumerate(feature_names)
            }
            feature_means = {
                name: round(float(cluster_data[name].mean()), 4)
                for name in feature_names
            }
            feature_stds = {
                name: round(float(cluster_data[name].std(ddof=1)), 4)
                for name in feature_names
            }

            profiles.append(
                ClusterProfile(
                    cluster_id=cluster_id,
                    n_members=n_members,
                    proportion=round(n_members / n_total, 4),
                    centroid=centroid,
                    feature_means=feature_means,
                    feature_stds=feature_stds,
                )
            )

        return profiles


class KMeansClusterCalculator:
    """K-Means clustering with elbow method, silhouette scoring, and profiling.

    Workflow:
        calculator = KMeansClusterCalculator()
        result = calculator.calculate(
            data_frame=df,
            feature_columns=["recency", "frequency", "monetary"],
            n_clusters=None,           # auto-selects optimal K
            k_range=(2, 8),            # optional, search range
            random_seed=42,
        )
    """

    _MINIMUM_SAMPLES: int = 20
    _MINIMUM_FEATURES: int = 2
    _DEFAULT_K_RANGE: tuple[int, int] = (2, 8)

    def __init__(self) -> None:
        self._elbow_calc = ElbowMethodCalculator()
        self._silhouette_calc = SilhouetteScoreCalculator()
        self._k_selector = OptimalKSelector()
        self._profile_builder = ClusterProfileBuilder()

    def calculate(
        self,
        data_frame: pd.DataFrame,
        feature_columns: list[str] | None = None,
        n_clusters: int | None = None,
        k_range: tuple[int, int] = _DEFAULT_K_RANGE,
        random_seed: int = 42,
    ) -> dict:
        """Run K-Means clustering with automatic or fixed K.

        Args:
            data_frame: Source DataFrame.
            feature_columns: Columns to cluster on. Defaults to all numeric.
            n_clusters: Fixed K. If None, auto-selects via silhouette.
            k_range: (min_k, max_k) for auto-selection search.
            random_seed: Seed for reproducibility.

        Returns:
            Dict with cluster labels, profiles, inertia, and evaluation metrics.

        Raises:
            KeyError: If feature columns are not found.
            ValueError: If data is insufficient or k_range is invalid.
        """
        if feature_columns is not None:
            missing = [c for c in feature_columns if c not in data_frame.columns]
            if missing:
                raise KeyError(f"Feature columns not found: {missing}")
            numeric_df = data_frame[feature_columns].select_dtypes(include=[np.number])
        else:
            numeric_df = data_frame.select_dtypes(include=[np.number])

        if numeric_df.shape[1] < self._MINIMUM_FEATURES:
            raise ValueError(
                f"At least {self._MINIMUM_FEATURES} numeric features required. "
                f"Got {numeric_df.shape[1]}."
            )

        clean = numeric_df.dropna()

        if len(clean) < self._MINIMUM_SAMPLES:
            raise ValueError(
                f"At least {self._MINIMUM_SAMPLES} observations required. "
                f"Got {len(clean)}."
            )

        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(clean.to_numpy(dtype=float))

        if n_clusters is not None:
            if n_clusters < 2:
                raise ValueError(f"n_clusters must be >= 2. Got {n_clusters}.")
            optimal_k = n_clusters
            inertia_curve: dict[int, float] = {}
            silhouette_scores: dict[int, float] = {}
        else:
            if k_range[0] < 2 or k_range[1] <= k_range[0]:
                raise ValueError(
                    f"k_range must satisfy k_range[0] >= 2 and k_range[1] > k_range[0]. "
                    f"Got {k_range}."
                )
            k_search = range(k_range[0], min(k_range[1] + 1, len(clean)))
            inertia_curve = self._elbow_calc.calculate(x_scaled, k_search, random_seed)
            silhouette_scores = self._silhouette_calc.calculate(
                x_scaled, k_search, random_seed
            )
            optimal_k = self._k_selector.select(silhouette_scores)

        model = KMeans(
            n_clusters=optimal_k,
            random_state=random_seed,
            n_init=10,
        )
        labels = model.fit_predict(x_scaled)
        inertia = float(model.inertia_)
        db_score = float(davies_bouldin_score(x_scaled, labels))
        sil_score = (
            float(silhouette_score(x_scaled, labels))
            if optimal_k >= 2 else 0.0
        )

        profiles = self._profile_builder.build(
            clean, labels, model.cluster_centers_, scaler
        )

        return {
            "labels": labels.tolist(),
            "n_clusters": optimal_k,
            "inertia": round(inertia, 4),
            "silhouette_score": round(sil_score, 4),
            "davies_bouldin_score": round(db_score, 4),
            "cluster_profiles": [
                {
                    "cluster_id": p.cluster_id,
                    "n_members": p.n_members,
                    "proportion": p.proportion,
                    "centroid": p.centroid,
                    "feature_means": p.feature_means,
                    "feature_stds": p.feature_stds,
                }
                for p in profiles
            ],
            "evaluation_curves": {
                "inertia": inertia_curve,
                "silhouette": silhouette_scores,
            },
            "feature_columns": clean.columns.tolist(),
            "n_observations": len(clean),
        }
