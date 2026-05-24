"""Hierarchical agglomerative clustering with dendrogram and linkage analysis."""

# #[AI_CONTEXT_START]
# - CONFIGURACIÓN DE FACTORY: Registrar este calculator en `SegmentationStatisticsFactory` (backends pandas | polars | spark) y exponerlo mediante `StatisticsInyeccionDependency`, inyectada por la Factory Maestra de Agentes junto a las demás fábricas de dominio.
# - ABSTRACCIÓN DEL DATO: Mutar constructores y `analyze`/`compute` para recibir el contenedor abstracto del backend (`pd.DataFrame`, `pl.DataFrame`/`pl.LazyFrame`, `pyspark.sql.DataFrame`) inyectado por la factory; eliminar `np.ndarray`/`pd.Series` sueltos en firmas públicas.
# - REFACTOR NATIVO: Resolver métricas con expresiones 100 % nativas del backend activo (Polars: `.select`/`.group_by`/`.agg` sin `.collect()` salvo materialización acordada; PySpark: `pyspark.sql.functions` y ventanas distribuidas; Pandas: operaciones vectorizadas). No convertir a NumPy/Pandas desde backends no-pandas.
# #[AI_CONTEXT_END]
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage, cophenet
from scipy.spatial.distance import pdist
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

@dataclass(frozen=True)
class HierarchicalClusterProfile:
    """Immutable profile for a single hierarchical cluster."""

    cluster_id: int
    n_members: int
    proportion: float
    feature_means: dict[str, float]
    feature_stds: dict[str, float]

class LinkageMatrixBuilder:
    """Builds the hierarchical linkage matrix from scaled data."""

    _VALID_METHODS: frozenset[str] = frozenset(
        {"ward", "complete", "average", "single"}
    )

    def build(self, x_scaled: np.ndarray, method: str) -> np.ndarray:
        """Build linkage matrix."""
        if method not in self._VALID_METHODS:
            raise ValueError(
                f"method must be one of {self._VALID_METHODS}. Got '{method}'."
            )
        return linkage(x_scaled, method=method)

class CopheneticCorrelationCalculator:
    """Computes cophenetic correlation coefficient for linkage quality."""

    def calculate(
        self, linkage_matrix: np.ndarray, x_scaled: np.ndarray
    ) -> float:
        """Compute cophenetic correlation coefficient."""
        c, _ = cophenet(linkage_matrix, pdist(x_scaled))
        return round(float(c), 4)

class OptimalCutoffSelector:
    """Selects optimal number of clusters by maximizing silhouette score."""

    def select(
        self,
        linkage_matrix: np.ndarray,
        x_scaled: np.ndarray,
        k_range: range,
    ) -> tuple[int, dict[int, float]]:
        """Select optimal K and return silhouette scores per K."""
        silhouette_scores: dict[int, float] = {}
        for k in k_range:
            if k >= len(x_scaled):
                continue
            labels = fcluster(linkage_matrix, k, criterion="maxclust")
            if len(set(labels)) < 2:
                continue
            score = float(silhouette_score(x_scaled, labels))
            silhouette_scores[k] = round(score, 4)
        if not silhouette_scores:
            return 2, {}
        optimal_k = max(silhouette_scores, key=silhouette_scores.get)
        return optimal_k, silhouette_scores

class DendrogramDataExtractor:
    """Extracts dendrogram structure data for visualization."""

    def extract(self, linkage_matrix: np.ndarray) -> dict:
        """Extract dendrogram node coordinates and structure."""
        ddata = dendrogram(linkage_matrix, no_plot=True)
        return {
            "icoord": ddata["icoord"],
            "dcoord": ddata["dcoord"],
            "leaves": ddata["leaves"],
            "color_list": ddata.get("color_list", []),
        }

class HierarchicalClusterProfileBuilder:
    """Builds statistical profiles per cluster from hierarchical assignments."""

    def build(
        self,
        x_original: pd.DataFrame,
        labels: np.ndarray,
    ) -> list[HierarchicalClusterProfile]:
        """Build per-cluster profiles."""
        n_total = len(labels)
        feature_names = x_original.columns.tolist()
        return [
            HierarchicalClusterProfile(
                cluster_id=int(cluster_id),
                n_members=int((labels == cluster_id).sum()),
                proportion=round(float((labels == cluster_id).sum()) / n_total, 4),
                feature_means={
                    name: round(
                        float(x_original[name][labels == cluster_id].mean()), 4
                    )
                    for name in feature_names
                },
                feature_stds={
                    name: round(
                        float(x_original[name][labels == cluster_id].std(ddof=1)), 4
                    )
                    for name in feature_names
                },
            )
            for cluster_id in sorted(set(labels))
        ]

class HierarchicalClusterCalculator:
    """Hierarchical agglomerative clustering with auto K selection."""

    _MINIMUM_SAMPLES: int = 10
    _MINIMUM_FEATURES: int = 2
    _DEFAULT_K_RANGE: tuple[int, int] = (2, 8)

    def __init__(self) -> None:
        self._linkage_builder = LinkageMatrixBuilder()
        self._cophenetic_calc = CopheneticCorrelationCalculator()
        self._cutoff_selector = OptimalCutoffSelector()
        self._dendrogram_extractor = DendrogramDataExtractor()
        self._profile_builder = HierarchicalClusterProfileBuilder()

    def calculate(
        self,
        data_frame: pd.DataFrame,
        feature_columns: list[str] | None = None,
        n_clusters: int | None = None,
        k_range: tuple[int, int] = _DEFAULT_K_RANGE,
        linkage_method: str = "ward",
        extract_dendrogram: bool = False,
    ) -> dict:
        """Run hierarchical clustering."""
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

        if k_range[0] < 2 or k_range[1] <= k_range[0]:
            raise ValueError(
                f"k_range must satisfy k_range[0] >= 2 and k_range[1] > k_range[0]. "
                f"Got {k_range}."
            )

        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(clean.to_numpy(dtype=float))

        linkage_matrix = self._linkage_builder.build(x_scaled, linkage_method)
        cophenetic_r = self._cophenetic_calc.calculate(linkage_matrix, x_scaled)

        if n_clusters is not None:
            if n_clusters < 2:
                raise ValueError(f"n_clusters must be >= 2. Got {n_clusters}.")
            optimal_k = n_clusters
            silhouette_scores: dict[int, float] = {}
        else:
            k_search = range(k_range[0], min(k_range[1] + 1, len(clean)))
            optimal_k, silhouette_scores = self._cutoff_selector.select(
                linkage_matrix, x_scaled, k_search
            )

        labels = fcluster(linkage_matrix, optimal_k, criterion="maxclust")
        sil_score = (
            round(float(silhouette_score(x_scaled, labels)), 4)
            if len(set(labels)) >= 2 else None
        )

        profiles = self._profile_builder.build(clean, labels)
        dendrogram_data = (
            self._dendrogram_extractor.extract(linkage_matrix)
            if extract_dendrogram else None
        )

        return {
            "labels": labels.tolist(),
            "n_clusters": optimal_k,
            "linkage_method": linkage_method,
            "cophenetic_correlation": cophenetic_r,
            "silhouette_score": sil_score,
            "evaluation_curves": {"silhouette": silhouette_scores},
            "cluster_profiles": [
                {
                    "cluster_id": p.cluster_id,
                    "n_members": p.n_members,
                    "proportion": p.proportion,
                    "feature_means": p.feature_means,
                    "feature_stds": p.feature_stds,
                }
                for p in profiles
            ],
            "dendrogram": dendrogram_data,
            "feature_columns": clean.columns.tolist(),
            "n_observations": len(clean),
        }
