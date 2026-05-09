"""PCA dimensionality reduction with variance explained analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class PrincipalComponent:
    """Immutable result for a single principal component."""

    component_index: int
    eigenvalue: float
    variance_explained: float
    cumulative_variance_explained: float
    loadings: dict[str, float]


class CovarianceMatrixComputer:
    """Computes the sample covariance matrix from standardized data.

    Standardization is required before PCA to prevent features with
    larger scales from dominating the principal components.
    """

    def compute(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Standardize data and compute its covariance matrix.

        Args:
            data: 2D feature matrix (n_samples × n_features).

        Returns:
            Tuple (standardized_data, covariance_matrix).
        """
        scaler = StandardScaler()
        standardized = scaler.fit_transform(data)
        cov_matrix = np.cov(standardized, rowvar=False)
        return standardized, cov_matrix


class EigenDecompositionCalculator:
    """Performs eigenvalue decomposition and sorts components by variance explained."""

    def decompose(
        self, cov_matrix: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Decompose covariance matrix into eigenvalues and eigenvectors.

        Args:
            cov_matrix: Square symmetric covariance matrix.

        Returns:
            Tuple (eigenvalues, eigenvectors) sorted descending by eigenvalue.
        """
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # eigh returns ascending order — reverse for descending
        desc_idx = np.argsort(eigenvalues)[::-1]
        return eigenvalues[desc_idx], eigenvectors[:, desc_idx]


class OptimalComponentSelector:
    """Selects the minimum number of components to explain a target variance."""

    def select(
        self,
        cumulative_variance: np.ndarray,
        target_variance: float,
    ) -> int:
        """Find minimum n_components for target cumulative variance.

        Args:
            cumulative_variance: Cumulative variance explained array.
            target_variance: Target explained variance in (0, 1].

        Returns:
            Minimum number of components needed.
        """
        for i, cum_var in enumerate(cumulative_variance):
            if cum_var >= target_variance:
                return i + 1
        return len(cumulative_variance)


class DimensionalityReductionCalculator:
    """PCA with loadings, variance explained, and optimal component selection.

    Workflow:
        calculator = DimensionalityReductionCalculator()
        result = calculator.calculate(
            data_frame=df[["age", "income", "score"]],
            n_components=None,              # optional, None = all
            target_variance_explained=0.95, # optional
        )
    """

    _MINIMUM_SAMPLES: int = 10
    _MINIMUM_FEATURES: int = 2

    def __init__(self) -> None:
        self._cov_computer = CovarianceMatrixComputer()
        self._eigen_calc = EigenDecompositionCalculator()
        self._component_selector = OptimalComponentSelector()

    def calculate(
        self,
        data_frame: pd.DataFrame,
        n_components: int | None = None,
        target_variance_explained: float = 0.95,
    ) -> dict:
        """Run PCA and return variance analysis.

        Args:
            data_frame: Numeric-only DataFrame.
            n_components: Number of components to extract. None = all features.
            target_variance_explained: Target cumulative variance for
                optimal component selection.

        Returns:
            Dict with components, variance explained, projected data,
            and optimal n_components.

        Raises:
            ValueError: If data is insufficient or parameters are invalid.
        """
        if data_frame.shape[0] < self._MINIMUM_SAMPLES:
            raise ValueError(
                f"At least {self._MINIMUM_SAMPLES} observations required. "
                f"Got {data_frame.shape[0]}."
            )
        if data_frame.shape[1] < self._MINIMUM_FEATURES:
            raise ValueError(
                f"At least {self._MINIMUM_FEATURES} features required. "
                f"Got {data_frame.shape[1]}."
            )
        if not 0.0 < target_variance_explained <= 1.0:
            raise ValueError(
                f"target_variance_explained must be in (0, 1]. "
                f"Got {target_variance_explained}."
            )

        clean = data_frame.select_dtypes(include=[np.number]).dropna()
        if clean.empty:
            raise ValueError("No numeric columns with sufficient non-null values.")

        feature_names = clean.columns.tolist()
        x = clean.to_numpy(dtype=float)
        n_features = x.shape[1]

        max_components = (
            min(n_components, n_features) if n_components is not None else n_features
        )

        standardized, cov_matrix = self._cov_computer.compute(x)
        eigenvalues, eigenvectors = self._eigen_calc.decompose(cov_matrix)

        total_variance = float(eigenvalues.sum())
        variance_explained = eigenvalues / total_variance if total_variance > 0 else eigenvalues
        cumulative_variance = np.cumsum(variance_explained)

        components: list[PrincipalComponent] = [
            PrincipalComponent(
                component_index=i + 1,
                eigenvalue=round(float(eigenvalues[i]), 6),
                variance_explained=round(float(variance_explained[i]), 6),
                cumulative_variance_explained=round(float(cumulative_variance[i]), 6),
                loadings={
                    name: round(float(eigenvectors[j, i]), 6)
                    for j, name in enumerate(feature_names)
                },
            )
            for i in range(max_components)
        ]

        projected = standardized @ eigenvectors[:, :max_components]
        optimal_n = self._component_selector.select(
            cumulative_variance, target_variance_explained
        )

        return {
            "components": [
                {
                    "component_index": c.component_index,
                    "eigenvalue": c.eigenvalue,
                    "variance_explained": c.variance_explained,
                    "cumulative_variance_explained": c.cumulative_variance_explained,
                    "loadings": c.loadings,
                }
                for c in components
            ],
            "projected_data": projected.tolist(),
            "optimal_n_components": optimal_n,
            "target_variance_explained": target_variance_explained,
            "n_features_original": n_features,
            "n_components_extracted": max_components,
            "n_observations": len(clean),
            "feature_names": feature_names,
        }
