"""Polars scaler implementations."""

from __future__ import annotations

from typing import Any

import polars as pl

from preproccesing.scalers.base import AbstractScaler


class PolarsStandardScaler(AbstractScaler[pl.Series]):
    """Standard (Z-score) scaling: ``(x - μ) / σ``.

    Scales data to zero mean and unit variance using Polars Series.

    Example:
        scaler = PolarsStandardScaler()
        scaled = scaler.fit_transform(pl.Series("col", [1, 2, 3, 4, 5]))
    """

    def __init__(self) -> None:
        self._mean: float | None = None
        self._std: float | None = None

    def fit(self, data: pl.Series) -> "PolarsStandardScaler":
        """Learn mean and std from the data.

        Args:
            data: Polars Series of numeric values.

        Returns:
            Self, for method chaining.

        Raises:
            ValueError: If data is empty.
        """
        if data.is_empty():
            raise ValueError("Cannot fit on empty Series.")
        self._mean = data.mean()
        self._std = data.std()
        if self._std == 0.0:
            self._std = 1.0  # avoid division by zero
        return self

    def transform(self, data: pl.Series) -> pl.Series:
        """Apply Z-score scaling.

        Args:
            data: Polars Series of numeric values.

        Returns:
            Scaled Polars Series.

        Raises:
            RuntimeError: If fit() has not been called.
        """
        if not self.is_fitted:
            raise RuntimeError("Scaler has not been fitted. Call fit() first.")
        return (data - self._mean) / self._std

    @property
    def is_fitted(self) -> bool:
        """Whether the scaler has been fitted."""
        return self._mean is not None and self._std is not None

    def get_params(self) -> dict[str, Any]:
        """Return learned parameters.

        Returns:
            Dict with ``mean`` and ``std`` keys.
        """
        if not self.is_fitted:
            raise RuntimeError("Scaler has not been fitted.")
        return {"mean": self._mean, "std": self._std}
