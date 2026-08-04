"""Abstract scaler contract — backend-agnostic.

Zero imports from Polars, PySpark, or Pandas.
All scalers follow a fit → transform → fit_transform lifecycle.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

# S represents the Series/Column type of the chosen backend.
# Polars → pl.Series or pl.Expr
# PySpark → pyspark.sql.Column
S = TypeVar("S")


class AbstractScaler(ABC, Generic[S]):
    """Contract for fitting and transforming numerical data via scaling.

    Implementations bind ``S`` to their concrete column/series type.

    Lifecycle:
        1. ``fit(data)``  — learn parameters (mean, std, etc.)
        2. ``transform(data)`` — apply the learned scaling
        3. ``fit_transform(data)`` — convenience: fit + transform

    Properties:
        ``is_fitted`` — whether ``fit()`` has been called.
    """

    @abstractmethod
    def fit(self, data: S) -> "AbstractScaler[S]":
        """Learn scaling parameters from the data.

        Args:
            data: A column/series of numeric values.

        Returns:
            Self, for method chaining.

        Raises:
            ValueError: If data is empty or non-numeric.
        """

    @abstractmethod
    def transform(self, data: S) -> S:
        """Apply the learned scaling to data.

        Args:
            data: A column/series of numeric values.

        Returns:
            Scaled column/series in the same backend type.

        Raises:
            RuntimeError: If ``fit()`` has not been called.
        """

    def fit_transform(self, data: S) -> S:
        """Fit and transform in a single call.

        Args:
            data: A column/series of numeric values.

        Returns:
            Scaled column/series.
        """
        self.fit(data)
        return self.transform(data)

    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """Whether the scaler has been fitted."""

    @abstractmethod
    def get_params(self) -> dict[str, Any]:
        """Return the learned parameters (e.g., mean, std).

        Returns:
            Dict of parameter names to values.

        Raises:
            RuntimeError: If ``fit()`` has not been called.
        """
