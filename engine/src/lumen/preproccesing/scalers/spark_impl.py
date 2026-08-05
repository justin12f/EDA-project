"""PySpark scaler implementations."""

from __future__ import annotations

from typing import Any

from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F

from lumen.preproccesing.scalers.base import AbstractScaler


class SparkStandardScaler(AbstractScaler[SparkDataFrame]):
    """Standard (Z-score) scaling for a single column in a Spark DataFrame.

    Since PySpark does not have a native Series type, this scaler
    operates on a full DataFrame and targets a specific column.

    Example:
        scaler = SparkStandardScaler(column="price")
        scaled_df = scaler.fit_transform(spark_df)
    """

    def __init__(self, column: str) -> None:
        self._column = column
        self._mean: float | None = None
        self._std: float | None = None

    def fit(self, data: SparkDataFrame) -> "SparkStandardScaler":
        """Learn mean and std from the target column.

        Args:
            data: PySpark DataFrame containing the target column.

        Returns:
            Self, for method chaining.

        Raises:
            ValueError: If the column is not found.
        """
        if self._column not in data.columns:
            raise ValueError(f"Column '{self._column}' not found in DataFrame.")

        # [ACTION: single Spark job to compute both aggregates]
        stats = data.agg(
            F.avg(self._column).alias("mean"),
            F.stddev(self._column).alias("std"),
        ).first()

        self._mean = float(stats["mean"]) if stats["mean"] is not None else 0.0
        self._std = float(stats["std"]) if stats["std"] is not None else 1.0
        if self._std == 0.0:
            self._std = 1.0
        return self

    def transform(self, data: SparkDataFrame) -> SparkDataFrame:
        """Apply Z-score scaling to the target column.

        Args:
            data: PySpark DataFrame.

        Returns:
            DataFrame with the target column scaled in-place.

        Raises:
            RuntimeError: If fit() has not been called.
        """
        if not self.is_fitted:
            raise RuntimeError("Scaler has not been fitted. Call fit() first.")
        return data.withColumn(
            self._column,
            (F.col(self._column) - F.lit(self._mean)) / F.lit(self._std),
        )

    @property
    def is_fitted(self) -> bool:
        """Whether the scaler has been fitted."""
        return self._mean is not None and self._std is not None

    def get_params(self) -> dict[str, Any]:
        """Return learned parameters.

        Returns:
            Dict with ``mean``, ``std``, and ``column`` keys.
        """
        if not self.is_fitted:
            raise RuntimeError("Scaler has not been fitted.")
        return {"mean": self._mean, "std": self._std, "column": self._column}
