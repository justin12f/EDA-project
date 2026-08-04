"""PySpark encoder implementations — OneHot and Ordinal."""

from __future__ import annotations

from typing import Any

from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, StringType

from lumen.preproccesing.encoders.implementations.base import AbstractEncoder, AbstractTransform


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

class SparkGetColumns:
    """Detect string columns in a Spark DataFrame."""

    def get_columns(self, data: SparkDataFrame) -> list[str]:
        """Return column names with StringType."""
        return [
            field.name for field in data.schema.fields
            if isinstance(field.dataType, StringType)
        ]


class SparkGetCategories:
    """Extract unique categories per column in Spark."""

    def get_categories(
        self,
        data: SparkDataFrame,
        columns: list[str],
    ) -> dict[str, list[str]]:
        """Return ``{column: [unique_values]}`` mapping.

        Note:
            Each column requires a ``.distinct().collect()`` action.
        """
        result: dict[str, list[str]] = {}
        for col in columns:
            # [ACTION: one Spark job per column for distinct values]
            rows = (
                data.select(col)
                .filter(F.col(col).isNotNull())
                .distinct()
                .orderBy(col)
                .collect()
            )
            result[col] = sorted([row[0] for row in rows])
        return result


# ─────────────────────────────────────────────────────────────────────────────
# OneHot
# ─────────────────────────────────────────────────────────────────────────────

class SparkOneHotEncoderTransform(AbstractTransform[SparkDataFrame]):
    """Stateless one-hot encoding for Spark DataFrames."""

    def __init__(
        self,
        data_frame: SparkDataFrame,
        columns_categories: dict[str, list[str]],
    ) -> None:
        self._data_frame = data_frame
        self._columns_categories = columns_categories

    def transform(self) -> SparkDataFrame:
        """Create binary columns for each category, drop originals.

        Returns:
            Spark DataFrame with one-hot encoded columns.
        """
        result = self._data_frame

        for column, categories in self._columns_categories.items():
            for category in categories:
                result = result.withColumn(
                    f"{column}_{category}",
                    F.when(F.col(column) == category, F.lit(1))
                    .otherwise(F.lit(0))
                    .cast(IntegerType()),
                )
            result = result.drop(column)

        return result


class SparkOneHotEncoder(AbstractEncoder[SparkDataFrame]):
    """One-hot encoder for PySpark DataFrames.

    Creates binary columns for each unique categorical value.
    """

    def __init__(self) -> None:
        self._columns_categories: dict[str, list[str]] | None = None
        self._data_frame: SparkDataFrame | None = None

    def fit(self, data: SparkDataFrame, **kwargs: Any) -> None:
        """Learn unique categories from string columns.

        Args:
            data: PySpark DataFrame.
        """
        columns = SparkGetColumns().get_columns(data)
        self._columns_categories = SparkGetCategories().get_categories(data, columns)
        self._data_frame = data

    def transform(self) -> SparkDataFrame:
        """Apply one-hot encoding.

        Returns:
            Spark DataFrame with binary columns.

        Raises:
            RuntimeError: If fit() has not been called.
        """
        if not self.is_fitted:
            raise RuntimeError("Encoder must be fitted before transformation.")
        return SparkOneHotEncoderTransform(
            self._data_frame, self._columns_categories
        ).transform()

    @property
    def is_fitted(self) -> bool:
        return self._columns_categories is not None and self._data_frame is not None


# ─────────────────────────────────────────────────────────────────────────────
# Ordinal
# ─────────────────────────────────────────────────────────────────────────────

class SparkOrdinalEncoderTransform(AbstractTransform[SparkDataFrame]):
    """Stateless ordinal encoding for Spark DataFrames."""

    def __init__(
        self,
        data_frame: SparkDataFrame,
        mappings: dict[str, dict[str, int]],
    ) -> None:
        self._data_frame = data_frame
        self._mappings = mappings

    def transform(self) -> SparkDataFrame:
        """Map categorical values to integer ordinals.

        Returns:
            Spark DataFrame with ordinal-encoded columns.

        Raises:
            ValueError: If unmapped values are found.
        """
        result = self._data_frame

        for column, mapping in self._mappings.items():
            if column not in result.columns:
                continue

            # Build CASE WHEN chain
            expr = F.lit(None).cast(IntegerType())
            for value, ordinal in mapping.items():
                expr = (
                    F.when(F.col(column) == value, F.lit(ordinal))
                    .otherwise(expr)
                )
            result = result.withColumn(column, expr)

        return result


class SparkOrdinalEncoder(AbstractEncoder[SparkDataFrame]):
    """Ordinal encoder for PySpark DataFrames.

    Maps categorical values to integers based on a provided hierarchy.
    """

    def __init__(self) -> None:
        self._columns_hierarchy: dict[str, list[str]] | None = None
        self._data_frame: SparkDataFrame | None = None

    def fit(self, data: SparkDataFrame, **kwargs: Any) -> None:
        """Learn ordinal mappings.

        Args:
            data: PySpark DataFrame.
            **kwargs: Must include ``columns_categories_hierarchy``.
        """
        self._columns_hierarchy = kwargs.get("columns_categories_hierarchy", {})
        self._data_frame = data

    def transform(self) -> SparkDataFrame:
        """Apply ordinal encoding.

        Returns:
            Spark DataFrame with integer-encoded columns.

        Raises:
            RuntimeError: If fit() has not been called.
        """
        if not self.is_fitted:
            raise RuntimeError("Encoder must be fitted before transformation.")

        mappings = {
            col: {val: i for i, val in enumerate(hierarchy)}
            for col, hierarchy in self._columns_hierarchy.items()
        }
        return SparkOrdinalEncoderTransform(self._data_frame, mappings).transform()

    @property
    def is_fitted(self) -> bool:
        return self._columns_hierarchy is not None and self._data_frame is not None
