"""Polars encoder implementations — OneHot and Ordinal."""

from __future__ import annotations

from typing import Any

import polars as pl

from lumen.preproccesing.encoders.implementations.base import AbstractEncoder, AbstractTransform


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

class PolarsGetColumns:
    """Detect string/categorical columns in a Polars DataFrame."""

    def get_columns(self, data: pl.DataFrame) -> list[str]:
        """Return column names with Utf8 or Categorical dtype."""
        return [
            col for col in data.columns
            if data[col].dtype in (pl.Utf8, pl.Categorical, pl.String)
        ]


class PolarsGetCategories:
    """Extract unique categories per column."""

    def get_categories(
        self,
        data: pl.DataFrame,
        columns: list[str],
    ) -> dict[str, list[str]]:
        """Return ``{column: [unique_values]}`` mapping."""
        return {
            col: data[col].drop_nulls().unique().sort().to_list()
            for col in columns
        }


# ─────────────────────────────────────────────────────────────────────────────
# OneHot
# ─────────────────────────────────────────────────────────────────────────────

class PolarsOneHotEncoderTransform(AbstractTransform[pl.DataFrame]):
    """Stateless one-hot encoding transform for Polars."""

    def __init__(
        self,
        data_frame: pl.DataFrame,
        columns_categories: dict[str, list[str]],
    ) -> None:
        self._data_frame = data_frame
        self._columns_categories = columns_categories

    def transform(self) -> pl.DataFrame:
        """Create binary columns for each category, drop originals.

        Returns:
            Polars DataFrame with one-hot encoded columns.
        """
        result = self._data_frame.clone()

        new_cols: list[pl.Expr] = []
        for column, categories in self._columns_categories.items():
            for category in categories:
                new_cols.append(
                    (pl.col(column) == category)
                    .cast(pl.Int8)
                    .alias(f"{column}_{category}")
                )

        result = result.with_columns(new_cols)
        result = result.drop(list(self._columns_categories.keys()))
        return result


class PolarsOneHotEncoder(AbstractEncoder[pl.DataFrame]):
    """One-hot encoder for Polars DataFrames.

    Encodes each unique string/categorical value as a separate
    binary column: ``column_value = 1`` if the row has that value.
    """

    def __init__(self) -> None:
        self._columns_categories: dict[str, list[str]] | None = None
        self._data_frame: pl.DataFrame | None = None

    def fit(self, data: pl.DataFrame, **kwargs: Any) -> None:
        """Learn unique categories from string columns.

        Args:
            data: Polars DataFrame.
        """
        columns = PolarsGetColumns().get_columns(data)
        self._columns_categories = PolarsGetCategories().get_categories(data, columns)
        self._data_frame = data

    def transform(self) -> pl.DataFrame:
        """Apply one-hot encoding.

        Returns:
            Polars DataFrame with binary columns.

        Raises:
            RuntimeError: If fit() has not been called.
        """
        if not self.is_fitted:
            raise RuntimeError("Encoder must be fitted before transformation.")
        return PolarsOneHotEncoderTransform(
            self._data_frame, self._columns_categories
        ).transform()

    @property
    def is_fitted(self) -> bool:
        return self._columns_categories is not None and self._data_frame is not None


# ─────────────────────────────────────────────────────────────────────────────
# Ordinal
# ─────────────────────────────────────────────────────────────────────────────

class PolarsOrdinalEncoderTransform(AbstractTransform[pl.DataFrame]):
    """Stateless ordinal encoding transform for Polars."""

    def __init__(
        self,
        data_frame: pl.DataFrame,
        mappings: dict[str, dict[str, int]],
    ) -> None:
        self._data_frame = data_frame
        self._mappings = mappings

    def transform(self) -> pl.DataFrame:
        """Map categorical values to integer ordinals.

        Returns:
            Polars DataFrame with ordinal-encoded columns.

        Raises:
            ValueError: If unmapped values are found.
        """
        result = self._data_frame.clone()

        for column, mapping in self._mappings.items():
            if column not in result.columns:
                continue

            # Build a replacement expression using when/then chains
            expr = pl.lit(None).cast(pl.Int64)
            for value, ordinal in mapping.items():
                expr = (
                    pl.when(pl.col(column) == value)
                    .then(pl.lit(ordinal))
                    .otherwise(expr)
                )
            result = result.with_columns(expr.alias(column))

            # Check for unmapped values (nulls where original was not null)
            null_count = result.filter(
                pl.col(column).is_null() & self._data_frame[column].is_not_null()
            ).height
            if null_count > 0:
                raise ValueError(
                    f"Column '{column}' has {null_count} unmapped values."
                )

        return result


class PolarsOrdinalEncoder(AbstractEncoder[pl.DataFrame]):
    """Ordinal encoder for Polars DataFrames.

    Maps categorical values to integers based on a provided hierarchy.
    """

    def __init__(self) -> None:
        self._columns_hierarchy: dict[str, list[str]] | None = None
        self._data_frame: pl.DataFrame | None = None

    def fit(self, data: pl.DataFrame, **kwargs: Any) -> None:
        """Learn ordinal mappings.

        Args:
            data: Polars DataFrame.
            **kwargs: Must include ``columns_categories_hierarchy``
                mapping ``{column: [ordered_values]}``.
        """
        self._columns_hierarchy = kwargs.get("columns_categories_hierarchy", {})
        self._data_frame = data

    def transform(self) -> pl.DataFrame:
        """Apply ordinal encoding.

        Returns:
            Polars DataFrame with integer-encoded columns.

        Raises:
            RuntimeError: If fit() has not been called.
        """
        if not self.is_fitted:
            raise RuntimeError("Encoder must be fitted before transformation.")

        mappings = {
            col: {val: i for i, val in enumerate(hierarchy)}
            for col, hierarchy in self._columns_hierarchy.items()
        }
        return PolarsOrdinalEncoderTransform(self._data_frame, mappings).transform()

    @property
    def is_fitted(self) -> bool:
        return self._columns_hierarchy is not None and self._data_frame is not None
