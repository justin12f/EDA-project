"""
data_cleaning/steps/polars_impl.py

Polars implementation of all data cleaning pipeline steps.

All steps operate on pl.LazyFrame or pl.DataFrame and return the same type.
Native Polars expressions are used exclusively — no .apply() or to_pandas().
"""

# #[AI_CONTEXT_START]
# - CONFIGURACIÓN DE FACTORY: `DataCleaningStepFactory` del backend correspondiente en `data_cleaning/steps/backends/`; inyección vía `DataCleaningInyeccionDependency` → Factory Maestra.
# - ABSTRACCIÓN DEL DATO: Canonicalizar en `backends/`; deprecar duplicados en `steps/implementations.py` y `steps/polars_impl.py` raíz tras verificar referencias.
# - REFACTOR NATIVO: Steps en inglés y 100 % API nativa del backend; sin NumPy salvo materialización local explícita.
# #[AI_CONTEXT_END]
from __future__ import annotations

import re
import unicodedata
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Final

import numpy as np
import polars as pl

from data_cleaning.steps.base import AbstractStep

# ─────────────────────────────────────────────────────────────────────────────
# Domain Types (reused from original — backend-agnostic)
# ─────────────────────────────────────────────────────────────────────────────

class ImputationStrategy(str, Enum):
    """Supported strategies for filling missing numeric values."""
    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"

class CategoricalImputationStrategy(str, Enum):
    """Supported strategies for filling missing categorical values."""
    MODE = "mode"
    FIXED = "fixed"

@dataclass(frozen=True)
class DomainBounds:
    """Immutable descriptor for domain validation on a single column."""
    lower: float | None = None
    upper: float | None = None

@dataclass(frozen=True)
class CrossColumnRule:
    """Immutable descriptor for a cross-column validation rule."""
    if_col: str
    equals: object
    then_col: str
    action: str = "set_nan"

# ─────────────────────────────────────────────────────────────────────────────
# Utility: Column Scope Decorator
# ─────────────────────────────────────────────────────────────────────────────

class PolarsColumnScopedStep(AbstractStep[pl.DataFrame]):
    """Decorator that restricts any step to a subset of columns.

    Applies the inner step only to the specified columns, leaving
    all other columns unchanged.
    """

    def __init__(
        self,
        inner_step: AbstractStep[pl.DataFrame],
        data_frame: pl.DataFrame,
        columns: list[str],
    ) -> None:
        super().__init__(data_frame)
        self._inner_step = inner_step
        self._scoped_columns = columns

    def process(self, data: pl.DataFrame) -> pl.DataFrame:
        available = [c for c in self._scoped_columns if c in data.columns]
        if not available:
            return data.clone()

        scoped = data.select(available)
        remainder = data.drop(available)

        cleaned = self._inner_step.process(scoped)

        return pl.concat([remainder, cleaned], how="horizontal").select(data.columns)

# ─────────────────────────────────────────────────────────────────────────────
# Steps: Schema & Structure
# ─────────────────────────────────────────────────────────────────────────────

class PolarsColumnsTitlesStep(AbstractStep[pl.DataFrame]):
    """Normalize column names: lowercase, strip, replace spaces with underscores,
    remove accents.
    """

    def __init__(self, data_frame: pl.DataFrame) -> None:
        super().__init__(data_frame)

    def process(self, data: pl.DataFrame) -> pl.DataFrame:
        new_names: dict[str, str] = {}
        for col in data.columns:
            clean = col.strip().lower()
            clean = re.sub(r"\s+", "_", clean)
            # Remove accents
            clean = "".join(
                c for c in unicodedata.normalize("NFD", clean)
                if unicodedata.category(c) != "Mn"
            )
            new_names[col] = clean
        return data.rename(new_names)

class PolarsEnforceSchemaStep(AbstractStep[pl.DataFrame]):
    """Validate that required columns exist and minimum row count is met."""

    def __init__(
        self,
        data_frame: pl.DataFrame,
        required_columns: list[str] | None = None,
        min_rows: int = 1,
    ) -> None:
        super().__init__(data_frame)
        self._required = required_columns or []
        self._min_rows = min_rows

    def process(self, data: pl.DataFrame) -> pl.DataFrame:
        if data.height < self._min_rows:
            raise ValueError(
                f"DataFrame has {data.height} rows, minimum is {self._min_rows}."
            )
        missing = [c for c in self._required if c not in data.columns]
        if missing:
            warnings.warn(f"Missing required columns: {missing}")
        return data

class PolarsDropHighMissingColumnsStep(AbstractStep[pl.DataFrame]):
    """Drop columns where the null fraction exceeds a threshold."""

    def __init__(
        self,
        data_frame: pl.DataFrame,
        threshold: float = 0.8,
    ) -> None:
        super().__init__(data_frame)
        self._threshold = threshold

    def process(self, data: pl.DataFrame) -> pl.DataFrame:
        n = data.height
        if n == 0:
            return data
        cols_to_drop = [
            col for col in data.columns
            if data[col].null_count() / n > self._threshold
        ]
        return data.drop(cols_to_drop) if cols_to_drop else data

class PolarsDropConstantColumnsStep(AbstractStep[pl.DataFrame]):
    """Drop columns with only one unique non-null value."""

    def __init__(self, data_frame: pl.DataFrame) -> None:
        super().__init__(data_frame)

    def process(self, data: pl.DataFrame) -> pl.DataFrame:
        cols_to_drop = [
            col for col in data.columns
            if data[col].drop_nulls().n_unique() <= 1
        ]
        return data.drop(cols_to_drop) if cols_to_drop else data

class PolarsHandleSentinelValuesStep(AbstractStep[pl.DataFrame]):
    """Replace common sentinel values with null."""

    _SENTINELS: Final[frozenset[str]] = frozenset({
        "n/a", "na", "nan", "null", "none", "unknown", "missing",
        "-", "--", ".", "?", "N/A", "NA", "NaN", "NULL", "None",
        "UNKNOWN", "MISSING", "", " ",
    })

    def __init__(self, data_frame: pl.DataFrame) -> None:
        super().__init__(data_frame)

    def process(self, data: pl.DataFrame) -> pl.DataFrame:
        sentinel_list = list(self._SENTINELS)
        exprs = []
        for col in data.columns:
            if data[col].dtype in (pl.Utf8, pl.String):
                exprs.append(
                    pl.when(
                        pl.col(col).str.strip_chars().str.to_lowercase().is_in(sentinel_list)
                    )
                    .then(pl.lit(None))
                    .otherwise(pl.col(col))
                    .alias(col)
                )
            else:
                exprs.append(pl.col(col))
        return data.select(exprs)

# ─────────────────────────────────────────────────────────────────────────────
# Steps: Imputation
# ─────────────────────────────────────────────────────────────────────────────

class PolarsImputeCategoricalStep(AbstractStep[pl.DataFrame]):
    """Impute missing categorical values using mode or a fixed value."""

    def __init__(
        self,
        data_frame: pl.DataFrame,
        strategy: str = "mode",
        fill_value: str = "missing",
    ) -> None:
        super().__init__(data_frame)
        self._strategy = CategoricalImputationStrategy(strategy)
        self._fill_value = fill_value

    def process(self, data: pl.DataFrame) -> pl.DataFrame:
        string_cols = [
            col for col in data.columns
            if data[col].dtype in (pl.Utf8, pl.String, pl.Categorical)
        ]
        if not string_cols:
            return data

        exprs = []
        for col in data.columns:
            if col in string_cols and data[col].null_count() > 0:
                if self._strategy == CategoricalImputationStrategy.MODE:
                    # [COLLECT: needed for mode scalar]
                    mode_val = (
                        data[col]
                        .drop_nulls()
                        .value_counts()
                        .sort("count", descending=True)
                        .get_column(col)[0]
                    )
                    exprs.append(pl.col(col).fill_null(pl.lit(mode_val)).alias(col))
                else:
                    exprs.append(pl.col(col).fill_null(pl.lit(self._fill_value)).alias(col))
            else:
                exprs.append(pl.col(col))
        return data.select(exprs)

# ─────────────────────────────────────────────────────────────────────────────
# Steps: Type Conversion
# ─────────────────────────────────────────────────────────────────────────────

class PolarsSafeConversionStep(AbstractStep[pl.DataFrame]):
    """Attempt safe type conversion: strings to numeric or date."""

    def __init__(self, data_frame: pl.DataFrame) -> None:
        super().__init__(data_frame)

    def process(self, data: pl.DataFrame) -> pl.DataFrame:
        exprs = []
        for col in data.columns:
            if data[col].dtype in (pl.Utf8, pl.String):
                # Try numeric first
                try:
                    test = data[col].str.strip_chars().cast(pl.Float64, strict=False)
                    non_null_original = data[col].drop_nulls().len()
                    non_null_cast = test.drop_nulls().len()
                    if non_null_original > 0 and non_null_cast / non_null_original > 0.7:
                        exprs.append(test.alias(col))
                        continue
                except Exception:
                    pass
                exprs.append(pl.col(col))
            else:
                exprs.append(pl.col(col))
        return data.select(exprs)

class PolarsFixDatesColumnsStep(AbstractStep[pl.DataFrame]):
    """Convert detected date columns to Date/Datetime type."""

    def __init__(self, data_frame: pl.DataFrame) -> None:
        super().__init__(data_frame)

    def process(self, data: pl.DataFrame) -> pl.DataFrame:
        exprs = []
        for col in data.columns:
            if data[col].dtype in (pl.Utf8, pl.String):
                try:
                    test = data[col].str.to_date(strict=False)
                    non_null_parsed = test.drop_nulls().len()
                    non_null_original = data[col].drop_nulls().len()
                    if non_null_original > 0 and non_null_parsed / non_null_original > 0.5:
                        exprs.append(test.alias(col))
                        continue
                except Exception:
                    pass
            exprs.append(pl.col(col))
        return data.select(exprs)

class PolarsFixBoolsColumnsStep(AbstractStep[pl.DataFrame]):
    """Convert columns with boolean-like values to Boolean type."""

    _TRUE_VALUES: Final[frozenset[str]] = frozenset({"yes", "y", "true", "1"})
    _FALSE_VALUES: Final[frozenset[str]] = frozenset({"no", "n", "false", "0"})
    _BOOL_VOCAB: Final[frozenset[str]] = _TRUE_VALUES | _FALSE_VALUES

    def __init__(self, data_frame: pl.DataFrame) -> None:
        super().__init__(data_frame)

    def process(self, data: pl.DataFrame) -> pl.DataFrame:
        exprs = []
        for col in data.columns:
            if data[col].dtype not in (pl.Utf8, pl.String):
                exprs.append(pl.col(col))
                continue

            unique_vals = set(
                data[col].drop_nulls().str.to_lowercase().str.strip_chars().unique().to_list()
            ) - {"nan", "<na>", "none", ""}

            if unique_vals and unique_vals.issubset(self._BOOL_VOCAB):
                true_list = list(self._TRUE_VALUES)
                exprs.append(
                    pl.col(col)
                    .str.to_lowercase()
                    .str.strip_chars()
                    .is_in(true_list)
                    .alias(col)
                )
            else:
                exprs.append(pl.col(col))
        return data.select(exprs)

class PolarsFixColumnsTypesStep(AbstractStep[pl.DataFrame]):
    """Final pass: cast columns to their most appropriate type."""

    def __init__(self, data_frame: pl.DataFrame) -> None:
        super().__init__(data_frame)

    def process(self, data: pl.DataFrame) -> pl.DataFrame:
        # Polars already handles types well; this is mostly a pass-through
        # with downcast for integer columns
        exprs = []
        for col in data.columns:
            if data[col].dtype == pl.Float64:
                # Check if all values are integer-like
                non_null = data[col].drop_nulls()
                if non_null.len() > 0:
                    is_int = (non_null == non_null.round(0)).all()
                    if is_int:
                        exprs.append(pl.col(col).cast(pl.Int64, strict=False).alias(col))
                        continue
            exprs.append(pl.col(col))
        return data.select(exprs)

class PolarsFixNumericColumnsStep(AbstractStep[pl.DataFrame]):
    """Impute missing numeric values using mean, median, or mode."""

    def __init__(
        self,
        data_frame: pl.DataFrame,
        fixcase: str = "median",
    ) -> None:
        super().__init__(data_frame)
        self._strategy = ImputationStrategy(fixcase)

    def process(self, data: pl.DataFrame) -> pl.DataFrame:
        numeric_cols = [
            col for col in data.columns
            if data[col].dtype.is_numeric()
        ]
        if not numeric_cols:
            return data

        exprs = []
        for col in data.columns:
            if col in numeric_cols and data[col].null_count() > 0:
                if self._strategy == ImputationStrategy.MEAN:
                    # [COLLECT: needed for mean scalar]
                    fill_val = data[col].mean()
                elif self._strategy == ImputationStrategy.MEDIAN:
                    fill_val = data[col].median()
                else:  # MODE
                    fill_val = (
                        data[col].drop_nulls()
                        .value_counts()
                        .sort("count", descending=True)
                        .get_column(col)[0]
                    )
                exprs.append(pl.col(col).fill_null(pl.lit(fill_val)).alias(col))
            else:
                exprs.append(pl.col(col))
        return data.select(exprs)

# ─────────────────────────────────────────────────────────────────────────────
# Steps: Outliers
# ─────────────────────────────────────────────────────────────────────────────

class PolarsIQROutlierStep(AbstractStep[pl.DataFrame]):
    """Remove rows with outliers based on IQR method."""

    def __init__(
        self,
        data_frame: pl.DataFrame,
        multiplier: float = 1.5,
    ) -> None:
        super().__init__(data_frame)
        self._multiplier = multiplier

    def process(self, data: pl.DataFrame) -> pl.DataFrame:
        numeric_cols = [c for c in data.columns if data[c].dtype.is_numeric()]
        if not numeric_cols:
            return data

        mask = pl.lit(True)
        for col in numeric_cols:
            # [COLLECT: needed for percentile scalars]
            q1 = data[col].quantile(0.25)
            q3 = data[col].quantile(0.75)
            if q1 is None or q3 is None:
                continue
            iqr = q3 - q1
            lower = q1 - self._multiplier * iqr
            upper = q3 + self._multiplier * iqr
            mask = mask & (
                pl.col(col).is_null() |
                ((pl.col(col) >= lower) & (pl.col(col) <= upper))
            )
        return data.filter(mask)

class PolarsZScoreOutlierStep(AbstractStep[pl.DataFrame]):
    """Remove rows with outliers based on Z-score."""

    def __init__(
        self,
        data_frame: pl.DataFrame,
        z_threshold: float = 3.0,
    ) -> None:
        super().__init__(data_frame)
        self._z_threshold = z_threshold

    def process(self, data: pl.DataFrame) -> pl.DataFrame:
        numeric_cols = [c for c in data.columns if data[c].dtype.is_numeric()]
        if not numeric_cols:
            return data

        mask = pl.lit(True)
        for col in numeric_cols:
            # [COLLECT: needed for mean/std scalars]
            mean_val = data[col].mean()
            std_val = data[col].std()
            if mean_val is None or std_val is None or std_val == 0:
                continue
            z_score = ((pl.col(col) - mean_val) / std_val).abs()
            mask = mask & (pl.col(col).is_null() | (z_score <= self._z_threshold))
        return data.filter(mask)

class PolarsCapOutliersStep(AbstractStep[pl.DataFrame]):
    """Cap outliers at the specified percentiles (winsorization)."""

    def __init__(
        self,
        data_frame: pl.DataFrame,
        lower_percentile: float = 0.01,
        upper_percentile: float = 0.99,
    ) -> None:
        super().__init__(data_frame)
        self._lower = lower_percentile
        self._upper = upper_percentile

    def process(self, data: pl.DataFrame) -> pl.DataFrame:
        numeric_cols = [c for c in data.columns if data[c].dtype.is_numeric()]
        if not numeric_cols:
            return data

        exprs = []
        for col in data.columns:
            if col in numeric_cols:
                # [COLLECT: needed for percentile scalars]
                lo = data[col].quantile(self._lower)
                hi = data[col].quantile(self._upper)
                if lo is not None and hi is not None:
                    exprs.append(pl.col(col).clip(lo, hi).alias(col))
                else:
                    exprs.append(pl.col(col))
            else:
                exprs.append(pl.col(col))
        return data.select(exprs)

# ─────────────────────────────────────────────────────────────────────────────
# Steps: Text & Normalization
# ─────────────────────────────────────────────────────────────────────────────

class PolarsFixNotNumericColumnsStep(AbstractStep[pl.DataFrame]):
    """Strip whitespace and normalize string columns."""

    def __init__(self, data_frame: pl.DataFrame) -> None:
        super().__init__(data_frame)

    def process(self, data: pl.DataFrame) -> pl.DataFrame:
        exprs = []
        for col in data.columns:
            if data[col].dtype in (pl.Utf8, pl.String):
                exprs.append(pl.col(col).str.strip_chars().alias(col))
            else:
                exprs.append(pl.col(col))
        return data.select(exprs)

class PolarsTextStandardizationStep(AbstractStep[pl.DataFrame]):
    """Standardize text columns: strip, lowercase, collapse whitespace."""

    def __init__(self, data_frame: pl.DataFrame) -> None:
        super().__init__(data_frame)

    def process(self, data: pl.DataFrame) -> pl.DataFrame:
        exprs = []
        for col in data.columns:
            if data[col].dtype in (pl.Utf8, pl.String):
                exprs.append(
                    pl.col(col)
                    .str.strip_chars()
                    .str.to_lowercase()
                    .str.replace_all(r"\s+", " ")
                    .alias(col)
                )
            else:
                exprs.append(pl.col(col))
        return data.select(exprs)

class PolarsNormalizeCategoriesStep(AbstractStep[pl.DataFrame]):
    """Normalize categorical values: strip + lowercase."""

    def __init__(self, data_frame: pl.DataFrame) -> None:
        super().__init__(data_frame)

    def process(self, data: pl.DataFrame) -> pl.DataFrame:
        exprs = []
        for col in data.columns:
            if data[col].dtype in (pl.Utf8, pl.String):
                exprs.append(
                    pl.col(col)
                    .str.strip_chars()
                    .str.to_lowercase()
                    .alias(col)
                )
            else:
                exprs.append(pl.col(col))
        return data.select(exprs)

# ─────────────────────────────────────────────────────────────────────────────
# Steps: Deduplication & Quality
# ─────────────────────────────────────────────────────────────────────────────

class PolarsRemoveDuplicatesRowsStep(AbstractStep[pl.DataFrame]):
    """Remove exact duplicate rows."""

    def __init__(self, data_frame: pl.DataFrame) -> None:
        super().__init__(data_frame)

    def process(self, data: pl.DataFrame) -> pl.DataFrame:
        return data.unique()

class PolarsFlagDataQualityStep(AbstractStep[pl.DataFrame]):
    """Add a ``_data_quality_score`` column: fraction of non-null values per row."""

    def __init__(self, data_frame: pl.DataFrame) -> None:
        super().__init__(data_frame)

    def process(self, data: pl.DataFrame) -> pl.DataFrame:
        n_cols = len(data.columns)
        if n_cols == 0:
            return data

        non_null_expr = pl.sum_horizontal(
            [pl.col(c).is_not_null().cast(pl.Int8) for c in data.columns]
        )
        return data.with_columns(
            (non_null_expr / n_cols).round(4).alias("_data_quality_score")
        )

class PolarsAddAuditColumnsStep(AbstractStep[pl.DataFrame]):
    """Add audit metadata: ``_cleaned_at`` timestamp and ``_row_hash``."""

    def __init__(self, data_frame: pl.DataFrame) -> None:
        super().__init__(data_frame)

    def process(self, data: pl.DataFrame) -> pl.DataFrame:
        now_str = datetime.now(timezone.utc).isoformat()
        return data.with_columns(
            pl.lit(now_str).alias("_cleaned_at"),
            pl.arange(0, data.height).alias("_row_id"),
        )

# ─────────────────────────────────────────────────────────────────────────────
# Steps: Validation
# ─────────────────────────────────────────────────────────────────────────────

class PolarsValidateDomainRulesStep(AbstractStep[pl.DataFrame]):
    """Validate numeric columns against domain bounds, setting violations to null."""

    def __init__(
        self,
        data_frame: pl.DataFrame,
        rules: dict[str, DomainBounds] | None = None,
    ) -> None:
        super().__init__(data_frame)
        self._rules = rules or {}

    def process(self, data: pl.DataFrame) -> pl.DataFrame:
        exprs = []
        for col in data.columns:
            if col in self._rules:
                bounds = self._rules[col]
                expr = pl.col(col)
                if bounds.lower is not None:
                    expr = pl.when(pl.col(col) < bounds.lower).then(None).otherwise(expr)
                if bounds.upper is not None:
                    expr = pl.when(pl.col(col) > bounds.upper).then(None).otherwise(expr)
                exprs.append(expr.alias(col))
            else:
                exprs.append(pl.col(col))
        return data.select(exprs) if exprs else data

class PolarsCrossColumnValidationStep(AbstractStep[pl.DataFrame]):
    """Apply cross-column validation rules."""

    def __init__(
        self,
        data_frame: pl.DataFrame,
        rules: list[CrossColumnRule] | None = None,
    ) -> None:
        super().__init__(data_frame)
        self._rules = rules or []

    def process(self, data: pl.DataFrame) -> pl.DataFrame:
        for rule in self._rules:
            if rule.if_col in data.columns and rule.then_col in data.columns:
                if rule.action == "set_nan":
                    data = data.with_columns(
                        pl.when(pl.col(rule.if_col) == rule.equals)
                        .then(None)
                        .otherwise(pl.col(rule.then_col))
                        .alias(rule.then_col)
                    )
        return data

# ─────────────────────────────────────────────────────────────────────────────
# Steps: Scaling
# ─────────────────────────────────────────────────────────────────────────────

class PolarsStandardScalerStep(AbstractStep[pl.DataFrame]):
    """Apply Z-score standardization to numeric columns."""

    def __init__(self, data_frame: pl.DataFrame) -> None:
        super().__init__(data_frame)

    def process(self, data: pl.DataFrame) -> pl.DataFrame:
        numeric_cols = [c for c in data.columns if data[c].dtype.is_numeric()]
        if not numeric_cols:
            return data

        exprs = []
        for col in data.columns:
            if col in numeric_cols:
                # [COLLECT: needed for mean/std scalars]
                mean_val = data[col].mean()
                std_val = data[col].std()
                if std_val is not None and std_val != 0:
                    exprs.append(
                        ((pl.col(col) - mean_val) / std_val).alias(col)
                    )
                else:
                    exprs.append(pl.col(col))
            else:
                exprs.append(pl.col(col))
        return data.select(exprs)
