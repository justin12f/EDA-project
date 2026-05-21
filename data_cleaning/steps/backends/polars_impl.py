"""
data_cleaning/steps/backends/polars_impl.py

BACKEND LOCAL-FIRST — Implementaciones Polars de todos los steps de limpieza.

Reglas estrictas:
    - Evaluación perezosa: se recibe y retorna pl.DataFrame (o LazyFrame para
      operaciones que no requieren inspección de schema).
    - Sin .to_pandas() en ningún método de transformación.
    - Sin map_elements / apply de Python, EXCEPTO TextStandardizationStep donde
      la normalización unicode NFKD no tiene equivalente vectorizado en Polars.
    - Scalar aggregations (mean, std, mode) se computan con collect() puntual
      SOLO cuando son necesarias para calcular el valor de imputación.
"""

from __future__ import annotations

import re
import unicodedata
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Final, FrozenSet, List, Optional

import polars as pl

from data_cleaning.steps.backends.abstract_steps import (
    AbstractAddAuditColumnsStep,
    AbstractCapOutliersStep,
    AbstractColumnScopedStep,
    AbstractColumnsTitlesStep,
    AbstractCrossColumnValidationStep,
    AbstractDropConstantColumnsStep,
    AbstractDropHighMissingColumnsStep,
    AbstractEnforceSchemaStep,
    AbstractFlagDataQualityStep,
    AbstractFixBoolsColumnsStep,
    AbstractFixColumnsTypesStep,
    AbstractFixDatesColumnsStep,
    AbstractFixNotNumericColumnsStep,
    AbstractFixNumericColumnsStep,
    AbstractHandleSentinelValuesStep,
    AbstractImputeCategoricalStep,
    AbstractIQROutlierStep,
    AbstractNormalizeCategoriesStep,
    AbstractRemoveDuplicatesRowsStep,
    AbstractSafeConversionStep,
    AbstractStandardScalerStep,
    AbstractTextStandardizationStep,
    AbstractValidateDomainRulesStep,
    AbstractZScoreOutlierStep,
)

# Re-export value objects (backend-agnostic)
from data_cleaning.steps.backends.pandas_impl import (
    CategoricalImputationStrategy,
    CrossColumnRule,
    DomainBounds,
    ImputationStrategy,
)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _is_string_dtype(dtype: pl.DataType) -> bool:
    """Return True for Utf8 / String / Categorical Polars dtypes."""
    return dtype in (pl.Utf8, pl.String, pl.Categorical)


def _is_numeric_dtype(dtype: pl.DataType) -> bool:
    """Return True for any numeric Polars dtype."""
    return dtype.is_numeric()


_NUMERIC_DTYPES = (
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Float32, pl.Float64,
)


# ─────────────────────────────────────────────────────────────────────────────
# Steps: Schema & Structure
# ─────────────────────────────────────────────────────────────────────────────

class ColumnScopedStep(AbstractColumnScopedStep[pl.DataFrame]):
    """Polars: Decorator restricting any step to a subset of columns."""

    def __init__(
        self,
        inner_step: AbstractColumnScopedStep,
        data_frame: pl.DataFrame,
        columns: List[str],
    ) -> None:
        super().__init__(data_frame)
        if not columns:
            raise ValueError("columns must be a non-empty list.")
        self._inner_step     = inner_step
        self._scoped_columns = columns

    def process(self, data: pl.DataFrame) -> pl.DataFrame:
        available = [c for c in self._scoped_columns if c in data.columns]
        if not available:
            return data.clone()

        remainder = [c for c in data.columns if c not in available]
        scoped    = data.select(available)
        rest      = data.select(remainder) if remainder else None

        cleaned = self._inner_step.process(scoped)

        if rest is None:
            return cleaned.select(data.columns)

        result = pl.concat([rest, cleaned], how="horizontal")
        return result.select(data.columns)   # Restore original column order


class ColumnsTitlesStep(AbstractColumnsTitlesStep[pl.DataFrame]):
    """Polars: Normalize column names (lowercase, underscored, no diacritics)."""

    def __init__(self, data_frame: pl.DataFrame) -> None:
        super().__init__(data_frame)

    def process(self, data: pl.DataFrame) -> pl.DataFrame:
        rename_map: Dict[str, str] = {}
        for col in data.columns:
            clean = col.strip().lower()
            clean = re.sub(r"\s+", "_", clean)
            clean = "".join(
                c for c in unicodedata.normalize("NFD", clean)
                if unicodedata.category(c) != "Mn"
            )
            rename_map[col] = clean
        return data.rename(rename_map)


class EnforceSchemaStep(AbstractEnforceSchemaStep[pl.DataFrame]):
    """Polars: Validate minimum structural requirements."""

    def __init__(
        self,
        data_frame: pl.DataFrame,
        required_columns: Optional[List[str]] = None,
        min_rows: int = 1,
    ) -> None:
        super().__init__(data_frame)
        self._required = required_columns or []
        self._min_rows = min_rows

    def process(self, data: pl.DataFrame) -> pl.DataFrame:
        if data.height < self._min_rows:
            warnings.warn(
                f"DataFrame has {data.height} row(s); expected at least {self._min_rows}.",
                stacklevel=2,
            )
        missing = [c for c in self._required if c not in data.columns]
        if missing:
            warnings.warn(f"Missing required columns: {missing}", stacklevel=2)
        return data


class DropHighMissingColumnsStep(AbstractDropHighMissingColumnsStep[pl.DataFrame]):
    """Polars: Drop columns where null fraction > threshold."""

    def __init__(self, data_frame: pl.DataFrame, threshold: float = 0.8) -> None:
        super().__init__(data_frame)
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"threshold must be in [0.0, 1.0], got {threshold}.")
        self.threshold = threshold

    def process(self, data: pl.DataFrame) -> pl.DataFrame:
        n = data.height
        if n == 0 or len(data.columns) == 0:
            return data
        null_counts = data.select(pl.all().null_count()).row(0)
        cols_to_drop = [
            col for col, count in zip(data.columns, null_counts)
            if (count / n) > self.threshold
        ]
        return data.drop(cols_to_drop) if cols_to_drop else data


class DropConstantColumnsStep(AbstractDropConstantColumnsStep[pl.DataFrame]):
    """Polars: Drop columns with ≤1 unique non-null value."""

    def __init__(self, data_frame: pl.DataFrame) -> None:
        super().__init__(data_frame)

    def process(self, data: pl.DataFrame) -> pl.DataFrame:
        if len(data.columns) == 0:
            return data
        unique_counts = data.select(pl.all().drop_nulls().n_unique()).row(0)
        constant_cols = [
            col for col, count in zip(data.columns, unique_counts)
            if count <= 1
        ]
        return data.drop(constant_cols) if constant_cols else data


# ─────────────────────────────────────────────────────────────────────────────
# Steps: Missing Values & Sentinel Handling
# ─────────────────────────────────────────────────────────────────────────────

class HandleSentinelValuesStep(AbstractHandleSentinelValuesStep[pl.DataFrame]):
    """Polars: Replace well-known placeholder strings with null."""

    _DEFAULT_SENTINELS: Final[FrozenSet[str]] = frozenset({
        "unknown", "nan", "none", "null", "n/a", "na",
        "-", "$-", "", "invalid_date", "undefined", "#n/a",
    })

    def __init__(
        self,
        data_frame: pl.DataFrame,
        extra_sentinels: Optional[FrozenSet[str]] = None,
    ) -> None:
        super().__init__(data_frame)
        self._sentinels: FrozenSet[str] = (
            self._DEFAULT_SENTINELS | (extra_sentinels or frozenset())
        )

    def process(self, data: pl.DataFrame) -> pl.DataFrame:
        sentinel_list = list(self._sentinels)
        exprs = []
        for col in data.columns:
            if _is_string_dtype(data[col].dtype):
                exprs.append(
                    pl.when(
                        pl.col(col)
                        .str.strip_chars()
                        .str.to_lowercase()
                        .is_in(sentinel_list)
                    )
                    .then(pl.lit(None, dtype=pl.Utf8))
                    .otherwise(pl.col(col))
                    .alias(col)
                )
            else:
                exprs.append(pl.col(col))
        return data.select(exprs)


class ImputeCategoricalStep(AbstractImputeCategoricalStep[pl.DataFrame]):
    """Polars: Impute missing categorical values (mode or fixed)."""

    def __init__(
        self,
        data_frame: pl.DataFrame,
        columns: Optional[List[str]] = None,
        strategy: CategoricalImputationStrategy = CategoricalImputationStrategy.MODE,
        fill_value: str = "unknown",
    ) -> None:
        super().__init__(data_frame)
        self.columns    = columns
        self.strategy   = strategy
        self.fill_value = fill_value

    def process(self, data: pl.DataFrame) -> pl.DataFrame:
        target_cols = (
            [c for c in self.columns if c in data.columns]
            if self.columns is not None
            else [c for c in data.columns if _is_string_dtype(data[c].dtype)]
        )
        df = data
        for col in target_cols:
            if df[col].null_count() == 0:
                continue
            if self.strategy == CategoricalImputationStrategy.MODE:
                # [COLLECT puntual: necesario para obtener escalar del modo]
                mode_series = (
                    df[col].drop_nulls()
                    .value_counts()
                    .sort("count", descending=True)
                    .get_column(col)
                )
                if mode_series.len() > 0:
                    mode_val = mode_series[0]
                    df = df.with_columns(
                        pl.col(col).fill_null(pl.lit(mode_val)).alias(col)
                    )
            else:
                df = df.with_columns(
                    pl.col(col).fill_null(pl.lit(self.fill_value)).alias(col)
                )
        return df


# ─────────────────────────────────────────────────────────────────────────────
# Steps: Type Conversion & Parsing
# ─────────────────────────────────────────────────────────────────────────────

class SafeConversionStep(AbstractSafeConversionStep[pl.DataFrame]):
    """Polars: Attempt numeric/date conversion on string columns.

    Strategy:
        1. Word-to-number replacement via pl.Expr.replace()
        2. Currency/symbol stripping + cast to Float64 (strict=False)
        3. Date parsing via str.to_datetime (strict=False) on remaining strings
    """

    _DIGIT_PATTERN: Final[re.Pattern] = re.compile(r"[\d$€£]")
    _WORDS_TO_NUM: Final[Dict[str, str]] = {
        "zero": "0",   "one": "1",    "two": "2",   "three": "3",  "four": "4",
        "five": "5",   "six": "6",    "seven": "7", "eight": "8",  "nine": "9",
        "ten": "10",   "twenty": "20", "thirty": "30", "forty": "40", "fifty": "50",
        "sixty": "60", "seventy": "70", "eighty": "80", "ninety": "90", "hundred": "100",
    }

    def __init__(
        self,
        data_frame: pl.DataFrame,
        columns: Optional[List[str]] = None,
        digit_threshold: float = 0.3,
    ) -> None:
        super().__init__(data_frame)
        self.columns         = columns
        self.digit_threshold = digit_threshold

    def process(self, data: pl.DataFrame) -> pl.DataFrame:
        cols = self._resolve_columns(data)
        if not cols:
            return data

        df = data
        old_words = list(self._WORDS_TO_NUM.keys())
        new_nums  = list(self._WORDS_TO_NUM.values())

        exprs = []
        for col in cols:
            # Step 1: word → number string replacement
            step1 = (
                pl.col(col).cast(pl.Utf8).str.to_lowercase().str.strip_chars()
                .replace(old_words, new_nums)
            )
            # Step 2: strip currency + cast to Float64
            numeric_cast = (
                step1
                .str.replace_all(r"[\$€£\s]", "")
                .str.replace_all(",", ".")
                .cast(pl.Float64, strict=False)
            )
            exprs.append(numeric_cast.alias(col))

        return data.with_columns(exprs)

    def _resolve_columns(self, data: pl.DataFrame) -> List[str]:
        if self.columns is not None:
            return [c for c in self.columns if c in data.columns]
        # Auto-detect: string columns with > digit_threshold digit-like cells
        result = []
        for col in data.columns:
            if not _is_string_dtype(data[col].dtype):
                continue
            non_null = data[col].drop_nulls()
            if non_null.len() == 0:
                continue
            digit_ratio = (
                non_null.str.contains(r"[\d$€£]").sum() / non_null.len()
            )
            if digit_ratio > self.digit_threshold:
                result.append(col)
        return result


class FixDatesColumnsStep(AbstractFixDatesColumnsStep[pl.DataFrame]):
    """Polars: Parse string columns to datetime and invalidate out-of-range."""

    _MIN_DATE: Final[str] = "1900-01-01"

    def __init__(
        self,
        data_frame: pl.DataFrame,
        columns: Optional[List[str]] = None,
    ) -> None:
        super().__init__(data_frame)
        self.columns = columns

    def process(self, data: pl.DataFrame) -> pl.DataFrame:
        # Auto-detect if no explicit columns given
        if self.columns is not None:
            target_cols = [c for c in self.columns if c in data.columns]
        else:
            target_cols = [
                col for col in data.columns
                if _is_string_dtype(data[col].dtype)
                and self._looks_like_dates(data[col])
            ]

        df = data
        min_dt = pl.lit(self._MIN_DATE).str.to_datetime(format="%Y-%m-%d")
        max_dt = pl.lit(datetime.now().isoformat()).str.to_datetime()

        exprs = []
        for col in target_cols:
            parsed = pl.col(col).str.to_datetime(strict=False)
            exprs.append(
                pl.when((parsed >= min_dt) & (parsed <= max_dt))
                .then(parsed)
                .otherwise(pl.lit(None, dtype=pl.Datetime))
                .alias(col)
            )
        return data.with_columns(exprs) if exprs else data

    @staticmethod
    def _looks_like_dates(series: pl.Series, sample: int = 10) -> bool:
        """Heuristic: try parsing a sample to detect date columns."""
        sample_vals = series.drop_nulls().head(sample)
        if sample_vals.len() == 0:
            return False
        parsed = sample_vals.str.to_datetime(strict=False)
        return parsed.drop_nulls().len() / sample_vals.len() >= 0.5


class FixBoolsColumnsStep(AbstractFixBoolsColumnsStep[pl.DataFrame]):
    """Polars: Convert text boolean representations to Boolean type."""

    _TRUE_VALUES:  Final[FrozenSet[str]] = frozenset({"yes", "y", "true", "1"})
    _FALSE_VALUES: Final[FrozenSet[str]] = frozenset({"no", "n", "false", "0"})
    _BOOL_VOCAB:   Final[FrozenSet[str]] = _TRUE_VALUES | _FALSE_VALUES

    def __init__(
        self,
        data_frame: pl.DataFrame,
        columns: Optional[List[str]] = None,
    ) -> None:
        super().__init__(data_frame)
        self.columns = columns

    def process(self, data: pl.DataFrame) -> pl.DataFrame:
        if self.columns is not None:
            target_cols = [c for c in self.columns if c in data.columns]
        else:
            target_cols = self._auto_detect(data)

        true_list = list(self._TRUE_VALUES)
        df = data
        exprs = []
        for col in target_cols:
            normalized = pl.col(col).cast(pl.Utf8).str.to_lowercase().str.strip_chars()
            exprs.append(
                pl.when(normalized.is_in(true_list))
                .then(pl.lit(True))
                .when(normalized.is_not_null())
                .then(pl.lit(False))
                .otherwise(pl.lit(None, dtype=pl.Boolean))
                .alias(col)
            )
        return data.with_columns(exprs) if exprs else data

    def _auto_detect(self, data: pl.DataFrame) -> List[str]:
        result = []
        for col in data.columns:
            if not _is_string_dtype(data[col].dtype):
                continue
            unique_vals = set(
                data[col].drop_nulls()
                .str.to_lowercase().str.strip_chars()
                .unique().to_list()
            ) - {"nan", "<na>", "none", ""}
            if unique_vals and unique_vals.issubset(self._BOOL_VOCAB):
                result.append(col)
        return result


class FixColumnsTypesStep(AbstractFixColumnsTypesStep[pl.DataFrame]):
    """Polars: Cast columns to their final target dtypes."""

    def __init__(
        self,
        data_frame: pl.DataFrame,
        numeric_columns: Optional[List[str]] = None,
        bool_columns: Optional[List[str]] = None,
        date_columns: Optional[List[str]] = None,
    ) -> None:
        super().__init__(data_frame)
        self.numeric_columns = numeric_columns
        self.bool_columns    = bool_columns
        self.date_columns    = date_columns

    def process(self, data: pl.DataFrame) -> pl.DataFrame:
        schema = data.schema
        num_cols = (
            [c for c in self.numeric_columns if c in data.columns]
            if self.numeric_columns is not None
            else [c for c, dt in schema.items() if dt in _NUMERIC_DTYPES]
        )
        bool_cols = (
            [c for c in self.bool_columns if c in data.columns]
            if self.bool_columns is not None
            else [c for c, dt in schema.items() if dt == pl.Boolean]
        )
        date_cols = (
            [c for c in self.date_columns if c in data.columns]
            if self.date_columns is not None
            else [c for c, dt in schema.items() if dt in (pl.Date, pl.Datetime)]
        )

        exprs = []
        for col in data.columns:
            if col in num_cols:
                exprs.append(pl.col(col).cast(pl.Float64, strict=False).alias(col))
            elif col in bool_cols:
                exprs.append(pl.col(col).cast(pl.Boolean, strict=False).alias(col))
            elif col in date_cols:
                exprs.append(pl.col(col).cast(pl.Datetime, strict=False).alias(col))
            else:
                exprs.append(pl.col(col))
        return data.select(exprs)


# ─────────────────────────────────────────────────────────────────────────────
# Steps: Numeric Cleaning & Imputation
# ─────────────────────────────────────────────────────────────────────────────

class FixNumericColumnsStep(AbstractFixNumericColumnsStep[pl.DataFrame]):
    """Polars: Clean numeric strings and impute NaN using a configurable strategy."""

    _REPLACE_PATTERNS: Final[List[tuple[str, str]]] = [
        (r"[\$€£]",   ""),
        (r"\s+",      ""),
        (r",",        "."),
        (r"[^\d.\-]", ""),
    ]

    def __init__(
        self,
        data_frame: pl.DataFrame,
        strategy: ImputationStrategy = ImputationStrategy.MEDIAN,
        columns: Optional[List[str]] = None,
    ) -> None:
        super().__init__(data_frame)
        self.strategy = strategy
        self.columns  = columns

    def process(self, data: pl.DataFrame) -> pl.DataFrame:
        target_cols = (
            [c for c in self.columns if c in data.columns]
            if self.columns is not None
            else [c for c in data.columns if _is_string_dtype(data[c].dtype) or _is_numeric_dtype(data[c].dtype)]
        )
        # Filter to only non-id numeric-looking columns
        target_cols = [c for c in target_cols if not c.endswith("_id") and c != "id"]

        exprs_clean = []
        for col in target_cols:
            # Clean string representation → cast to float
            cleaned_expr = pl.col(col).cast(pl.Utf8)
            for pattern, replacement in self._REPLACE_PATTERNS:
                cleaned_expr = cleaned_expr.str.replace_all(pattern, replacement)
            cleaned_expr = cleaned_expr.cast(pl.Float64, strict=False)
            exprs_clean.append(cleaned_expr.alias(col))

        if exprs_clean:
            df = data.with_columns(exprs_clean)
        else:
            df = data

        # Impute nulls
        exprs_impute = []
        for col in target_cols:
            if df[col].null_count() == 0:
                continue
            # [COLLECT puntual: escalar de imputación]
            fill_val = self._compute_fill_value(df[col])
            if fill_val is not None:
                exprs_impute.append(pl.col(col).fill_null(pl.lit(fill_val)).alias(col))
                
        return df.with_columns(exprs_impute) if exprs_impute else df

    def _compute_fill_value(self, series: pl.Series) -> float | None:
        non_null = series.drop_nulls()
        if non_null.len() == 0:
            return None
        if self.strategy == ImputationStrategy.MEAN:
            return float(non_null.mean())
        elif self.strategy == ImputationStrategy.MEDIAN:
            return float(non_null.median())
        else:  # MODE
            vc = non_null.value_counts().sort("count", descending=True)
            if vc.height > 0:
                return float(vc.get_column(series.name)[0])
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Steps: Outlier Handling
# ─────────────────────────────────────────────────────────────────────────────

class IQROutlierStep(AbstractIQROutlierStep[pl.DataFrame]):
    """Polars: Clip outliers to Tukey IQR fence [Q1 - 1.5·IQR, Q3 + 1.5·IQR]."""

    def __init__(
        self,
        data_frame: pl.DataFrame,
        columns: Optional[List[str]] = None,
    ) -> None:
        super().__init__(data_frame)
        self.columns = columns

    def process(self, data: pl.DataFrame) -> pl.DataFrame:
        target_cols = self._resolve_columns(data)
        exprs = []
        for col in data.columns:
            if col in target_cols:
                # [COLLECT puntual: cuartiles]
                q1 = data[col].quantile(0.25, interpolation="midpoint")
                q3 = data[col].quantile(0.75, interpolation="midpoint")
                if q1 is not None and q3 is not None:
                    iqr   = q3 - q1
                    lower = q1 - 1.5 * iqr
                    upper = q3 + 1.5 * iqr
                    exprs.append(pl.col(col).clip(lower, upper).alias(col))
                else:
                    exprs.append(pl.col(col))
            else:
                exprs.append(pl.col(col))
        return data.select(exprs)

    def _resolve_columns(self, data: pl.DataFrame) -> List[str]:
        if self.columns is not None:
            return [c for c in self.columns if c in data.columns]
        return [
            c for c in data.columns
            if _is_numeric_dtype(data[c].dtype)
            and not c.endswith("_id") and c != "id"
        ]


class ZScoreOutlierStep(AbstractZScoreOutlierStep[pl.DataFrame]):
    """Polars: Nullify values whose absolute Z-score exceeds z_threshold."""

    _MIN_OBSERVATIONS: Final[int] = 4

    def __init__(
        self,
        data_frame: pl.DataFrame,
        columns: Optional[List[str]] = None,
        z_threshold: float = 3.0,
    ) -> None:
        super().__init__(data_frame)
        if z_threshold <= 0:
            raise ValueError(f"z_threshold must be strictly positive, got {z_threshold}.")
        self.columns     = columns
        self.z_threshold = z_threshold

    def process(self, data: pl.DataFrame) -> pl.DataFrame:
        target_cols = (
            [c for c in self.columns if c in data.columns]
            if self.columns is not None
            else [
                c for c in data.columns
                if _is_numeric_dtype(data[c].dtype)
                and not c.endswith("_id") and c != "id"
            ]
        )
        exprs = []
        for col in data.columns:
            if col not in target_cols:
                exprs.append(pl.col(col))
                continue
            # [COLLECT puntual: mean y std]
            valid = data[col].drop_nulls()
            if valid.len() < self._MIN_OBSERVATIONS:
                exprs.append(pl.col(col))
                continue
            mean = float(valid.mean())
            std  = float(valid.std())
            if std == 0:
                exprs.append(pl.col(col))
                continue
            z_expr = ((pl.col(col) - mean) / std).abs()
            exprs.append(
                pl.when(z_expr > self.z_threshold)
                .then(pl.lit(None, dtype=data[col].dtype))
                .otherwise(pl.col(col))
                .alias(col)
            )
        return data.select(exprs)


class CapOutliersStep(AbstractCapOutliersStep[pl.DataFrame]):
    """Polars: Winsorize outliers at configurable percentile bounds."""

    def __init__(
        self,
        data_frame: pl.DataFrame,
        columns: Optional[List[str]] = None,
        lower_percentile: float = 0.01,
        upper_percentile: float = 0.99,
    ) -> None:
        super().__init__(data_frame)
        if not (0.0 <= lower_percentile < upper_percentile <= 1.0):
            raise ValueError(
                f"Require 0 ≤ lower_percentile < upper_percentile ≤ 1, "
                f"got ({lower_percentile}, {upper_percentile})."
            )
        self.columns          = columns
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile

    def process(self, data: pl.DataFrame) -> pl.DataFrame:
        target_cols = (
            [c for c in self.columns if c in data.columns]
            if self.columns is not None
            else [
                c for c in data.columns
                if _is_numeric_dtype(data[c].dtype)
                and not c.endswith("_id") and c != "id"
            ]
        )
        exprs = []
        for col in data.columns:
            if col in target_cols:
                # [COLLECT puntual: percentiles]
                lo = data[col].quantile(self.lower_percentile, interpolation="midpoint")
                hi = data[col].quantile(self.upper_percentile, interpolation="midpoint")
                if lo is not None and hi is not None:
                    exprs.append(pl.col(col).clip(lo, hi).alias(col))
                else:
                    exprs.append(pl.col(col))
            else:
                exprs.append(pl.col(col))
        return data.select(exprs)


# ─────────────────────────────────────────────────────────────────────────────
# Steps: Text Normalization
# ─────────────────────────────────────────────────────────────────────────────

class FixNotNumericColumnsStep(AbstractFixNotNumericColumnsStep[pl.DataFrame]):
    """Polars: Normalize text/categorical columns (strip, lowercase, underscore)."""

    def __init__(
        self,
        data_frame: pl.DataFrame,
        columns: Optional[List[str]] = None,
    ) -> None:
        super().__init__(data_frame)
        self.columns = columns

    def process(self, data: pl.DataFrame) -> pl.DataFrame:
        if self.columns is not None:
            target_cols = [c for c in self.columns if c in data.columns]
        else:
            target_cols = [c for c in data.columns if _is_string_dtype(data[c].dtype)]

        exprs = []
        for col in data.columns:
            if col in target_cols:
                exprs.append(
                    pl.col(col)
                    .str.strip_chars()
                    .str.to_lowercase()
                    .str.replace_all(r"\s+", "_")
                    .alias(col)
                )
            else:
                exprs.append(pl.col(col))
        return data.select(exprs)


class NormalizeCategoriesStep(AbstractNormalizeCategoriesStep[pl.DataFrame]):
    """Polars: Unify category variants using synonym maps."""

    def __init__(
        self,
        data_frame: pl.DataFrame,
        mappings: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> None:
        super().__init__(data_frame)
        self.mappings: Dict[str, Dict[str, str]] = mappings or {}

    def process(self, data: pl.DataFrame) -> pl.DataFrame:
        exprs = []
        for col, mapping in self.mappings.items():
            if col not in data.columns or not mapping:
                continue
            old_vals = list(mapping.keys())
            new_vals = list(mapping.values())
            exprs.append(
                pl.col(col)
                .str.strip_chars()
                .str.to_lowercase()
                .replace(old_vals, new_vals)
                .alias(col)
            )
        return data.with_columns(exprs) if exprs else data


class TextStandardizationStep(AbstractTextStandardizationStep[pl.DataFrame]):
    """Polars: NFKD unicode + special char removal + whitespace collapse.

    NOTE: NFKD diacritic removal requires map_elements (no vectorized Polars op).
    This is the ONLY place in the Polars backend that uses a Python-level call,
    and it is documented as an accepted exception per architecture decision.
    """

    _SPECIAL_CHAR_PATTERN: Final[re.Pattern] = re.compile(r"[^\w\s\-]")
    _WHITESPACE_PATTERN:   Final[re.Pattern] = re.compile(r"\s+")

    def __init__(
        self,
        data_frame: pl.DataFrame,
        columns: Optional[List[str]] = None,
    ) -> None:
        super().__init__(data_frame)
        self.columns = columns

    def process(self, data: pl.DataFrame) -> pl.DataFrame:
        target_cols = (
            self.columns
            if self.columns is not None
            else [c for c in data.columns if _is_string_dtype(data[c].dtype)]
        )
        exprs = []
        for col in (c for c in target_cols if c in data.columns):
            # map_elements justified: unicodedata.normalize has no Polars equivalent
            exprs.append(
                pl.col(col)
                .map_elements(self._normalize_text, return_dtype=pl.Utf8)
                .alias(col)
            )
        return data.with_columns(exprs) if exprs else data

    def _normalize_text(self, text: str | None) -> str | None:
        if text is None or text == "nan":
            return text
        text = (
            unicodedata.normalize("NFKD", text)
            .encode("ascii", "ignore")
            .decode("utf-8")
        )
        text = self._SPECIAL_CHAR_PATTERN.sub(" ", text)
        return self._WHITESPACE_PATTERN.sub(" ", text).strip().lower()


# ─────────────────────────────────────────────────────────────────────────────
# Steps: Validation
# ─────────────────────────────────────────────────────────────────────────────

class ValidateDomainRulesStep(AbstractValidateDomainRulesStep[pl.DataFrame]):
    """Polars: Nullify per-column values violating domain bounds."""

    def __init__(
        self,
        data_frame: pl.DataFrame,
        rules: Optional[Dict[str, DomainBounds]] = None,
    ) -> None:
        super().__init__(data_frame)
        self.rules: Dict[str, DomainBounds] = rules or {}

    def process(self, data: pl.DataFrame) -> pl.DataFrame:
        exprs = []
        for col in data.columns:
            if col not in self.rules:
                exprs.append(pl.col(col))
                continue
            bounds = self.rules[col]
            dtype  = data[col].dtype
            expr   = pl.col(col).cast(pl.Float64, strict=False)
            if bounds.lower is not None:
                expr = pl.when(expr < bounds.lower).then(pl.lit(None, dtype=dtype)).otherwise(pl.col(col))
            if bounds.upper is not None:
                expr = pl.when(
                    pl.col(col).cast(pl.Float64, strict=False) > bounds.upper
                ).then(pl.lit(None, dtype=dtype)).otherwise(expr)
            exprs.append(expr.alias(col))
        return data.select(exprs)


class CrossColumnValidationStep(AbstractCrossColumnValidationStep[pl.DataFrame]):
    """Polars: Validate consistency between pairs of related columns."""

    def __init__(
        self,
        data_frame: pl.DataFrame,
        rules: Optional[List[CrossColumnRule]] = None,
    ) -> None:
        super().__init__(data_frame)
        self.rules: List[CrossColumnRule] = rules or []

    def process(self, data: pl.DataFrame) -> pl.DataFrame:
        df = data
        for rule in self.rules:
            if rule.if_col not in df.columns or rule.then_col not in df.columns:
                continue
            if rule.action == "set_nan":
                condition = (pl.col(rule.if_col) == pl.lit(rule.equals)) & pl.col(rule.then_col).is_null()
                df = df.with_columns(
                    pl.when(condition)
                    .then(pl.lit(None, dtype=df[rule.if_col].dtype))
                    .otherwise(pl.col(rule.if_col))
                    .alias(rule.if_col)
                )
        return df


# ─────────────────────────────────────────────────────────────────────────────
# Steps: Duplicate Handling
# ─────────────────────────────────────────────────────────────────────────────

class RemoveDuplicatesRowsStep(AbstractRemoveDuplicatesRowsStep[pl.DataFrame]):
    """Polars: Drop exact duplicate rows."""

    def __init__(self, data_frame: pl.DataFrame) -> None:
        super().__init__(data_frame)

    def process(self, data: pl.DataFrame) -> pl.DataFrame:
        return data.unique(maintain_order=True)


# ─────────────────────────────────────────────────────────────────────────────
# Steps: Quality & Audit
# ─────────────────────────────────────────────────────────────────────────────

class FlagDataQualityStep(AbstractFlagDataQualityStep[pl.DataFrame]):
    """Polars: Append '_quality_score' column (fraction of non-null per row)."""

    _QUALITY_COLUMN: Final[str] = "_quality_score"

    def __init__(self, data_frame: pl.DataFrame) -> None:
        super().__init__(data_frame)

    def process(self, data: pl.DataFrame) -> pl.DataFrame:
        n_cols = len(data.columns)
        if n_cols == 0:
            return data
        non_null_expr = pl.sum_horizontal(
            [pl.col(c).is_not_null().cast(pl.Int32) for c in data.columns]
        )
        return data.with_columns(
            (non_null_expr / pl.lit(n_cols)).alias(self._QUALITY_COLUMN)
        )


class AddAuditColumnsStep(AbstractAddAuditColumnsStep[pl.DataFrame]):
    """Polars: Append lineage-tracking columns (_original_index, _cleaned_at)."""

    _ORIGINAL_INDEX_COL: Final[str] = "_original_index"
    _CLEANED_AT_COL:     Final[str] = "_cleaned_at"

    def __init__(self, data_frame: pl.DataFrame) -> None:
        super().__init__(data_frame)

    def process(self, data: pl.DataFrame) -> pl.DataFrame:
        df = data
        if self._ORIGINAL_INDEX_COL not in df.columns:
            df = df.with_columns(
                pl.arange(0, df.height, eager=True).alias(self._ORIGINAL_INDEX_COL)
            )
        df = df.with_columns(
            pl.lit(datetime.now(tz=timezone.utc).isoformat()).alias(self._CLEANED_AT_COL)
        )
        return df


# ─────────────────────────────────────────────────────────────────────────────
# Steps: Feature Scaling
# ─────────────────────────────────────────────────────────────────────────────

class StandardScalerStep(AbstractStandardScalerStep[pl.DataFrame]):
    """Polars: Z-score normalization with fit/transform separation."""

    def __init__(
        self,
        data_frame: pl.DataFrame,
        columns: Optional[List[str]] = None,
    ) -> None:
        super().__init__(data_frame)
        self.columns    = columns
        self._stats:     Dict[str, Dict[str, float]] = {}
        self._is_fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(self, data: pl.DataFrame) -> "StandardScalerStep":
        cols = self._resolve_columns(data)
        if not cols:
            self._is_fitted = True
            return self
            
        exprs = []
        for col in cols:
            exprs.append(pl.col(col).mean().alias(f"{col}__mean"))
            exprs.append(pl.col(col).std().alias(f"{col}__std"))
            
        # [COLLECT puntual: mean y std para escalar, calculado en 1 solo pase]
        stats = data.select(exprs).row(0)
        
        for i, col in enumerate(cols):
            self._stats[col] = {
                "mean": float(stats[i*2] if stats[i*2] is not None else 0.0),
                "std":  float(stats[i*2 + 1] if stats[i*2 + 1] is not None else 0.0),
            }
        self._is_fitted = True
        return self

    def transform(self, data: pl.DataFrame) -> pl.DataFrame:
        if not self._is_fitted:
            raise RuntimeError(
                "StandardScalerStep.transform() called before fit(). "
                "Call fit(training_data) first."
            )
        exprs = []
        for col in data.columns:
            if col in self._stats:
                mean = self._stats[col]["mean"]
                std  = self._stats[col]["std"]
                if std > 0:
                    exprs.append(((pl.col(col) - mean) / std).alias(col))
                else:
                    exprs.append((pl.col(col) - mean).alias(col))
            else:
                exprs.append(pl.col(col))
        return data.select(exprs)

    def process(self, data: pl.DataFrame) -> pl.DataFrame:
        return self.fit(data).transform(data)

    def _resolve_columns(self, data: pl.DataFrame) -> List[str]:
        if self.columns is not None:
            return [c for c in self.columns if c in data.columns]
        return [c for c in data.columns if _is_numeric_dtype(data[c].dtype)]
