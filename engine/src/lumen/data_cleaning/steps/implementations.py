"""
DEPRECATED: use data_cleaning/steps/backends/pandas_impl.py via DataCleaningStepFactory.

data_cleaning/steps/implementations.py

Production-ready step implementations for the data cleaning pipeline.

Design principles applied:
    - SOLID (all five) — see Key Improvements section for specifics.
    - Decorator pattern  : ColumnScopedStep wraps any BaseStep.
    - Strategy pattern   : ImputationStrategy enum decouples algorithm selection.
    - Template Method    : BaseStep defines the process() contract.
    - Value Objects      : DomainBounds and CrossColumnRule as frozen dataclasses.
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
import pandas as pd

from lumen.data_cleaning.steps.base import BaseStep

# ─────────────────────────────────────────────────────────────────────────────
# Domain Types  (Value Objects + Strategy Enums)
# ─────────────────────────────────────────────────────────────────────────────

class ImputationStrategy(str, Enum):
    """Supported strategies for filling missing numeric values."""
    MEAN   = "mean"
    MEDIAN = "median"
    MODE   = "mode"

class CategoricalImputationStrategy(str, Enum):
    """Supported strategies for filling missing categorical values."""
    MODE  = "mode"
    FIXED = "fixed"

@dataclass(frozen=True)
class DomainBounds:
    """
    Immutable descriptor for domain validation on a single column.

    Attributes:
        lower: Minimum allowed value (inclusive). None means unbounded.
        upper: Maximum allowed value (inclusive). None means unbounded.

    Example:
        DomainBounds(lower=0, upper=120)  # valid age range
    """
    lower: float | None = None
    upper: float | None = None

@dataclass(frozen=True)
class CrossColumnRule:
    """
    Immutable descriptor for a cross-column validation rule.

    Attributes:
        if_col:   Column whose value is evaluated as the condition.
        equals:   Value that triggers the rule.
        then_col: Column affected when the condition is met.
        action:   Action to perform. Currently supports 'set_nan'.

    Example:
        CrossColumnRule(if_col="status", equals="inactive",
                        then_col="salary", action="set_nan")
    """
    if_col:   str
    equals:   object
    then_col: str
    action:   str = "set_nan"

# ─────────────────────────────────────────────────────────────────────────────
# Utility: Column Type Detection  (ISP — grouped by responsibility)
# ─────────────────────────────────────────────────────────────────────────────

class ColumnInspector:
    """
    Static utility class for inferring column semantics from a DataFrame.

    All methods are pure functions — stateless, side-effect free, and
    independently testable. Consumers depend only on the methods they need (ISP).
    """

    _BOOL_VOCABULARY: Final[frozenset[str]] = frozenset(
        {"yes", "no", "true", "false", "y", "n", "1", "0"}
    )
    _NUMERIC_DOMINANCE_THRESHOLD: Final[float] = 0.7
    _DATE_SAMPLE_SIZE: Final[int] = 10
    _DATE_HIT_RATIO:   Final[float] = 0.5
    _MIN_VALID_DATE:   Final[pd.Timestamp] = pd.Timestamp("1900-01-01")

    @staticmethod
    def infer_column_type(series: pd.Series) -> str:
        """
        Infer whether a column should be treated as 'numeric' or 'categorical'.

        A column is classified as numeric when >70% of its non-null string
        representations parse as valid numbers.

        Args:
            series: The pandas Series to inspect.

        Returns:
            'numeric' if the numeric dominance threshold is exceeded,
            otherwise 'categorical'.
        """
        non_null = series.dropna()
        total = len(non_null)
        if total == 0:
            return "categorical"

        numeric_count: int = non_null.apply(
            lambda x: str(x).replace(".", "", 1).lstrip("-").isdigit()
        ).sum()

        return (
            "numeric"
            if (numeric_count / total) > ColumnInspector._NUMERIC_DOMINANCE_THRESHOLD
            else "categorical"
        )

    @classmethod
    def detect_bool_columns(cls, df: pd.DataFrame) -> list[str]:
        """
        Auto-detect columns whose non-null unique values are a strict subset of
        the boolean vocabulary {yes, no, true, false, y, n, 1, 0}.

        Args:
            df: DataFrame to inspect.

        Returns:
            List of column names identified as boolean.
        """
        bool_cols: list[str] = []

        for col in df.columns:  # ← Bug fix: was `for columns in df.columns`
            if not (
                pd.api.types.is_string_dtype(df[col])
                or pd.api.types.is_object_dtype(df[col])
                or pd.api.types.is_bool_dtype(df[col])
            ):
                continue

            unique_vals = df[col].dropna().astype(str).str.lower().str.strip().unique()
            val_set = set(unique_vals) - {"nan", "<na>", "none", ""}

            if val_set and val_set.issubset(cls._BOOL_VOCABULARY):
                bool_cols.append(col)

        return bool_cols

    @classmethod
    def detect_date_columns(cls, df: pd.DataFrame) -> list[str]:
        """
        Auto-detect columns already typed as datetime64, or object columns
        where ≥50% of a sampled subset parse as valid dates in [1900, now].

        Args:
            df: DataFrame to inspect.

        Returns:
            List of column names identified as date/datetime.
        """
        date_cols: list[str] = []
        max_date = pd.Timestamp.now()

        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                date_cols.append(col)
                continue

            if df[col].dtype != object:
                continue

            sample = df[col].dropna().astype(str).head(cls._DATE_SAMPLE_SIZE)
            if sample.empty:
                continue

            hit_count = sum(
                1 for val in sample
                if cls._is_valid_date_string(val, max_date)
            )
            if hit_count / len(sample) >= cls._DATE_HIT_RATIO:
                date_cols.append(col)

        return date_cols

    @staticmethod
    def detect_numeric_columns(
        df: pd.DataFrame,
        exclude_id_columns: bool = True,
    ) -> list[str]:
        """
        Return native numeric columns plus object columns inferred as numeric.

        Args:
            df: DataFrame to inspect.
            exclude_id_columns: When True, drops 'id' and '*_id' columns
                (configurable instead of hardcoded).

        Returns:
            List of inferred numeric column names.
        """
        numeric_cols: list[str] = list(df.select_dtypes(include=[np.number]).columns)

        for col in df.select_dtypes(include="object").columns:
            if ColumnInspector.infer_column_type(df[col]) == "numeric":
                numeric_cols.append(col)

        if exclude_id_columns:
            numeric_cols = [
                c for c in numeric_cols
                if not c.endswith("_id") and c != "id"
            ]

        return numeric_cols

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _is_valid_date_string(value: str, max_date: pd.Timestamp) -> bool:
        """Return True if *value* parses as a date within [1900, max_date]."""
        try:
            parsed = pd.to_datetime(value, errors="raise")
            return pd.Timestamp("1900-01-01") <= parsed <= max_date
        except (ValueError, TypeError):
            return False

# ─────────────────────────────────────────────────────────────────────────────
# Utility: Value Conversion  (SRP — atomic value-level conversions only)
# ─────────────────────────────────────────────────────────────────────────────

class ValueParser:
    """
    Static utility class for atomic, cell-level type conversions.

    Each method is a pure function: deterministic, no side effects, and
    independently testable. Responsibilities are strictly limited to
    text→number, currency→float, and string→date conversions.
    """

    _WORDS_TO_NUM: Final[dict[str, int]] = {
        "zero": 0,   "one": 1,    "two": 2,     "three": 3,  "four": 4,
        "five": 5,   "six": 6,    "seven": 7,   "eight": 8,  "nine": 9,
        "ten": 10,   "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
        "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100,
    }
    _CURRENCY_PATTERN: Final[re.Pattern] = re.compile(r"[\$€£\s]")
    _MIN_VALID_DATE:   Final[pd.Timestamp] = pd.Timestamp("1900-01-01")

    @classmethod
    def text_to_number(cls, value: str) -> int | str:
        """
        Convert a written-out English number word to its integer equivalent.

        Args:
            value: A string that may represent a number word (e.g. 'thirty').

        Returns:
            Integer if the word is in the lookup table, otherwise the original string.
        """
        return cls._WORDS_TO_NUM.get(str(value).strip().lower(), value)

    @classmethod
    def detect_numeric(cls, value: object) -> float | object:
        """
        Strip currency symbols and attempt a float conversion.

        Args:
            value: A value that may be a numeric string with currency symbols.

        Returns:
            Float on success, otherwise the original value unchanged.
        """
        if isinstance(value, (int, float)):
            return value
        if isinstance(value, str):
            cleaned = cls._CURRENCY_PATTERN.sub("", value).replace(",", ".")
            try:
                return float(cleaned)
            except ValueError:
                pass
        return value

    @classmethod
    def smart_date_parse(cls, value: object) -> pd.Timestamp | object:
        """
        Parse a string as a date if it falls within [1900, now].

        Purely numeric strings (e.g. IDs) are explicitly excluded to
        avoid false positives.

        Args:
            value: A value that may represent a date string.

        Returns:
            pd.Timestamp on success within the valid range, otherwise original value.
        """
        if not isinstance(value, str):
            return value
        # Skip purely numeric strings (avoid mis-parsing IDs as dates)
        if value.replace(".", "").replace(",", "").isdigit():
            return value
        try:
            parsed = pd.to_datetime(value, errors="raise")
            if cls._MIN_VALID_DATE <= parsed <= pd.Timestamp.now():
                return parsed
        except (ValueError, TypeError):
            pass
        return value

# ─────────────────────────────────────────────────────────────────────────────
# Decorator Step: Column Scope  (LSP fix — now a proper BaseStep)
# ─────────────────────────────────────────────────────────────────────────────

class ColumnScopedStep(BaseStep):
    """
    Decorator that restricts any BaseStep to a subset of columns, leaving
    all other columns in the DataFrame unchanged.

    Implements the Decorator pattern over BaseStep. By also inheriting from
    BaseStep, this class satisfies LSP — it is a valid substitution wherever
    a BaseStep is expected.

    Example:
        inner = FixNumericColumnsStep(df)
        scoped = ColumnScopedStep(inner, df, columns=["age", "salary"])
        result = scoped.process(df)
    """

    def __init__(
        self,
        inner_step: BaseStep,
        data_frame: pd.DataFrame,
        columns: list[str],
    ) -> None:
        """
        Args:
            inner_step:  The BaseStep to apply to the scoped columns.
            data_frame:  Reference DataFrame (forwarded to BaseStep.__init__).
            columns:     Columns to which the inner step will be applied.

        Raises:
            TypeError:  If inner_step is not a BaseStep instance.
            ValueError: If columns is empty.
        """
        super().__init__(data_frame)
        if not isinstance(inner_step, BaseStep):
            raise TypeError(
                f"inner_step must be a BaseStep instance, "
                f"got {type(inner_step).__name__!r}."
            )
        if not columns:
            raise ValueError("columns must be a non-empty list.")

        self._inner_step = inner_step
        self._scoped_columns = columns

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the inner step exclusively to the scoped columns.

        The original column order is fully restored after processing.

        Args:
            data: Input DataFrame.

        Returns:
            DataFrame with inner step applied only to the scoped columns.
        """
        available_cols = [c for c in self._scoped_columns if c in data.columns]
        if not available_cols:
            return data.copy()

        remainder_cols  = [c for c in data.columns if c not in available_cols]
        scoped_subset   = data[available_cols].copy()
        remainder_subset = data[remainder_cols].copy()

        cleaned_subset = self._inner_step.process(scoped_subset)

        result = pd.concat([remainder_subset, cleaned_subset], axis=1)
        return result[data.columns]  # Restore original column order

# ─────────────────────────────────────────────────────────────────────────────
# Steps: Schema & Structure
# ─────────────────────────────────────────────────────────────────────────────

class ColumnsTitlesStep(BaseStep):
    """
    Normalize column names: strip whitespace, lowercase, replace internal
    whitespace sequences with underscores.

    Example: " First Name " → "first_name"
    """

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
            .str.replace(r"\s+", "_", regex=True)
        )
        df.columns = [
            "".join(
                c for c in unicodedata.normalize('NFD', col)
                if unicodedata.category(c) != 'Mn'
            )
            for col in df.columns
        ]
        return df

class EnforceSchemaStep(BaseStep):
    """
    Validate minimum structural requirements for the DataFrame.

    Emits UserWarnings (rather than raising exceptions) so the pipeline
    remains running and callers can decide how to handle violations.
    """

    def __init__(
        self,
        data_frame: pd.DataFrame,
        required_columns: list[str] | None = None,
        min_rows: int = 1,
    ) -> None:
        """
        Args:
            data_frame:        Reference DataFrame for BaseStep.
            required_columns:  Column names that must be present.
            min_rows:          Minimum acceptable row count.
        """
        super().__init__(data_frame)
        self.required_columns: list[str] = required_columns or []
        self.min_rows = min_rows

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        if len(df) < self.min_rows:
            warnings.warn(
                f"DataFrame has {len(df)} row(s); "
                f"expected at least {self.min_rows}.",
                stacklevel=2,
            )

        missing_cols = [c for c in self.required_columns if c not in df.columns]
        if missing_cols:
            warnings.warn(
                f"Missing required columns: {missing_cols}",
                stacklevel=2,
            )

        return df

class DropHighMissingColumnsStep(BaseStep):
    """
    Drop columns where the fraction of NaN values exceeds *threshold*.

    Args:
        threshold: Maximum allowed missing fraction (0–1, default 0.8).
    """

    def __init__(
        self,
        data_frame: pd.DataFrame,
        threshold: float = 0.8,
    ) -> None:
        super().__init__(data_frame)
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(
                f"threshold must be in [0.0, 1.0], got {threshold}."
            )
        self.threshold = threshold

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        missing_fractions = df.isnull().mean()
        cols_to_drop = missing_fractions[missing_fractions > self.threshold].index
        return df.drop(columns=cols_to_drop)

class DropConstantColumnsStep(BaseStep):
    """Drop columns that carry zero information (≤1 unique non-null value)."""

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        constant_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
        return df.drop(columns=constant_cols)

# ─────────────────────────────────────────────────────────────────────────────
# Steps: Missing Values & Sentinel Handling
# ─────────────────────────────────────────────────────────────────────────────

class HandleSentinelValuesStep(BaseStep):
    """
    Replace well-known placeholder strings with np.nan to standardize
    the missing-value representation across the entire pipeline.

    The default sentinel vocabulary is immutable (frozenset). Callers
    may extend it without risk of mutating the class-level default.
    """

    _DEFAULT_SENTINELS: Final[frozenset[str]] = frozenset({
        "unknown", "nan", "none", "null", "n/a", "na",
        "-", "$-", "", "invalid_date", "undefined", "#n/a",
    })

    def __init__(
        self,
        data_frame: pd.DataFrame,
        extra_sentinels: frozenset[str] | None = None,
    ) -> None:
        """
        Args:
            data_frame:       Reference DataFrame for BaseStep.
            extra_sentinels:  Additional strings to treat as NaN.
        """
        super().__init__(data_frame)
        self._sentinels: frozenset[str] = (
            self._DEFAULT_SENTINELS | (extra_sentinels or frozenset())
        )

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        for col in df.columns:
            mask = df[col].astype(str).str.lower().str.strip().isin(self._sentinels)
            df.loc[mask, col] = np.nan
        return df

class ImputeCategoricalStep(BaseStep):
    """
    Impute missing values in categorical/object columns.

    Strategies:
        CategoricalImputationStrategy.MODE  — fill with the most frequent value.
        CategoricalImputationStrategy.FIXED — fill with a constant string.
    """

    def __init__(
        self,
        data_frame: pd.DataFrame,
        columns: list[str] | None = None,
        strategy: CategoricalImputationStrategy = CategoricalImputationStrategy.MODE,
        fill_value: str = "unknown",
    ) -> None:
        """
        Args:
            data_frame:  Reference DataFrame for BaseStep.
            columns:     Columns to impute. Defaults to all object/category cols.
            strategy:    Imputation strategy enum value.
            fill_value:  Constant used when strategy is FIXED.
        """
        super().__init__(data_frame)
        self.columns    = columns
        self.strategy   = strategy
        self.fill_value = fill_value

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        target_cols = (
            self.columns
            if self.columns is not None
            else list(df.select_dtypes(include=["object", "category"]).columns)
        )

        for col in (c for c in target_cols if c in df.columns):
            if self.strategy == CategoricalImputationStrategy.MODE:
                mode_vals = df[col].mode()
                if not mode_vals.empty:
                    df[col] = df[col].fillna(mode_vals.iloc[0])
            else:
                df[col] = df[col].fillna(self.fill_value)

        return df

# ─────────────────────────────────────────────────────────────────────────────
# Steps: Type Conversion & Parsing
# ─────────────────────────────────────────────────────────────────────────────

class SafeConversionStep(BaseStep):
    """
    Attempt numeric / date conversion on columns that contain digit or
    currency characters. Pure-text columns (e.g. gender) are auto-skipped.

    Conversion chain per cell: written number → numeric → date.
    """

    _DIGIT_PATTERN: Final[re.Pattern] = re.compile(r"[\d$€£]")

    def __init__(
        self,
        data_frame: pd.DataFrame,
        columns: list[str] | None = None,
        digit_threshold: float = 0.3,
    ) -> None:
        """
        Args:
            data_frame:       Reference DataFrame for BaseStep.
            columns:          Explicit columns to convert. None triggers
                              auto-detection via digit_threshold.
            digit_threshold:  Minimum fraction of cells matching the digit/
                              currency pattern for auto-selection.
        """
        super().__init__(data_frame)
        self.columns         = columns
        self.digit_threshold = digit_threshold

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        for col in self._resolve_columns(df):
            df[col] = df[col].apply(self._convert_cell)
        return df

    def _resolve_columns(self, df: pd.DataFrame) -> list[str]:
        """Return columns to process, honouring explicit overrides."""
        if self.columns is not None:
            return [c for c in self.columns if c in df.columns]
        return [
            col for col in df.columns
            if df[col].dtype == object
            and df[col]
            .apply(lambda x: bool(self._DIGIT_PATTERN.search(str(x))))
            .mean() > self.digit_threshold
        ]

    @staticmethod
    def _convert_cell(value: object) -> object:
        """
        Apply the full conversion chain to a single cell.

        Chain: NaN passthrough → word-number → numeric → date.

        Args:
            value: Raw cell value.

        Returns:
            Converted value, or the original if no conversion applied.
        """
        if pd.isna(value):
            return value
        value = ValueParser.text_to_number(str(value))
        value = ValueParser.detect_numeric(value)
        return ValueParser.smart_date_parse(value)

class FixDatesColumnsStep(BaseStep):
    """
    Parse string/object columns to datetime64 and invalidate out-of-range
    dates (before 1900 or strictly in the future).
    """

    _MIN_DATE: Final[pd.Timestamp] = pd.Timestamp("1900-01-01")

    def __init__(
        self,
        data_frame: pd.DataFrame,
        columns: list[str] | None = None,
    ) -> None:
        super().__init__(data_frame)
        self.columns = columns

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        max_date = pd.Timestamp.now()
        target_cols = (
            [c for c in self.columns if c in df.columns]
            if self.columns is not None
            else ColumnInspector.detect_date_columns(df)
        )

        for col in target_cols:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            out_of_range = (df[col] < self._MIN_DATE) | (df[col] > max_date)
            df.loc[out_of_range, col] = pd.NaT

        return df

class FixBoolsColumnsStep(BaseStep):
    """Convert text boolean representations to nullable pd.BooleanDtype."""

    _BOOL_MAP: Final[dict[str, bool]] = {
        "y": True,   "n": False,
        "yes": True, "no": False,
        "1": True,   "0": False,
        "true": True, "false": False,
    }

    def __init__(
        self,
        data_frame: pd.DataFrame,
        columns: list[str] | None = None,
    ) -> None:
        super().__init__(data_frame)
        self.columns = columns

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        target_cols = (
            [c for c in self.columns if c in df.columns]
            if self.columns is not None
            else ColumnInspector.detect_bool_columns(df)
        )

        for col in target_cols:
            mapped = df[col].astype(str).str.lower().str.strip().map(self._BOOL_MAP)
            df[col] = mapped.astype("boolean")

        return df

class FixColumnsTypesStep(BaseStep):
    """Cast columns to their final target dtypes after all cleaning is done."""

    def __init__(
        self,
        data_frame: pd.DataFrame,
        numeric_columns: list[str] | None = None,
        bool_columns:    list[str] | None = None,
        date_columns:    list[str] | None = None,
    ) -> None:
        super().__init__(data_frame)
        self.numeric_columns = numeric_columns
        self.bool_columns    = bool_columns
        self.date_columns    = date_columns

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        num_cols = (
            [c for c in self.numeric_columns if c in df.columns]
            if self.numeric_columns is not None
            else list(df.select_dtypes(include=[np.number]).columns)
        )
        bool_cols = (
            [c for c in self.bool_columns if c in df.columns]
            if self.bool_columns is not None
            else list(df.select_dtypes(include=[np.bool_]).columns)
        )
        date_cols = (
            [c for c in self.date_columns if c in df.columns]
            if self.date_columns is not None
            else list(df.select_dtypes(include=["datetime64"]).columns)
        )

        for col in num_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        for col in bool_cols:
            df[col] = df[col].astype(bool)
        for col in date_cols:
            df[col] = pd.to_datetime(df[col], errors="coerce")

        return df

# ─────────────────────────────────────────────────────────────────────────────
# Steps: Numeric Cleaning & Imputation
# ─────────────────────────────────────────────────────────────────────────────

class FixNumericColumnsStep(BaseStep):
    """
    Clean numeric strings (strip currency/whitespace) and impute NaN values
    using a configurable ImputationStrategy.

    Applies the Strategy pattern: algorithm selection is externalized into
    the ImputationStrategy enum, keeping this class Open for new strategies
    without modification (OCP).
    """

    _REPLACE_PATTERNS: Final[dict[str, str]] = {
        r"[\$€£]":   "",
        r"\s+":      "",
        r",":        ".",
        r"[^\d.\-]": "",
    }

    def __init__(
        self,
        data_frame: pd.DataFrame,
        strategy: ImputationStrategy = ImputationStrategy.MEDIAN,
        columns: list[str] | None = None,
    ) -> None:
        """
        Args:
            data_frame:  Reference DataFrame for BaseStep.
            strategy:    Imputation strategy for post-conversion NaN values.
            columns:     Explicit columns to process. Defaults to auto-detection.
        """
        super().__init__(data_frame)
        self.strategy = strategy
        self.columns  = columns

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        target_cols = (
            [c for c in self.columns if c in df.columns]
            if self.columns is not None
            else ColumnInspector.detect_numeric_columns(df)
        )

        for col in target_cols:
            df[col] = self._clean_to_numeric(df[col])

        return self._impute_missing(df, target_cols)

    def _clean_to_numeric(self, series: pd.Series) -> pd.Series:
        """Strip non-numeric characters and coerce to float64."""
        cleaned = series.astype(str)
        for pattern, replacement in self._REPLACE_PATTERNS.items():
            cleaned = cleaned.str.replace(pattern, replacement, regex=True)
        return pd.to_numeric(cleaned, errors="coerce")

    def _impute_missing(
        self, df: pd.DataFrame, cols: list[str]
    ) -> pd.DataFrame:
        """Fill NaN values in *cols* using the configured strategy."""
        for col in cols:
            fill_value = self._compute_fill_value(df[col])
            if fill_value is not None:
                df[col] = df[col].fillna(fill_value)
        return df

    def _compute_fill_value(self, series: pd.Series) -> float | None:
        """
        Compute the scalar fill value for a single Series.

        Args:
            series: Numeric Series (post-coercion).

        Returns:
            Fill value as float, or None if computation is not possible
            (e.g. empty mode).
        """
        match self.strategy:
            case ImputationStrategy.MEAN:
                return float(series.mean())
            case ImputationStrategy.MEDIAN:
                return float(series.median())
            case ImputationStrategy.MODE:
                mode_vals = series.mode()
                return float(mode_vals.iloc[0]) if not mode_vals.empty else None
        return None  # Unreachable but satisfies type checkers

# ─────────────────────────────────────────────────────────────────────────────
# Steps: Outlier Handling  (SRP — one class per strategy, was previously mixed)
# ─────────────────────────────────────────────────────────────────────────────

class IQROutlierStep(BaseStep):
    """
    Clip outliers to the Tukey IQR fence: [Q1 − 1.5·IQR, Q3 + 1.5·IQR].

    Values beyond the fence are clipped (not removed) to preserve row count.
    """

    def __init__(
        self,
        data_frame: pd.DataFrame,
        columns: list[str] | None = None,
    ) -> None:
        super().__init__(data_frame)
        self.columns = columns

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        for col in self._resolve_columns(df):
            numeric = pd.to_numeric(df[col], errors="coerce")
            q1, q3  = numeric.quantile(0.25), numeric.quantile(0.75)
            iqr     = q3 - q1
            df[col] = numeric.clip(lower=q1 - 1.5 * iqr, upper=q3 + 1.5 * iqr)
        return df

    def _resolve_columns(self, df: pd.DataFrame) -> list[str]:
        if self.columns is not None:
            return [c for c in self.columns if c in df.columns]
        return [
            c for c in df.select_dtypes(include=[np.number]).columns
            if not c.endswith("_id") and c != "id"
        ]

class ZScoreOutlierStep(BaseStep):
    """
    Nullify values whose absolute Z-score exceeds *z_threshold*.

    Columns with σ = 0 are skipped (all values identical → no outliers).
    Requires at least 4 valid observations to produce a meaningful Z-score.
    """

    _MIN_OBSERVATIONS: Final[int] = 4

    def __init__(
        self,
        data_frame: pd.DataFrame,
        columns: list[str] | None = None,
        z_threshold: float = 3.0,
    ) -> None:
        """
        Args:
            data_frame:   Reference DataFrame for BaseStep.
            columns:      Columns to inspect. Defaults to all numeric columns.
            z_threshold:  Absolute Z-score beyond which a value becomes NaN.

        Raises:
            ValueError: If z_threshold is not strictly positive.
        """
        super().__init__(data_frame)
        if z_threshold <= 0:
            raise ValueError(
                f"z_threshold must be strictly positive, got {z_threshold}."
            )
        self.columns     = columns
        self.z_threshold = z_threshold

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        target_cols = (
            [c for c in self.columns if c in df.columns]
            if self.columns is not None
            else [
                c for c in df.select_dtypes(include=[np.number]).columns
                if not c.endswith("_id") and c != "id"
            ]
        )

        for col in target_cols:
            valid = df[col].dropna()
            if len(valid) < self._MIN_OBSERVATIONS:
                continue

            mean, std = valid.mean(), valid.std()
            if std == 0:
                continue  # No spread → no outliers by definition

            z_scores = (df[col] - mean).abs() / std
            df.loc[z_scores > self.z_threshold, col] = np.nan

        return df

class CapOutliersStep(BaseStep):
    """
    Winsorize outliers by capping values at configurable percentile bounds.

    Unlike IQR-based clipping, this strategy is driven by rank order,
    making it more robust to heavily skewed distributions.
    """

    def __init__(
        self,
        data_frame: pd.DataFrame,
        columns: list[str] | None = None,
        lower_percentile: float = 0.01,
        upper_percentile: float = 0.99,
    ) -> None:
        """
        Args:
            lower_percentile: Lower cap percentile (e.g. 0.01 = 1st percentile).
            upper_percentile: Upper cap percentile (e.g. 0.99 = 99th percentile).

        Raises:
            ValueError: If percentile bounds are invalid.
        """
        super().__init__(data_frame)
        if not (0.0 <= lower_percentile < upper_percentile <= 1.0):
            raise ValueError(
                f"Require 0 ≤ lower_percentile < upper_percentile ≤ 1, "
                f"got ({lower_percentile}, {upper_percentile})."
            )
        self.columns          = columns
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        target_cols = (
            self.columns
            if self.columns is not None
            else [
                c for c in df.select_dtypes(include=[np.number]).columns
                if not c.endswith("_id") and c != "id"
            ]
        )

        for col in (c for c in target_cols if c in df.columns):
            q_low  = df[col].quantile(self.lower_percentile)
            q_high = df[col].quantile(self.upper_percentile)
            df[col] = df[col].clip(lower=q_low, upper=q_high)

        return df

# ─────────────────────────────────────────────────────────────────────────────
# Steps: Text Normalization
# ─────────────────────────────────────────────────────────────────────────────

class FixNotNumericColumnsStep(BaseStep):
    """
    Normalize text/categorical columns: strip whitespace, lowercase,
    replace internal spaces with underscores.

    Columns inferred as numeric by ColumnInspector are automatically skipped.
    """

    def __init__(
        self,
        data_frame: pd.DataFrame,
        columns: list[str] | None = None,
    ) -> None:
        super().__init__(data_frame)
        self.columns = columns

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        target_cols = (
            [c for c in self.columns if c in df.columns]
            if self.columns is not None
            else [
                c for c in df.select_dtypes(include="object").columns
                if ColumnInspector.infer_column_type(df[c]) == "categorical"
            ]
        )

        for col in target_cols:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.lower()
                .str.replace(r"\s+", "_", regex=True)
                .replace("nan", np.nan)
            )

        return df

class NormalizeCategoriesStep(BaseStep):
    """
    Unify known category variants using a per-column synonym map.

    Useful for harmonizing free-text fields with known aliases
    (e.g. {"gender": {"m": "male", "f": "female"}}).
    """

    def __init__(
        self,
        data_frame: pd.DataFrame,
        mappings: dict[str, dict[str, str]] | None = None,
    ) -> None:
        """
        Args:
            mappings: Per-column replacement dicts.
                      E.g. {"city": {"ny": "new_york", "la": "los_angeles"}}.
        """
        super().__init__(data_frame)
        self.mappings: dict[str, dict[str, str]] = mappings or {}

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        for col, mapping in self.mappings.items():
            if col not in df.columns or not mapping:
                continue
            df[col] = (
                df[col].astype(str).str.lower().str.strip().replace(mapping)
            )
        return df

class TextStandardizationStep(BaseStep):
    """
    Advanced text normalization pipeline per cell:
        1. NFKD Unicode decomposition → strip diacritical marks (é→e, ñ→n).
        2. Remove special characters (preserve word chars, spaces, hyphens).
        3. Collapse repeated whitespace and lowercase.
    """

    _SPECIAL_CHAR_PATTERN: Final[re.Pattern] = re.compile(r"[^\w\s\-]")
    _WHITESPACE_PATTERN:   Final[re.Pattern] = re.compile(r"\s+")

    def __init__(
        self,
        data_frame: pd.DataFrame,
        columns: list[str] | None = None,
    ) -> None:
        super().__init__(data_frame)
        self.columns = columns

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        target_cols = (
            self.columns
            if self.columns is not None
            else list(df.select_dtypes(include="object").columns)
        )

        for col in (c for c in target_cols if c in df.columns):
            df[col] = (
                df[col]
                .astype(str)
                .apply(self._normalize_text)
                .replace("nan", np.nan)
            )

        return df

    def _normalize_text(self, text: str) -> str:
        """
        Normalize a single string value.

        Args:
            text: Raw cell string.

        Returns:
            Cleaned, lowercase string, or the original 'nan' sentinel.
        """
        if text == "nan" or pd.isna(text):
            return text

        # Step 1: Strip diacritical marks via Unicode decomposition
        text = (
            unicodedata.normalize("NFKD", text)
            .encode("ascii", "ignore")
            .decode("utf-8")
        )
        # Step 2: Replace special characters with whitespace
        text = self._SPECIAL_CHAR_PATTERN.sub(" ", text)
        # Step 3: Collapse spaces and normalize case
        return self._WHITESPACE_PATTERN.sub(" ", text).strip().lower()

# ─────────────────────────────────────────────────────────────────────────────
# Steps: Validation
# ─────────────────────────────────────────────────────────────────────────────

class ValidateDomainRulesStep(BaseStep):
    """
    Validate per-column domain constraints, nullifying out-of-range values.

    Uses DomainBounds (frozen dataclass) instead of raw lists for type-safe,
    self-documenting, and immutable rule definitions.

    Example:
        rules = {"age": DomainBounds(lower=0, upper=120)}
    """

    def __init__(
        self,
        data_frame: pd.DataFrame,
        rules: dict[str, DomainBounds] | None = None,
    ) -> None:
        super().__init__(data_frame)
        self.rules: dict[str, DomainBounds] = rules or {}

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        for col, bounds in self.rules.items():
            if col not in df.columns:
                continue

            numeric_col     = pd.to_numeric(df[col], errors="coerce")
            violation_mask  = pd.Series(False, index=df.index)

            if bounds.lower is not None:
                violation_mask |= numeric_col < bounds.lower
            if bounds.upper is not None:
                violation_mask |= numeric_col > bounds.upper

            df.loc[violation_mask, col] = np.nan

        return df

class CrossColumnValidationStep(BaseStep):
    """
    Validate consistency between pairs of related columns using
    CrossColumnRule descriptors (frozen dataclasses).

    Replaces the original untyped list[dict] API with strongly typed,
    immutable rule objects — self-documenting and IDE-friendly.
    """

    def __init__(
        self,
        data_frame: pd.DataFrame,
        rules: list[CrossColumnRule] | None = None,
    ) -> None:
        super().__init__(data_frame)
        self.rules: list[CrossColumnRule] = rules or []

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        for rule in self.rules:
            if rule.if_col not in df.columns or rule.then_col not in df.columns:
                continue

            condition_mask = df[rule.if_col] == rule.equals

            if rule.action == "set_nan":
                df.loc[
                    condition_mask & df[rule.then_col].isna(),
                    rule.if_col,
                ] = np.nan

        return df

# ─────────────────────────────────────────────────────────────────────────────
# Steps: Duplicate Handling
# ─────────────────────────────────────────────────────────────────────────────

class RemoveDuplicatesRowsStep(BaseStep):
    """Drop exact duplicate rows, preserving the first occurrence."""

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.copy().drop_duplicates()

# ─────────────────────────────────────────────────────────────────────────────
# Steps: Quality & Audit
# ─────────────────────────────────────────────────────────────────────────────

class FlagDataQualityStep(BaseStep):
    """
    Append a '_quality_score' column: fraction of non-null fields per row.

    Score = 1.0 − (null_count / total_columns).
    A score of 1.0 indicates a fully complete row.
    """

    _QUALITY_COLUMN: Final[str] = "_quality_score"

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df[self._QUALITY_COLUMN] = 1.0 - (
            df.isnull().sum(axis=1) / len(df.columns)
        )
        return df

class AddAuditColumnsStep(BaseStep):
    """
    Append lineage-tracking columns:
        - '_original_index': Row index before cleaning.
        - '_cleaned_at':     UTC timestamp of this step's execution.
    """

    _ORIGINAL_INDEX_COL: Final[str] = "_original_index"
    _CLEANED_AT_COL:     Final[str] = "_cleaned_at"

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        if self._ORIGINAL_INDEX_COL not in df.columns:
            df[self._ORIGINAL_INDEX_COL] = df.index
        df[self._CLEANED_AT_COL] = datetime.now(tz=timezone.utc)
        return df

# ─────────────────────────────────────────────────────────────────────────────
# Steps: Feature Scaling
# ─────────────────────────────────────────────────────────────────────────────

class StandardScalerStep(BaseStep):
    """
    Z-score normalization: z = (x − μ) / σ.

    Implements scikit-learn-style fit/transform separation to prevent
    data leakage when used in a train/test split context:

        scaler = StandardScalerStep(train_df)
        scaler.fit(train_df)
        train_scaled = scaler.transform(train_df)
        test_scaled  = scaler.transform(test_df)   ← uses train μ/σ, no leakage

    The convenience process() method calls fit() + transform() on the same
    data — appropriate ONLY for training sets.

    Attributes:
        _stats:     Column → {"mean": float, "std": float}, populated by fit().
        _is_fitted: True after fit() has been called at least once.
    """

    def __init__(
        self,
        data_frame: pd.DataFrame,
        columns: list[str] | None = None,
    ) -> None:
        super().__init__(data_frame)
        self.columns    = columns
        self._stats:     dict[str, dict[str, float]] = {}
        self._is_fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        """True if fit() has been called at least once."""
        return self._is_fitted

    def fit(self, data: pd.DataFrame) -> "StandardScalerStep":
        """
        Compute μ and σ from *data* and store them for later transform() calls.

        Args:
            data: Training DataFrame (must contain the target columns).

        Returns:
            self — enables fit(data).transform(other) chaining.
        """
        for col in self._resolve_columns(data):
            self._stats[col] = {
                "mean": float(data[col].mean()),
                "std":  float(data[col].std()),
            }
        self._is_fitted = True
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the fitted scaling parameters to *data*.

        Args:
            data: DataFrame to scale (train or test).

        Returns:
            Scaled copy of the input DataFrame.

        Raises:
            RuntimeError: If called before fit().
        """
        if not self._is_fitted:
            raise RuntimeError(
                "StandardScalerStep.transform() called before fit(). "
                "Call fit(training_data) first."
            )
        df = data.copy()
        for col, stats in self._stats.items():
            if col not in df.columns:
                continue
            mean, std = stats["mean"], stats["std"]
            df[col] = (df[col] - mean) / std if std > 0 else (df[col] - mean)
        return df

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fit on *data* and immediately transform it (training convenience).

        Warning:
            Do NOT call process() on test/validation data.
            Use fit(train).transform(test) to avoid data leakage.

        Args:
            data: Training DataFrame.

        Returns:
            Scaled DataFrame with parameters stored internally.
        """
        return self.fit(data).transform(data)

    def _resolve_columns(self, df: pd.DataFrame) -> list[str]:
        """Return columns to scale, filtered to those present in df."""
        if self.columns is not None:
            return [c for c in self.columns if c in df.columns]
        return list(df.select_dtypes(include=[np.number]).columns)
