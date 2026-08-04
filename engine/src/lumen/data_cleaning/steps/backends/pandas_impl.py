
"""
data_cleaning/steps/backends/pandas_impl.py

BACKEND TRADICIONAL — Implementaciones Pandas de todos los steps de limpieza.

Hereda de los contratos abstractos en abstract_steps.py.
Toda la lógica analítica interna usa pd.DataFrame nativamente.
100% compatible con la lógica original de implementations.py.
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
from typing import Dict, Final, FrozenSet, List, Optional

import numpy as np
import pandas as pd

from lumen.data_cleaning.steps.backends.abstract_steps import (
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

# ─────────────────────────────────────────────────────────────────────────────
# Domain Types  (Value Objects + Strategy Enums)  ← backend-agnostic, reused
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
    """Immutable descriptor for domain validation on a single column."""
    lower: float | None = None
    upper: float | None = None

@dataclass(frozen=True)
class CrossColumnRule:
    """Immutable descriptor for a cross-column validation rule."""
    if_col:   str
    equals:   object
    then_col: str
    action:   str = "set_nan"

# ─────────────────────────────────────────────────────────────────────────────
# Utility: Column Type Detection
# ─────────────────────────────────────────────────────────────────────────────

class ColumnInspector:
    """Static utility for inferring column semantics — stateless, side-effect free."""

    _BOOL_VOCABULARY: Final[FrozenSet[str]] = frozenset(
        {"yes", "no", "true", "false", "y", "n", "1", "0"}
    )
    _NUMERIC_DOMINANCE_THRESHOLD: Final[float] = 0.7
    _DATE_SAMPLE_SIZE: Final[int] = 10
    _DATE_HIT_RATIO:   Final[float] = 0.5

    @staticmethod
    def infer_column_type(series: pd.Series) -> str:
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
        bool_cols: list[str] = []
        for col in df.columns:
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

    @staticmethod
    def _is_valid_date_string(value: str, max_date: pd.Timestamp) -> bool:
        try:
            parsed = pd.to_datetime(value, errors="raise")
            return pd.Timestamp("1900-01-01") <= parsed <= max_date
        except (ValueError, TypeError):
            return False

# ─────────────────────────────────────────────────────────────────────────────
# Utility: Value Conversion
# ─────────────────────────────────────────────────────────────────────────────

class ValueParser:
    """Static utility for atomic, cell-level type conversions."""

    _WORDS_TO_NUM: Final[dict[str, int]] = {
        "zero": 0,   "one": 1,    "two": 2,     "three": 3,  "four": 4,
        "five": 5,   "six": 6,    "seven": 7,   "eight": 8,  "nine": 9,
        "ten": 10,   "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
        "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100,
    }
    _CURRENCY_PATTERN: Final[re.Pattern] = re.compile(r"[\$€£\s]")

    @classmethod
    def text_to_number(cls, value: str) -> int | str:
        return cls._WORDS_TO_NUM.get(str(value).strip().lower(), value)

    @classmethod
    def detect_numeric(cls, value: object) -> float | object:
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
        if not isinstance(value, str):
            return value
        if value.replace(".", "").replace(",", "").isdigit():
            return value
        try:
            parsed = pd.to_datetime(value, errors="raise")
            if pd.Timestamp("1900-01-01") <= parsed <= pd.Timestamp.now():
                return parsed
        except (ValueError, TypeError):
            pass
        return value

# ─────────────────────────────────────────────────────────────────────────────
# Steps: Schema & Structure
# ─────────────────────────────────────────────────────────────────────────────

class ColumnScopedStep(AbstractColumnScopedStep[pd.DataFrame]):
    """Pandas: Decorator restricting any step to a subset of columns."""

    def __init__(
        self,
        inner_step: AbstractColumnScopedStep,
        data_frame: pd.DataFrame,
        columns: List[str],
    ) -> None:
        super().__init__(data_frame)
        if not columns:
            raise ValueError("columns must be a non-empty list.")
        self._inner_step = inner_step
        self._scoped_columns = columns

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        available_cols = [c for c in self._scoped_columns if c in data.columns]
        if not available_cols:
            return data.copy()
        remainder_cols   = [c for c in data.columns if c not in available_cols]
        scoped_subset    = data[available_cols].copy()
        remainder_subset = data[remainder_cols].copy()
        cleaned_subset   = self._inner_step.process(scoped_subset)
        result = pd.concat([remainder_subset, cleaned_subset], axis=1)
        return result[data.columns]

class ColumnsTitlesStep(AbstractColumnsTitlesStep[pd.DataFrame]):
    """Pandas: Normalize column names."""

    def __init__(self, data_frame: pd.DataFrame) -> None:
        super().__init__(data_frame)

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
                c for c in unicodedata.normalize("NFD", col)
                if unicodedata.category(c) != "Mn"
            )
            for col in df.columns
        ]
        return df

class EnforceSchemaStep(AbstractEnforceSchemaStep[pd.DataFrame]):
    """Pandas: Validate minimum structural requirements."""

    def __init__(
        self,
        data_frame: pd.DataFrame,
        required_columns: Optional[List[str]] = None,
        min_rows: int = 1,
    ) -> None:
        super().__init__(data_frame)
        self.required_columns: List[str] = required_columns or []
        self.min_rows = min_rows

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        if len(df) < self.min_rows:
            warnings.warn(
                f"DataFrame has {len(df)} row(s); expected at least {self.min_rows}.",
                stacklevel=2,
            )
        missing_cols = [c for c in self.required_columns if c not in df.columns]
        if missing_cols:
            warnings.warn(f"Missing required columns: {missing_cols}", stacklevel=2)
        return df

class DropHighMissingColumnsStep(AbstractDropHighMissingColumnsStep[pd.DataFrame]):
    """Pandas: Drop columns where null fraction > threshold."""

    def __init__(self, data_frame: pd.DataFrame, threshold: float = 0.8) -> None:
        super().__init__(data_frame)
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"threshold must be in [0.0, 1.0], got {threshold}.")
        self.threshold = threshold

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        missing_fractions = df.isnull().mean()
        cols_to_drop = missing_fractions[missing_fractions > self.threshold].index
        return df.drop(columns=cols_to_drop)

class DropConstantColumnsStep(AbstractDropConstantColumnsStep[pd.DataFrame]):
    """Pandas: Drop columns with ≤1 unique non-null value."""

    def __init__(self, data_frame: pd.DataFrame) -> None:
        super().__init__(data_frame)

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        constant_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
        return df.drop(columns=constant_cols)

# ─────────────────────────────────────────────────────────────────────────────
# Steps: Missing Values & Sentinel Handling
# ─────────────────────────────────────────────────────────────────────────────

class HandleSentinelValuesStep(AbstractHandleSentinelValuesStep[pd.DataFrame]):
    """Pandas: Replace well-known placeholder strings with np.nan."""

    _DEFAULT_SENTINELS: Final[FrozenSet[str]] = frozenset({
        "unknown", "nan", "none", "null", "n/a", "na",
        "-", "$-", "", "invalid_date", "undefined", "#n/a",
    })

    def __init__(
        self,
        data_frame: pd.DataFrame,
        extra_sentinels: Optional[FrozenSet[str]] = None,
    ) -> None:
        super().__init__(data_frame)
        self._sentinels: FrozenSet[str] = (
            self._DEFAULT_SENTINELS | (extra_sentinels or frozenset())
        )

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        for col in df.columns:
            mask = df[col].astype(str).str.lower().str.strip().isin(self._sentinels)
            df.loc[mask, col] = np.nan
        return df

class ImputeCategoricalStep(AbstractImputeCategoricalStep[pd.DataFrame]):
    """Pandas: Impute missing categorical values (mode or fixed)."""

    def __init__(
        self,
        data_frame: pd.DataFrame,
        columns: Optional[List[str]] = None,
        strategy: CategoricalImputationStrategy = CategoricalImputationStrategy.MODE,
        fill_value: str = "unknown",
    ) -> None:
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

class SafeConversionStep(AbstractSafeConversionStep[pd.DataFrame]):
    """Pandas: Attempt numeric/date conversion on string columns."""

    _DIGIT_PATTERN: Final[re.Pattern] = re.compile(r"[\d$€£]")

    def __init__(
        self,
        data_frame: pd.DataFrame,
        columns: Optional[List[str]] = None,
        digit_threshold: float = 0.3,
    ) -> None:
        super().__init__(data_frame)
        self.columns         = columns
        self.digit_threshold = digit_threshold

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        for col in self._resolve_columns(df):
            df[col] = df[col].apply(self._convert_cell)
        return df

    def _resolve_columns(self, df: pd.DataFrame) -> list[str]:
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
        if pd.isna(value):
            return value
        value = ValueParser.text_to_number(str(value))
        value = ValueParser.detect_numeric(value)
        return ValueParser.smart_date_parse(value)

class FixDatesColumnsStep(AbstractFixDatesColumnsStep[pd.DataFrame]):
    """Pandas: Parse string columns to datetime and invalidate out-of-range."""

    _MIN_DATE: Final[pd.Timestamp] = pd.Timestamp("1900-01-01")

    def __init__(
        self,
        data_frame: pd.DataFrame,
        columns: Optional[List[str]] = None,
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

class FixBoolsColumnsStep(AbstractFixBoolsColumnsStep[pd.DataFrame]):
    """Pandas: Convert text boolean representations to nullable pd.BooleanDtype."""

    _BOOL_MAP: Final[dict[str, bool]] = {
        "y": True,   "n": False,
        "yes": True, "no": False,
        "1": True,   "0": False,
        "true": True, "false": False,
    }

    def __init__(
        self,
        data_frame: pd.DataFrame,
        columns: Optional[List[str]] = None,
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

class FixColumnsTypesStep(AbstractFixColumnsTypesStep[pd.DataFrame]):
    """Pandas: Cast columns to their final target dtypes."""

    def __init__(
        self,
        data_frame: pd.DataFrame,
        numeric_columns: Optional[List[str]] = None,
        bool_columns: Optional[List[str]] = None,
        date_columns: Optional[List[str]] = None,
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

class FixNumericColumnsStep(AbstractFixNumericColumnsStep[pd.DataFrame]):
    """Pandas: Clean numeric strings and impute NaN using a configurable strategy."""

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
        columns: Optional[List[str]] = None,
    ) -> None:
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
        cleaned = series.astype(str)
        for pattern, replacement in self._REPLACE_PATTERNS.items():
            cleaned = cleaned.str.replace(pattern, replacement, regex=True)
        return pd.to_numeric(cleaned, errors="coerce")

    def _impute_missing(self, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        for col in cols:
            fill_value = self._compute_fill_value(df[col])
            if fill_value is not None:
                df[col] = df[col].fillna(fill_value)
        return df

    def _compute_fill_value(self, series: pd.Series) -> float | None:
        match self.strategy:
            case ImputationStrategy.MEAN:
                return float(series.mean())
            case ImputationStrategy.MEDIAN:
                return float(series.median())
            case ImputationStrategy.MODE:
                mode_vals = series.mode()
                return float(mode_vals.iloc[0]) if not mode_vals.empty else None
        return None

# ─────────────────────────────────────────────────────────────────────────────
# Steps: Outlier Handling
# ─────────────────────────────────────────────────────────────────────────────

class IQROutlierStep(AbstractIQROutlierStep[pd.DataFrame]):
    """Pandas: Clip outliers to Tukey IQR fence."""

    def __init__(
        self,
        data_frame: pd.DataFrame,
        columns: Optional[List[str]] = None,
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

class ZScoreOutlierStep(AbstractZScoreOutlierStep[pd.DataFrame]):
    """Pandas: Nullify values whose absolute Z-score exceeds z_threshold."""

    _MIN_OBSERVATIONS: Final[int] = 4

    def __init__(
        self,
        data_frame: pd.DataFrame,
        columns: Optional[List[str]] = None,
        z_threshold: float = 3.0,
    ) -> None:
        super().__init__(data_frame)
        if z_threshold <= 0:
            raise ValueError(f"z_threshold must be strictly positive, got {z_threshold}.")
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
                continue
            z_scores = (df[col] - mean).abs() / std
            df.loc[z_scores > self.z_threshold, col] = np.nan
        return df

class CapOutliersStep(AbstractCapOutliersStep[pd.DataFrame]):
    """Pandas: Winsorize outliers at configurable percentile bounds."""

    def __init__(
        self,
        data_frame: pd.DataFrame,
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

class FixNotNumericColumnsStep(AbstractFixNotNumericColumnsStep[pd.DataFrame]):
    """Pandas: Normalize text/categorical columns."""

    def __init__(
        self,
        data_frame: pd.DataFrame,
        columns: Optional[List[str]] = None,
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

class NormalizeCategoriesStep(AbstractNormalizeCategoriesStep[pd.DataFrame]):
    """Pandas: Unify category variants using synonym maps."""

    def __init__(
        self,
        data_frame: pd.DataFrame,
        mappings: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> None:
        super().__init__(data_frame)
        self.mappings: Dict[str, Dict[str, str]] = mappings or {}

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        for col, mapping in self.mappings.items():
            if col not in df.columns or not mapping:
                continue
            df[col] = (
                df[col].astype(str).str.lower().str.strip().replace(mapping)
            )
        return df

class TextStandardizationStep(AbstractTextStandardizationStep[pd.DataFrame]):
    """Pandas: NFKD unicode + special char removal + whitespace collapse."""

    _SPECIAL_CHAR_PATTERN: Final[re.Pattern] = re.compile(r"[^\w\s\-]")
    _WHITESPACE_PATTERN:   Final[re.Pattern] = re.compile(r"\s+")

    def __init__(
        self,
        data_frame: pd.DataFrame,
        columns: Optional[List[str]] = None,
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
        if text == "nan" or pd.isna(text):
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

class ValidateDomainRulesStep(AbstractValidateDomainRulesStep[pd.DataFrame]):
    """Pandas: Nullify per-column values violating domain bounds."""

    def __init__(
        self,
        data_frame: pd.DataFrame,
        rules: Optional[Dict[str, DomainBounds]] = None,
    ) -> None:
        super().__init__(data_frame)
        self.rules: Dict[str, DomainBounds] = rules or {}

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        for col, bounds in self.rules.items():
            if col not in df.columns:
                continue
            numeric_col    = pd.to_numeric(df[col], errors="coerce")
            violation_mask = pd.Series(False, index=df.index)
            if bounds.lower is not None:
                violation_mask |= numeric_col < bounds.lower
            if bounds.upper is not None:
                violation_mask |= numeric_col > bounds.upper
            df.loc[violation_mask, col] = np.nan
        return df

class CrossColumnValidationStep(AbstractCrossColumnValidationStep[pd.DataFrame]):
    """Pandas: Validate consistency between pairs of related columns."""

    def __init__(
        self,
        data_frame: pd.DataFrame,
        rules: Optional[List[CrossColumnRule]] = None,
    ) -> None:
        super().__init__(data_frame)
        self.rules: List[CrossColumnRule] = rules or []

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

class RemoveDuplicatesRowsStep(AbstractRemoveDuplicatesRowsStep[pd.DataFrame]):
    """Pandas: Drop exact duplicate rows."""

    def __init__(self, data_frame: pd.DataFrame) -> None:
        super().__init__(data_frame)

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.copy().drop_duplicates()

# ─────────────────────────────────────────────────────────────────────────────
# Steps: Quality & Audit
# ─────────────────────────────────────────────────────────────────────────────

class FlagDataQualityStep(AbstractFlagDataQualityStep[pd.DataFrame]):
    """Pandas: Append '_quality_score' column (fraction of non-null fields per row)."""

    _QUALITY_COLUMN: Final[str] = "_quality_score"

    def __init__(self, data_frame: pd.DataFrame) -> None:
        super().__init__(data_frame)

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df[self._QUALITY_COLUMN] = 1.0 - (
            df.isnull().sum(axis=1) / len(df.columns)
        )
        return df

class AddAuditColumnsStep(AbstractAddAuditColumnsStep[pd.DataFrame]):
    """Pandas: Append lineage-tracking columns (_original_index, _cleaned_at)."""

    _ORIGINAL_INDEX_COL: Final[str] = "_original_index"
    _CLEANED_AT_COL:     Final[str] = "_cleaned_at"

    def __init__(self, data_frame: pd.DataFrame) -> None:
        super().__init__(data_frame)

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        if self._ORIGINAL_INDEX_COL not in df.columns:
            df[self._ORIGINAL_INDEX_COL] = df.index
        df[self._CLEANED_AT_COL] = datetime.now(tz=timezone.utc)
        return df

# ─────────────────────────────────────────────────────────────────────────────
# Steps: Feature Scaling
# ─────────────────────────────────────────────────────────────────────────────

class StandardScalerStep(AbstractStandardScalerStep[pd.DataFrame]):
    """Pandas: Z-score normalization with fit/transform separation."""

    def __init__(
        self,
        data_frame: pd.DataFrame,
        columns: Optional[List[str]] = None,
    ) -> None:
        super().__init__(data_frame)
        self.columns    = columns
        self._stats:     dict[str, dict[str, float]] = {}
        self._is_fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(self, data: pd.DataFrame) -> "StandardScalerStep":
        for col in self._resolve_columns(data):
            self._stats[col] = {
                "mean": float(data[col].mean()),
                "std":  float(data[col].std()),
            }
        self._is_fitted = True
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
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
        return self.fit(data).transform(data)

    def _resolve_columns(self, df: pd.DataFrame) -> list[str]:
        if self.columns is not None:
            return [c for c in self.columns if c in df.columns]
        return list(df.select_dtypes(include=[np.number]).columns)
