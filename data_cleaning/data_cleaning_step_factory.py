"""Module containing abstract base classes and step implementations for data cleaning."""

import re
import unicodedata
from datetime import datetime
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class BaseStep(ABC):
    """Base class for all pipeline steps."""

    def __init__(self, data_frame: pd.DataFrame) -> None:
        self._data_frame = data_frame

    @abstractmethod
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process the DataFrame and return the cleaned version."""


# ---------------------------------------------------------------------------
# Auto-detection helpers
# ---------------------------------------------------------------------------


def text_to_number(value_str: str):
    """Convert written-out English numbers to integers (e.g. 'thirty' → 30)."""
    words_to_num = {
        "zero": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "twenty": 20,
        "thirty": 30,
        "forty": 40,
        "fifty": 50,
        "sixty": 60,
        "seventy": 70,
        "eighty": 80,
        "ninety": 90,
        "hundred": 100,
    }
    cleaned = str(value_str).strip().lower()
    return words_to_num.get(cleaned, value_str)


def detect_numeric(value):
    """Strip currency symbols and try to convert to float."""
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        cleaned = re.sub(r"[\$€£\s]", "", value)
        cleaned = cleaned.replace(",", ".")
        try:
            return float(cleaned)
        except ValueError:
            pass
    return value


def smart_date_parse(value):
    """Try to parse a string as a date. Returns Timestamp or original value."""
    if isinstance(value, str) and not value.replace(".", "").replace(",", "").isdigit():
        try:
            parsed = pd.to_datetime(value, errors="raise")
            # Reject dates before 1900 or in the far future
            if pd.Timestamp("1900-01-01") <= parsed <= pd.Timestamp.now():
                return parsed
        except (ValueError, TypeError):
            pass
    return value


def infer_column_type(series: pd.Series) -> str:
    """
    Infer whether a column should be treated as numeric or categorical
    based on its string-represented values.
    """
    numeric_count = (
        series.dropna()
        .apply(lambda x: str(x).replace(".", "", 1).lstrip("-").isdigit())
        .sum()
    )
    total = len(series.dropna())
    if total == 0:
        return "categorical"
    return "numeric" if (numeric_count / total) > 0.7 else "categorical"


def detect_bool_columns(df: pd.DataFrame) -> list[str]:
    """
    Auto-detect columns whose non-null unique values are a subset of
    common boolean representations (yes/no, true/false, 1/0, y/n).
    """
    bool_vocab = {"yes", "no", "true", "false", "y", "n", "1", "0"}
    bool_cols = []
    for columns in df.columns:
        if df[columns].dtype == object or df[columns].dtype == bool:
            unique_vals = (
                df[columns].dropna().astype(str).str.lower().str.strip().unique()
            )
            if len(unique_vals) > 0 and set(unique_vals).issubset(bool_vocab):
                bool_cols.append(columns)
    return bool_cols


def detect_date_columns(df: pd.DataFrame) -> list[str]:
    """
    Auto-detect columns that are already datetime64, or object columns
    where > 50 % of non-null sample values parse as valid dates.
    """
    date_cols = []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            date_cols.append(col)
        elif df[col].dtype == object:
            sample = df[col].dropna().astype(str).head(10)
            if len(sample) == 0:
                continue
            hits = 0
            for val in sample:
                try:
                    parsed = pd.to_datetime(val, errors="raise")
                    if pd.Timestamp("1900-01-01") <= parsed <= pd.Timestamp.now():
                        hits += 1
                except (ValueError, TypeError):
                    pass
            if hits / len(sample) >= 0.5:
                date_cols.append(col)
    return date_cols


def detect_numeric_columns(df: pd.DataFrame) -> list[str]:
    """
    Return numeric (float64/int64) columns PLUS object columns where
    infer_column_type() says 'numeric'.
    """
    num_cols = list(df.select_dtypes(include=[np.number]).columns)
    for col in df.select_dtypes(include="object").columns:
        if infer_column_type(df[col]) == "numeric":
            num_cols.append(col)
    return num_cols


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


class ColumnsTitlesStep(BaseStep):
    """Normalise column names: strip whitespace, lowercase, underscores."""

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df.columns = (
            df.columns.str.strip().str.lower().str.replace(r"\s+", "_", regex=True)
        )
        return df


class HandleSentinelValuesStep(BaseStep):
    """Replace known placeholder values with NaN."""

    DEFAULT_SENTINELS: set[str] = {
        "unknown",
        "nan",
        "none",
        "null",
        "n/a",
        "na",
        "-",
        "$-",
        "",
        "invalid_date",
        "undefined",
        "#n/a",
    }

    def __init__(
        self,
        data_frame: pd.DataFrame,
        extra_sentinels: set[str] | None = None,
    ) -> None:
        super().__init__(data_frame)
        self.sentinels = self.DEFAULT_SENTINELS | (extra_sentinels or set())

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        for col in df.columns:
            mask = df[col].astype(str).str.lower().str.strip().isin(self.sentinels)
            df.loc[mask, col] = np.nan
        return df


class NormalizeCategoriesStep(BaseStep):
    """
    Unify category variants using a per-column synonym map.
    If mappings is empty or None, the step passes through unchanged.
    """

    def __init__(
        self,
        data_frame: pd.DataFrame,
        mappings: dict[str, dict[str, str]] | None = None,
    ) -> None:
        super().__init__(data_frame)
        self.mappings = mappings or {}

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        for column, mapping in self.mappings.items():
            if column in data.columns and mapping:
                data[column] = (
                    data[column].astype(str).str.lower().str.strip().replace(mapping)
                )
        return data


class SafeConversionStep(BaseStep):
    """
    Attempt numeric / date conversion on columns that actually contain
    digits or currency symbols. Pure-text columns (e.g. gender) are skipped.
    """

    def __init__(
        self,
        data_frame: pd.DataFrame,
        columns: list[str] | None = None,
        digit_threshold: float = 0.3,
    ) -> None:
        super().__init__(data_frame)
        self.columns = columns
        self.digit_threshold = digit_threshold

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        if self.columns is not None:
            columns_to_convert = [c for c in self.columns if c in data.columns]
        else:
            columns_to_convert = [
                c
                for c in data.columns
                if data[c].dtype == object
                and data[c].apply(lambda x: bool(re.search(r"[\d$€£]", str(x)))).mean()
                > self.digit_threshold
            ]
        for column in columns_to_convert:
            data[column] = data[column].apply(self._safe_convert)
        return data

    def _safe_convert(self, value):
        if pd.isna(value):
            return value
        value = text_to_number(str(value))
        value = detect_numeric(value)
        parsed_date = smart_date_parse(value)
        if isinstance(parsed_date, pd.Timestamp):
            return parsed_date
        return value


class FixNotNumericColumnsStep(BaseStep):
    """Normalise text columns: strip, lowercase, replace spaces with underscores."""

    def __init__(
        self,
        data_frame: pd.DataFrame,
        columns: list[str] | None = None,
    ) -> None:
        super().__init__(data_frame)
        self.columns = columns

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        if self.columns is not None:
            cols = [c for c in self.columns if c in df.columns]
        else:
            # Only process object columns that infer_column_type says are categorical
            cols = [
                c
                for c in df.select_dtypes(include="object").columns
                if infer_column_type(df[c]) == "categorical"
            ]
        for col in cols:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.lower()
                .str.replace(r"\s+", "_", regex=True)
                .replace("nan", np.nan)
            )
        return df


class RemoveDuplicatesRowsStep(BaseStep):
    """Drop repeated rows."""

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.copy().drop_duplicates()


class ValidateDomainRulesStep(BaseStep):
    """
    Validate domain constraints per column.
    Values outside [lo, hi] → NaN. Pass None for an unbounded side.
    If rules is empty, the step is a no-op.
    """

    def __init__(
        self,
        data_frame: pd.DataFrame,
        rules: dict[str, list] | None = None,
    ) -> None:
        super().__init__(data_frame)
        self.rules = rules or {}

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        for col, bounds in self.rules.items():
            if col not in df.columns:
                continue
            lo, hi = bounds[0], bounds[1] if len(bounds) > 1 else None
            numeric_col = pd.to_numeric(df[col], errors="coerce")
            mask = pd.Series(False, index=df.index)
            if lo is not None:
                mask |= numeric_col < lo
            if hi is not None:
                mask |= numeric_col > hi
            df.loc[mask, col] = np.nan
        return df


class FixNumericColumnsStep(BaseStep):
    """Clean numeric strings and impute missing values (default: median)."""

    def __init__(
        self,
        data_frame: pd.DataFrame,
        fixcase: str = "median",
        columns: list[str] | None = None,
    ) -> None:
        super().__init__(data_frame)
        self._fix_case = fixcase
        self.columns = columns

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        if self.columns is not None:
            cols = [c for c in self.columns if c in df.columns]
        else:
            cols = detect_numeric_columns(df)

        replace_map: dict[str, str] = {
            r"[\$€£]": "",
            r"\s+": "",
            r",": ".",
            r"[^\d.\-]": "",
        }
        for col in cols:
            df[col] = df[col].astype(str).replace(replace_map, regex=True)
            df[col] = pd.to_numeric(df[col], errors="coerce")

        match self._fix_case:
            case "mean":
                for col in cols:
                    df[col] = df[col].fillna(float(df[col].mean()))
            case "median":
                for col in cols:
                    df[col] = df[col].fillna(float(df[col].median()))
            case "mode":
                for col in cols:
                    if not df[col].mode().empty:
                        df[col] = df[col].fillna(float(df[col].mode()[0]))
            case _:
                raise ValueError(f"Invalid fixcase: '{self._fix_case}'")
        return df


class HandleOutliersStep(BaseStep):
    """
    Detect and clip outliers using IQR.
    Additionally, values with |Z-score| > z_threshold → NaN.
    """

    def __init__(
        self,
        data_frame: pd.DataFrame,
        columns: list[str] | None = None,
        z_threshold: float = 3.0,
    ) -> None:
        super().__init__(data_frame)
        self.columns = columns
        self.z_threshold = z_threshold

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        if self.columns is not None:
            cols = [c for c in self.columns if c in df.columns]
        else:
            cols = list(df.select_dtypes(include=[np.number]).columns)

        for col in cols:
            numeric = pd.to_numeric(df[col], errors="coerce")
            q1 = numeric.quantile(0.25)
            q3 = numeric.quantile(0.75)
            iqr = q3 - q1
            df[col] = numeric.clip(lower=q1 - 1.5 * iqr, upper=q3 + 1.5 * iqr)

            # Z-score for extreme outliers
            valid = df[col].dropna()
            if len(valid) > 3:
                mean, std = valid.mean(), valid.std()
                if std > 0:
                    z_scores = (df[col] - mean).abs() / std
                    df.loc[z_scores > self.z_threshold, col] = np.nan
        return df


class FixBoolsColumnsStep(BaseStep):
    """Convert text boolean representations to actual bool dtype."""

    BOOL_MAP: dict[str, bool] = {
        "y": True,
        "n": False,
        "yes": True,
        "no": False,
        "1": True,
        "0": False,
        "true": True,
        "false": False,
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
        if self.columns is not None:
            cols = [c for c in self.columns if c in df.columns]
        else:
            cols = detect_bool_columns(df)

        for col in cols:
            df[col] = (
                df[col]
                .astype(str)
                .str.lower()
                .str.strip()
                .replace(self.BOOL_MAP)
                .astype(bool)
            )
        return df


class FixDatesColumnsStep(BaseStep):
    """Parse date columns and invalidate impossible dates (before 1900 or future)."""

    MIN_DATE = pd.Timestamp("1900-01-01")

    def __init__(
        self,
        data_frame: pd.DataFrame,
        columns: list[str] | None = None,
    ) -> None:
        super().__init__(data_frame)
        self.columns = columns

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        if self.columns is not None:
            cols = [c for c in self.columns if c in df.columns]
        else:
            cols = detect_date_columns(df)

        max_date = pd.Timestamp.now()
        for col in cols:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            invalid = (df[col] < self.MIN_DATE) | (df[col] > max_date)
            df.loc[invalid, col] = pd.NaT
        return df


class CrossColumnValidationStep(BaseStep):
    """
    Validate consistency between related columns.
    Each rule: {if_col, equals, then_col, action='set_nan'}
    """

    def __init__(
        self,
        data_frame: pd.DataFrame,
        rules: list[dict] | None = None,
    ) -> None:
        super().__init__(data_frame)
        self.rules = rules or []

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        for rule in self.rules:
            cond_col = rule.get("if_col")
            cond_val = rule.get("equals")
            target_col = rule.get("then_col")
            action = rule.get("action")
            if cond_col in df.columns and target_col in df.columns:
                mask = df[cond_col] == cond_val
                if action == "set_nan":
                    df.loc[mask & df[target_col].isna(), cond_col] = np.nan
        return df


class FlagDataQualityStep(BaseStep):
    """
    Add a '_quality_score' column: fraction of non-null fields per row.
    Score 1.0 = no missing values.
    """

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df["_quality_score"] = 1.0 - (df.isnull().sum(axis=1) / len(df.columns))
        return df


class FixColumnsTypesStep(BaseStep):
    """Cast columns to their final target dtypes."""

    def __init__(
        self,
        data_frame: pd.DataFrame,
        numeric_columns: list[str] | None = None,
        bool_columns: list[str] | None = None,
        date_columns: list[str] | None = None,
    ) -> None:
        super().__init__(data_frame)
        self.numeric_columns = numeric_columns
        self.bool_columns = bool_columns
        self.date_columns = date_columns

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


class DropHighMissingColumnsStep(BaseStep):
    """Drop columns with more than `threshold` fraction of missing values."""

    def __init__(
        self,
        data_frame: pd.DataFrame,
        threshold: float = 0.8,
    ) -> None:
        super().__init__(data_frame)
        self.threshold = threshold

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        missing_frac = df.isnull().mean()
        cols_to_drop = missing_frac[missing_frac > self.threshold].index
        if len(cols_to_drop) > 0:
            df = df.drop(columns=cols_to_drop)
        return df


class DropConstantColumnsStep(BaseStep):
    """Drop columns that have only 1 unique value (or 0) across all valid rows."""

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        cols_to_drop = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
        return df


class TextStandardizationStep(BaseStep):
    """Advanced text normalization: strip accents, weird chars, double spaces."""

    def __init__(
        self,
        data_frame: pd.DataFrame,
        columns: list[str] | None = None,
    ) -> None:
        super().__init__(data_frame)
        self.columns = columns

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        cols = (
            self.columns
            if self.columns is not None
            else list(df.select_dtypes(include="object").columns)
        )
        for col in cols:
            df[col] = (
                df[col]
                .astype(str)
                .apply(self._standardize_string)
                .replace("nan", np.nan)
            )
        return df

    def _standardize_string(self, text: str) -> str:
        if pd.isna(text) or text == "nan":
            return text
        text = str(text)
        # remove accents
        text = (
            unicodedata.normalize("NFKD", text)
            .encode("ascii", "ignore")
            .decode("utf-8")
        )
        # replace special chars with space (except underscores/hyphens used in words)
        text = re.sub(r"[^\w\s\-]", " ", text)
        # remove double spaces
        text = re.sub(r"\s+", " ", text).strip().lower()
        return text


class CapOutliersStep(BaseStep):
    """Winsorize outliers (cap at percentiles) instead of turning them to NaN."""

    def __init__(
        self,
        data_frame: pd.DataFrame,
        columns: list[str] | None = None,
        lower_percentile: float = 0.01,
        upper_percentile: float = 0.99,
    ) -> None:
        super().__init__(data_frame)
        self.columns = columns
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        cols = (
            self.columns
            if self.columns is not None
            else list(df.select_dtypes(include=[np.number]).columns)
        )
        for col in cols:
            q_low = df[col].quantile(self.lower_percentile)
            q_high = df[col].quantile(self.upper_percentile)
            df[col] = df[col].clip(lower=q_low, upper=q_high)
        return df


class ImputeCategoricalStep(BaseStep):
    """Impute missing values in categorical columns with mode or fixed label."""

    def __init__(
        self,
        data_frame: pd.DataFrame,
        columns: list[str] | None = None,
        strategy: str = "mode",  # "mode" or "fixed"
        fill_value: str = "unknown",  # used if strategy="fixed"
    ) -> None:
        super().__init__(data_frame)
        self.columns = columns
        self.strategy = strategy
        self.fill_value = fill_value

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        cols = (
            self.columns
            if self.columns is not None
            else list(df.select_dtypes(include=["object", "category"]).columns)
        )
        for col in cols:
            if self.strategy == "mode":
                mode_vals = df[col].mode()
                if not mode_vals.empty:
                    df[col] = df[col].fillna(mode_vals[0])
            else:
                df[col] = df[col].fillna(self.fill_value)
        return df


class EnforceSchemaStep(BaseStep):
    """Ensure certain required columns exist and the df covers min constraints."""

    def __init__(
        self,
        data_frame: pd.DataFrame,
        required_columns: list[str] | None = None,
        min_rows: int = 1,
    ) -> None:
        super().__init__(data_frame)
        self.required_columns = required_columns or []
        self.min_rows = min_rows

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        if len(df) < self.min_rows:
            import warnings

            warnings.warn(f"DataFrame has fewer than {self.min_rows} rows.")
        missing_cols = [c for c in self.required_columns if c not in df.columns]
        if missing_cols:
            import warnings

            warnings.warn(f"Missing required columns in DataFrame: {missing_cols}")
        return df


class AddAuditColumnsStep(BaseStep):
    """Add an audit timestamp and original index column."""

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        if "_original_index" not in df.columns:
            df["_original_index"] = df.index
        df["_cleaned_at"] = datetime.now()
        return df


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class DataCleaningStepFactory:
    """Factory for creating and registering pipeline steps."""

    _registry: dict[str, type[BaseStep]] = {}

    @classmethod
    def register(cls, step_name: str, step: type[BaseStep]) -> None:
        """Register a step class under a name."""
        cls._registry[step_name] = step

    @classmethod
    def create(cls, step_name: str, data_frame: pd.DataFrame, **kwargs) -> BaseStep:
        """Instantiate a registered step with the given arguments."""
        step_class = cls._registry.get(step_name)
        if not step_class:
            raise ValueError(
                f"Step '{step_name}' not registered. "
                f"Available: {list(cls._registry.keys())}"
            )
        return step_class(data_frame, **kwargs)


# Register all steps
DataCleaningStepFactory.register("fix_columns_titles", ColumnsTitlesStep)
DataCleaningStepFactory.register("handle_sentinel_values", HandleSentinelValuesStep)
DataCleaningStepFactory.register("normalize_categories", NormalizeCategoriesStep)
DataCleaningStepFactory.register("safe_conversion", SafeConversionStep)
DataCleaningStepFactory.register("fix_not_numeric_columns", FixNotNumericColumnsStep)
DataCleaningStepFactory.register("remove_duplicates_rows", RemoveDuplicatesRowsStep)
DataCleaningStepFactory.register("validate_domain_rules", ValidateDomainRulesStep)
DataCleaningStepFactory.register("fix_numeric_columns", FixNumericColumnsStep)
DataCleaningStepFactory.register("handle_outliers", HandleOutliersStep)
DataCleaningStepFactory.register("fix_bools_columns", FixBoolsColumnsStep)
DataCleaningStepFactory.register("fix_dates_columns", FixDatesColumnsStep)
DataCleaningStepFactory.register("cross_column_validation", CrossColumnValidationStep)
DataCleaningStepFactory.register("flag_data_quality", FlagDataQualityStep)
DataCleaningStepFactory.register("fix_columns_types", FixColumnsTypesStep)

# New steps
DataCleaningStepFactory.register("drop_high_missing_columns", DropHighMissingColumnsStep)
DataCleaningStepFactory.register("drop_constant_columns", DropConstantColumnsStep)
DataCleaningStepFactory.register("text_standardization", TextStandardizationStep)
DataCleaningStepFactory.register("cap_outliers", CapOutliersStep)
DataCleaningStepFactory.register("impute_categorical", ImputeCategoricalStep)
DataCleaningStepFactory.register("enforce_schema", EnforceSchemaStep)
DataCleaningStepFactory.register("add_audit_columns", AddAuditColumnsStep)
