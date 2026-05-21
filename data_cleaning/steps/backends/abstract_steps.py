"""
data_cleaning/steps/backends/abstract_steps.py

Pure abstract contracts (interfaces) for all data cleaning pipeline steps.

Design rules strictly enforced here:
    - ZERO imports from Pandas, Polars, or PySpark.
    - All DataFrame types are expressed as Generic[T] (Any).
    - Each abstract class defines the exact constructor signature that
      every backend implementation MUST replicate.
    - method process() is the single transformation entrypoint (Template Method).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, FrozenSet, Generic, List, Optional, TypeVar

# T: represents the backend-specific DataFrame type bound at implementation time.
# e.g.  pd.DataFrame  |  pl.DataFrame  |  pyspark.sql.DataFrame
T = TypeVar("T")


# ─────────────────────────────────────────────────────────────────────────────
# Root Generic Contract
# ─────────────────────────────────────────────────────────────────────────────

class AbstractBaseStep(ABC, Generic[T]):
    """Root generic contract for a single data cleaning pipeline step.

    T is bound to the concrete DataFrame type of the chosen backend.
    All concrete implementations must:
        1. Accept T in __init__ as data_frame.
        2. Implement process(data: T) -> T returning a transformed copy.
        3. Never mutate the input DataFrame in-place.
    """

    def __init__(self, data_frame: T) -> None:
        self._data_frame: T = data_frame

    @abstractmethod
    def process(self, data: T) -> T:
        """Transform data and return a cleaned copy of the same backend type.

        Args:
            data: Input DataFrame (backend-specific type T).

        Returns:
            Cleaned DataFrame of the same type T.
        """


# ─────────────────────────────────────────────────────────────────────────────
# Abstract Steps — Schema & Structure
# ─────────────────────────────────────────────────────────────────────────────

class AbstractColumnScopedStep(AbstractBaseStep[T], Generic[T]):
    """Contract: Decorator restricting any step to a column subset.

    Wraps an inner AbstractBaseStep[T] and applies it only to the
    specified columns, leaving all other columns unchanged.
    """

    @abstractmethod
    def __init__(
        self,
        inner_step: AbstractBaseStep[T],
        data_frame: T,
        columns: List[str],
    ) -> None: ...

    @abstractmethod
    def process(self, data: T) -> T: ...


class AbstractColumnsTitlesStep(AbstractBaseStep[T], Generic[T]):
    """Contract: Normalize column names.

    Expected transformation: strip whitespace → lowercase →
    replace whitespace sequences with underscores → remove diacritics.
    Example: ' First Namé ' → 'first_name'
    """

    @abstractmethod
    def process(self, data: T) -> T: ...


class AbstractEnforceSchemaStep(AbstractBaseStep[T], Generic[T]):
    """Contract: Validate minimum structural requirements.

    Emits warnings (does NOT raise) when violations are detected
    so the pipeline remains running.
    """

    @abstractmethod
    def __init__(
        self,
        data_frame: T,
        required_columns: Optional[List[str]] = None,
        min_rows: int = 1,
    ) -> None: ...

    @abstractmethod
    def process(self, data: T) -> T: ...


class AbstractDropHighMissingColumnsStep(AbstractBaseStep[T], Generic[T]):
    """Contract: Drop columns where null fraction > threshold.

    Args:
        threshold: Maximum allowed missing fraction in [0.0, 1.0].
                   Default 0.8 (80%).
    """

    @abstractmethod
    def __init__(
        self,
        data_frame: T,
        threshold: float = 0.8,
    ) -> None: ...

    @abstractmethod
    def process(self, data: T) -> T: ...


class AbstractDropConstantColumnsStep(AbstractBaseStep[T], Generic[T]):
    """Contract: Drop columns carrying zero information (≤1 unique non-null value)."""

    @abstractmethod
    def process(self, data: T) -> T: ...


# ─────────────────────────────────────────────────────────────────────────────
# Abstract Steps — Missing Values & Sentinels
# ─────────────────────────────────────────────────────────────────────────────

class AbstractHandleSentinelValuesStep(AbstractBaseStep[T], Generic[T]):
    """Contract: Replace well-known placeholder strings with null/NaN.

    Default sentinel vocabulary: {unknown, nan, none, null, n/a, na,
    -, $-, , invalid_date, undefined, #n/a}.
    """

    @abstractmethod
    def __init__(
        self,
        data_frame: T,
        extra_sentinels: Optional[FrozenSet[str]] = None,
    ) -> None: ...

    @abstractmethod
    def process(self, data: T) -> T: ...


class AbstractImputeCategoricalStep(AbstractBaseStep[T], Generic[T]):
    """Contract: Impute missing values in categorical/object columns.

    Strategies:
        'mode'  — fill with the most frequent non-null value.
        'fixed' — fill with a constant string (fill_value).
    """

    @abstractmethod
    def __init__(
        self,
        data_frame: T,
        columns: Optional[List[str]] = None,
        strategy: Any = None,   # CategoricalImputationStrategy enum
        fill_value: str = "unknown",
    ) -> None: ...

    @abstractmethod
    def process(self, data: T) -> T: ...


# ─────────────────────────────────────────────────────────────────────────────
# Abstract Steps — Type Conversion & Parsing
# ─────────────────────────────────────────────────────────────────────────────

class AbstractSafeConversionStep(AbstractBaseStep[T], Generic[T]):
    """Contract: Attempt numeric / date conversion on string columns.

    Conversion chain per cell: written number → numeric → date.
    Pure-text columns are auto-skipped via digit_threshold.
    """

    @abstractmethod
    def __init__(
        self,
        data_frame: T,
        columns: Optional[List[str]] = None,
        digit_threshold: float = 0.3,
    ) -> None: ...

    @abstractmethod
    def process(self, data: T) -> T: ...


class AbstractFixDatesColumnsStep(AbstractBaseStep[T], Generic[T]):
    """Contract: Parse string columns to datetime and invalidate out-of-range dates.

    Valid range: [1900-01-01, now].
    Out-of-range values are set to null/NaT.
    """

    @abstractmethod
    def __init__(
        self,
        data_frame: T,
        columns: Optional[List[str]] = None,
    ) -> None: ...

    @abstractmethod
    def process(self, data: T) -> T: ...


class AbstractFixBoolsColumnsStep(AbstractBaseStep[T], Generic[T]):
    """Contract: Convert text boolean representations to native Boolean type.

    Boolean vocabulary: {y/n, yes/no, 1/0, true/false}.
    """

    @abstractmethod
    def __init__(
        self,
        data_frame: T,
        columns: Optional[List[str]] = None,
    ) -> None: ...

    @abstractmethod
    def process(self, data: T) -> T: ...


class AbstractFixColumnsTypesStep(AbstractBaseStep[T], Generic[T]):
    """Contract: Cast columns to their final target dtypes after all cleaning."""

    @abstractmethod
    def __init__(
        self,
        data_frame: T,
        numeric_columns: Optional[List[str]] = None,
        bool_columns: Optional[List[str]] = None,
        date_columns: Optional[List[str]] = None,
    ) -> None: ...

    @abstractmethod
    def process(self, data: T) -> T: ...


# ─────────────────────────────────────────────────────────────────────────────
# Abstract Steps — Numeric Cleaning & Imputation
# ─────────────────────────────────────────────────────────────────────────────

class AbstractFixNumericColumnsStep(AbstractBaseStep[T], Generic[T]):
    """Contract: Clean numeric strings and impute NaN using a configurable strategy.

    Imputation strategies: mean | median | mode.
    """

    @abstractmethod
    def __init__(
        self,
        data_frame: T,
        strategy: Any = None,   # ImputationStrategy enum
        columns: Optional[List[str]] = None,
    ) -> None: ...

    @abstractmethod
    def process(self, data: T) -> T: ...


# ─────────────────────────────────────────────────────────────────────────────
# Abstract Steps — Outlier Handling
# ─────────────────────────────────────────────────────────────────────────────

class AbstractIQROutlierStep(AbstractBaseStep[T], Generic[T]):
    """Contract: Clip outliers to Tukey IQR fence [Q1 - 1.5·IQR, Q3 + 1.5·IQR].

    Values beyond the fence are CLIPPED (not removed) to preserve row count.
    """

    @abstractmethod
    def __init__(
        self,
        data_frame: T,
        columns: Optional[List[str]] = None,
    ) -> None: ...

    @abstractmethod
    def process(self, data: T) -> T: ...


class AbstractZScoreOutlierStep(AbstractBaseStep[T], Generic[T]):
    """Contract: Nullify values whose absolute Z-score exceeds z_threshold.

    Columns with σ = 0 are skipped. Requires ≥ 4 valid observations.
    """

    @abstractmethod
    def __init__(
        self,
        data_frame: T,
        columns: Optional[List[str]] = None,
        z_threshold: float = 3.0,
    ) -> None: ...

    @abstractmethod
    def process(self, data: T) -> T: ...


class AbstractCapOutliersStep(AbstractBaseStep[T], Generic[T]):
    """Contract: Winsorize outliers by capping at configurable percentile bounds."""

    @abstractmethod
    def __init__(
        self,
        data_frame: T,
        columns: Optional[List[str]] = None,
        lower_percentile: float = 0.01,
        upper_percentile: float = 0.99,
    ) -> None: ...

    @abstractmethod
    def process(self, data: T) -> T: ...


# ─────────────────────────────────────────────────────────────────────────────
# Abstract Steps — Text Normalization
# ─────────────────────────────────────────────────────────────────────────────

class AbstractFixNotNumericColumnsStep(AbstractBaseStep[T], Generic[T]):
    """Contract: Normalize text/categorical columns.

    Transformation: strip → lowercase → replace internal spaces with underscores.
    Columns inferred as numeric are automatically skipped.
    """

    @abstractmethod
    def __init__(
        self,
        data_frame: T,
        columns: Optional[List[str]] = None,
    ) -> None: ...

    @abstractmethod
    def process(self, data: T) -> T: ...


class AbstractNormalizeCategoriesStep(AbstractBaseStep[T], Generic[T]):
    """Contract: Unify known category variants using a per-column synonym map.

    Args:
        mappings: dict[col_name, dict[alias, canonical_value]]
                  e.g. {'gender': {'m': 'male', 'f': 'female'}}
    """

    @abstractmethod
    def __init__(
        self,
        data_frame: T,
        mappings: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> None: ...

    @abstractmethod
    def process(self, data: T) -> T: ...


class AbstractTextStandardizationStep(AbstractBaseStep[T], Generic[T]):
    """Contract: Advanced text normalization per cell.

    Pipeline:
        1. NFKD Unicode decomposition → strip diacritical marks (é→e, ñ→n).
        2. Remove special characters (preserve word chars, spaces, hyphens).
        3. Collapse repeated whitespace and lowercase.
    """

    @abstractmethod
    def __init__(
        self,
        data_frame: T,
        columns: Optional[List[str]] = None,
    ) -> None: ...

    @abstractmethod
    def process(self, data: T) -> T: ...


# ─────────────────────────────────────────────────────────────────────────────
# Abstract Steps — Validation
# ─────────────────────────────────────────────────────────────────────────────

class AbstractValidateDomainRulesStep(AbstractBaseStep[T], Generic[T]):
    """Contract: Nullify per-column values violating domain bounds.

    Args:
        rules: dict[col_name, DomainBounds(lower, upper)]
    """

    @abstractmethod
    def __init__(
        self,
        data_frame: T,
        rules: Optional[Dict[str, Any]] = None,   # Dict[str, DomainBounds]
    ) -> None: ...

    @abstractmethod
    def process(self, data: T) -> T: ...


class AbstractCrossColumnValidationStep(AbstractBaseStep[T], Generic[T]):
    """Contract: Validate consistency between pairs of related columns.

    Args:
        rules: list[CrossColumnRule(if_col, equals, then_col, action)]
    """

    @abstractmethod
    def __init__(
        self,
        data_frame: T,
        rules: Optional[List[Any]] = None,   # List[CrossColumnRule]
    ) -> None: ...

    @abstractmethod
    def process(self, data: T) -> T: ...


# ─────────────────────────────────────────────────────────────────────────────
# Abstract Steps — Duplicate Handling, Quality & Audit
# ─────────────────────────────────────────────────────────────────────────────

class AbstractRemoveDuplicatesRowsStep(AbstractBaseStep[T], Generic[T]):
    """Contract: Drop exact duplicate rows, preserving the first occurrence."""

    @abstractmethod
    def process(self, data: T) -> T: ...


class AbstractFlagDataQualityStep(AbstractBaseStep[T], Generic[T]):
    """Contract: Append a '_quality_score' column.

    Score = 1.0 − (null_count_per_row / total_columns).
    A score of 1.0 indicates a fully complete row.
    """

    @abstractmethod
    def process(self, data: T) -> T: ...


class AbstractAddAuditColumnsStep(AbstractBaseStep[T], Generic[T]):
    """Contract: Append lineage-tracking columns.

    Columns added:
        '_original_index': Row index before cleaning.
        '_cleaned_at':     UTC timestamp of this step's execution.
    """

    @abstractmethod
    def process(self, data: T) -> T: ...


# ─────────────────────────────────────────────────────────────────────────────
# Abstract Steps — Feature Scaling
# ─────────────────────────────────────────────────────────────────────────────

class AbstractStandardScalerStep(AbstractBaseStep[T], Generic[T]):
    """Contract: Z-score normalization z = (x − μ) / σ with fit/transform separation.

    Implements scikit-learn-style API to prevent data leakage:
        scaler.fit(train_data)
        train_scaled = scaler.transform(train_data)
        test_scaled  = scaler.transform(test_data)   ← uses train μ/σ, no leakage

    process() = fit() + transform() on the same data (training convenience only).
    """

    @abstractmethod
    def __init__(
        self,
        data_frame: T,
        columns: Optional[List[str]] = None,
    ) -> None: ...

    @abstractmethod
    def fit(self, data: T) -> "AbstractStandardScalerStep":
        """Compute μ and σ from data and store for later transform() calls.

        Returns:
            self — enables fit(data).transform(other) chaining.
        """

    @abstractmethod
    def transform(self, data: T) -> T:
        """Apply the fitted scaling parameters to data.

        Raises:
            RuntimeError: If called before fit().
        """

    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """True if fit() has been called at least once."""

    @abstractmethod
    def process(self, data: T) -> T: ...
