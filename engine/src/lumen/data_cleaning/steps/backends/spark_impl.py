"""
data_cleaning/steps/backends/spark_impl.py

BACKEND DISTRIBUIDO ENTERPRISE — Implementaciones PySpark de todos los steps.

Reglas estrictas:
    - Usa exclusivamente pyspark.sql.functions (import as F).
    - Evaluación perezosa de Spark respetada en todos los métodos transform.
    - .collect() SOLO cuando se necesita un escalar de agregación global
      (imputation fill values, outlier bounds) — documentado como [ACTION puntual].
    - Sin .toPandas() en ningún step de transformación.
    - UDFs usados únicamente en TextStandardizationStep (NFKD no vectorizable en Spark).
    - approxQuantile() acepta un pequeño error de aproximación (relativeError=0.01)
      para evitar full sort distribuido.
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

from pyspark.sql import DataFrame as SparkDataFrame
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import Window

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

# Re-export value objects (backend-agnostic)
from lumen.data_cleaning.steps.backends.pandas_impl import (
    CategoricalImputationStrategy,
    CrossColumnRule,
    DomainBounds,
    ImputationStrategy,
)

# ─────────────────────────────────────────────────────────────────────────────
# Steps: Schema & Structure
# ─────────────────────────────────────────────────────────────────────────────

class ColumnScopedStep(AbstractColumnScopedStep[SparkDataFrame]):
    """Spark: Decorator restricting any step to a subset of columns.

    Strategy: adds a stable monotonically_increasing_id for row alignment,
    processes the scoped subset, then joins back with the remainder.
    """

    _ROW_ID: Final[str] = "__UNIQUE_INDEX_FOR_JOIN_DO_NOT_TOUCH__"

    def __init__(
        self,
        inner_step: AbstractColumnScopedStep,
        data_frame: SparkDataFrame,
        columns: List[str],
    ) -> None:
        super().__init__(inner_step, data_frame, columns)

    def process(self, data: SparkDataFrame) -> SparkDataFrame:
        available = [c for c in self._scoped_columns if c in data.columns]
        if not available:
            return data

        remainder = [c for c in data.columns if c not in available]

        # Add stable row ID using a true UUID to avoid shuffle misalignments
        df_with_id = data.withColumn(self._ROW_ID, F.expr("uuid()"))

        # Wrap in a StructType so that no inner_step (like text normalization) touches it
        df_with_id = df_with_id.withColumn(
            self._ROW_ID, F.struct(F.col(self._ROW_ID).alias("val"))
        )

        scoped_df    = df_with_id.select(available + [self._ROW_ID])
        remainder_df = df_with_id.select(remainder + [self._ROW_ID])

        # Apply inner step. The StructType protects _ROW_ID from being altered.
        cleaned_scoped = self._inner_step.process(scoped_df)

        result = remainder_df.join(
            cleaned_scoped,
            on=self._ROW_ID,
            how="inner",
        ).drop(self._ROW_ID)

        # Restore original column order
        return result.select(data.columns)

class ColumnsTitlesStep(AbstractColumnsTitlesStep[SparkDataFrame]):
    """Spark: Normalize column names (lowercase, underscored, no diacritics)."""

    def __init__(self, data_frame: SparkDataFrame) -> None:
        super().__init__(data_frame)

    def process(self, data: SparkDataFrame) -> SparkDataFrame:
        df = data
        for old_col in data.columns:
            clean = old_col.strip().lower()
            clean = re.sub(r"\s+", "_", clean)
            clean = "".join(
                c for c in unicodedata.normalize("NFD", clean)
                if unicodedata.category(c) != "Mn"
            )
            if clean != old_col:
                df = df.withColumnRenamed(old_col, clean)
        return df

class EnforceSchemaStep(AbstractEnforceSchemaStep[SparkDataFrame]):
    """Spark: Validate minimum structural requirements (warns, does not raise)."""

    def __init__(
        self,
        data_frame: SparkDataFrame,
        required_columns: Optional[List[str]] = None,
        min_rows: int = 1,
    ) -> None:
        super().__init__(data_frame)
        self._required = required_columns or []
        self._min_rows = min_rows

    def process(self, data: SparkDataFrame) -> SparkDataFrame:
        if self._min_rows > 1:
            # [ACTION puntual: count necesario para validar min_rows]
            n_rows = data.count()
            if n_rows < self._min_rows:
                warnings.warn(
                    f"DataFrame has {n_rows} row(s); expected at least {self._min_rows}.",
                    stacklevel=2,
                )
        missing = [c for c in self._required if c not in data.columns]
        if missing:
            warnings.warn(f"Missing required columns: {missing}", stacklevel=2)
        return data

class DropHighMissingColumnsStep(AbstractDropHighMissingColumnsStep[SparkDataFrame]):
    """Spark: Drop columns where null fraction > threshold."""

    def __init__(self, data_frame: SparkDataFrame, threshold: float = 0.8) -> None:
        super().__init__(data_frame)
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"threshold must be in [0.0, 1.0], got {threshold}.")
        self.threshold = threshold

    def process(self, data: SparkDataFrame) -> SparkDataFrame:
        # [ACTION puntual: agregación para calcular fracciones nulas por columna]
        null_fracs = data.agg(
            *[
                (F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)) / F.count("*"))
                .alias(c)
                for c in data.columns
            ]
        ).collect()[0].asDict()

        cols_to_drop = [c for c, frac in null_fracs.items() if frac > self.threshold]
        return data.drop(*cols_to_drop) if cols_to_drop else data

class DropConstantColumnsStep(AbstractDropConstantColumnsStep[SparkDataFrame]):
    """Spark: Drop columns with ≤1 unique non-null value."""

    def __init__(self, data_frame: SparkDataFrame) -> None:
        super().__init__(data_frame)

    def process(self, data: SparkDataFrame) -> SparkDataFrame:
        # [ACTION puntual: approxCountDistinct por columna]
        n_unique = data.agg(
            *[F.approx_count_distinct(c).alias(c) for c in data.columns]
        ).collect()[0].asDict()
        cols_to_drop = [c for c, n in n_unique.items() if n <= 1]
        return data.drop(*cols_to_drop) if cols_to_drop else data

# ─────────────────────────────────────────────────────────────────────────────
# Steps: Missing Values & Sentinel Handling
# ─────────────────────────────────────────────────────────────────────────────

class HandleSentinelValuesStep(AbstractHandleSentinelValuesStep[SparkDataFrame]):
    """Spark: Replace sentinel strings with null using native F.when expressions."""

    _DEFAULT_SENTINELS: Final[FrozenSet[str]] = frozenset({
        "unknown", "nan", "none", "null", "n/a", "na",
        "-", "$-", "", "invalid_date", "undefined", "#n/a",
    })

    def __init__(
        self,
        data_frame: SparkDataFrame,
        extra_sentinels: Optional[FrozenSet[str]] = None,
    ) -> None:
        super().__init__(data_frame)
        self._sentinels: FrozenSet[str] = (
            self._DEFAULT_SENTINELS | (extra_sentinels or frozenset())
        )

    def process(self, data: SparkDataFrame) -> SparkDataFrame:
        df = data
        sentinel_list = list(self._sentinels)
        for col in data.columns:
            df = df.withColumn(
                col,
                F.when(
                    F.lower(F.trim(F.col(col).cast(T.StringType()))).isin(sentinel_list),
                    F.lit(None)
                ).otherwise(F.col(col))
            )
        return df

class ImputeCategoricalStep(AbstractImputeCategoricalStep[SparkDataFrame]):
    """Spark: Impute missing categorical values (mode via groupBy or fixed)."""

    def __init__(
        self,
        data_frame: SparkDataFrame,
        columns: Optional[List[str]] = None,
        strategy: CategoricalImputationStrategy = CategoricalImputationStrategy.MODE,
        fill_value: str = "unknown",
    ) -> None:
        super().__init__(data_frame)
        self.columns    = columns
        self.strategy   = strategy
        self.fill_value = fill_value

    def process(self, data: SparkDataFrame) -> SparkDataFrame:
        string_cols = [
            c for c in data.columns
            if isinstance(data.schema[c].dataType, (T.StringType, T.BinaryType))
        ]
        target_cols = (
            [c for c in self.columns if c in data.columns]
            if self.columns is not None
            else string_cols
        )

        df = data
        for col in target_cols:
            if self.strategy == CategoricalImputationStrategy.MODE:
                # [ACTION puntual: mode via groupBy + collect]
                mode_row = (
                    df.where(F.col(col).isNotNull())
                    .groupBy(col).count()
                    .orderBy(F.desc("count"))
                    .limit(1)
                    .collect()
                )
                if mode_row:
                    mode_val = mode_row[0][col]
                    df = df.withColumn(
                        col,
                        F.when(F.col(col).isNull(), F.lit(mode_val)).otherwise(F.col(col))
                    )
            else:
                df = df.withColumn(
                    col,
                    F.when(F.col(col).isNull(), F.lit(self.fill_value)).otherwise(F.col(col))
                )
        return df

# ─────────────────────────────────────────────────────────────────────────────
# Steps: Type Conversion & Parsing
# ─────────────────────────────────────────────────────────────────────────────

class SafeConversionStep(AbstractSafeConversionStep[SparkDataFrame]):
    """Spark: Attempt numeric conversion via regex + cast on string columns."""

    _DIGIT_REGEX: Final[str] = r".*[\d$€£].*"

    def __init__(
        self,
        data_frame: SparkDataFrame,
        columns: Optional[List[str]] = None,
        digit_threshold: float = 0.3,
    ) -> None:
        super().__init__(data_frame)
        self.columns         = columns
        self.digit_threshold = digit_threshold

    def process(self, data: SparkDataFrame) -> SparkDataFrame:
        cols = self._resolve_columns(data)
        df   = data
        for col in cols:
            # Step 1: word → number (top 20 English number words)
            word_map = {
                "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
                "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
                "ten": "10", "twenty": "20", "thirty": "30", "forty": "40", "fifty": "50",
                "sixty": "60", "seventy": "70", "eighty": "80", "ninety": "90", "hundred": "100",
            }
            normalized = F.lower(F.trim(F.col(col).cast(T.StringType())))
            word_expr = normalized
            for word, num in word_map.items():
                word_expr = F.when(normalized == F.lit(word), F.lit(num)).otherwise(word_expr)

            # Step 2: strip currency symbols + cast to double
            stripped = F.regexp_replace(
                F.regexp_replace(word_expr, r"[$€£\s]", ""),
                ",", "."
            )
            df = df.withColumn(col, stripped.cast(T.DoubleType()))
        return df

    def _resolve_columns(self, data: SparkDataFrame) -> List[str]:
        if self.columns is not None:
            return [c for c in self.columns if c in data.columns]
        # Heuristic: string columns — check digit ratio via agg
        string_cols = [
            c for c in data.columns
            if isinstance(data.schema[c].dataType, T.StringType)
        ]
        if not string_cols:
            return []
        # [ACTION puntual: ratio de dígitos por columna]
        total_count = data.count()
        if total_count == 0:
            return []
        digit_fracs = data.agg(
            *[
                (F.sum(F.when(F.col(c).rlike(self._DIGIT_REGEX), 1).otherwise(0)) / F.lit(total_count))
                .alias(c)
                for c in string_cols
            ]
        ).collect()[0].asDict()
        return [c for c, frac in digit_fracs.items() if frac > self.digit_threshold]

class FixDatesColumnsStep(AbstractFixDatesColumnsStep[SparkDataFrame]):
    """Spark: Parse string columns to timestamp, invalidate out-of-range dates."""

    _MIN_DATE: Final[str] = "1900-01-01"

    def __init__(
        self,
        data_frame: SparkDataFrame,
        columns: Optional[List[str]] = None,
    ) -> None:
        super().__init__(data_frame)
        self.columns = columns

    def process(self, data: SparkDataFrame) -> SparkDataFrame:
        if self.columns is not None:
            target_cols = [c for c in self.columns if c in data.columns]
        else:
            target_cols = [
                c for c in data.columns
                if isinstance(data.schema[c].dataType, T.StringType)
            ]

        df   = data
        min_ts = F.to_timestamp(F.lit(self._MIN_DATE), "yyyy-MM-dd")
        max_ts = F.current_timestamp()

        for col in target_cols:
            parsed = F.to_timestamp(F.col(col))
            df = df.withColumn(
                col,
                F.when(
                    (parsed >= min_ts) & (parsed <= max_ts), parsed
                ).otherwise(F.lit(None).cast(T.TimestampType()))
            )
        return df

class FixBoolsColumnsStep(AbstractFixBoolsColumnsStep[SparkDataFrame]):
    """Spark: Convert text boolean representations to BooleanType."""

    _TRUE_VALS:  Final[List[str]] = ["yes", "y", "true", "1"]
    _FALSE_VALS: Final[List[str]] = ["no", "n", "false", "0"]
    _BOOL_VOCAB: Final[List[str]] = _TRUE_VALS + _FALSE_VALS

    def __init__(
        self,
        data_frame: SparkDataFrame,
        columns: Optional[List[str]] = None,
    ) -> None:
        super().__init__(data_frame)
        self.columns = columns

    def process(self, data: SparkDataFrame) -> SparkDataFrame:
        target_cols = (
            [c for c in self.columns if c in data.columns]
            if self.columns is not None
            else self._auto_detect(data)
        )
        df = data
        for col in target_cols:
            normalized = F.lower(F.trim(F.col(col).cast(T.StringType())))
            df = df.withColumn(
                col,
                F.when(normalized.isin(self._TRUE_VALS), F.lit(True))
                .when(normalized.isin(self._FALSE_VALS), F.lit(False))
                .otherwise(F.lit(None).cast(T.BooleanType()))
            )
        return df

    def _auto_detect(self, data: SparkDataFrame) -> List[str]:
        """Heuristic auto-detection of boolean columns via distinct value check."""
        string_cols = [
            c for c in data.columns
            if isinstance(data.schema[c].dataType, T.StringType)
        ]
        result = []
        for col in string_cols:
            # [ACTION puntual: distinct values para detectar booleanos]
            distinct_vals = {
                row[col].lower().strip()
                for row in data.select(F.lower(F.trim(F.col(col))).alias(col))
                .where(F.col(col).isNotNull()).distinct().limit(10).collect()
            } - {"nan", "<na>", "none", ""}
            if distinct_vals and distinct_vals.issubset(set(self._BOOL_VOCAB)):
                result.append(col)
        return result

class FixColumnsTypesStep(AbstractFixColumnsTypesStep[SparkDataFrame]):
    """Spark: Cast columns to their final target dtypes."""

    def __init__(
        self,
        data_frame: SparkDataFrame,
        numeric_columns: Optional[List[str]] = None,
        bool_columns: Optional[List[str]] = None,
        date_columns: Optional[List[str]] = None,
    ) -> None:
        super().__init__(data_frame)
        self.numeric_columns = numeric_columns
        self.bool_columns    = bool_columns
        self.date_columns    = date_columns

    def process(self, data: SparkDataFrame) -> SparkDataFrame:
        schema = {f.name: f.dataType for f in data.schema.fields}
        num_cols = (
            [c for c in self.numeric_columns if c in data.columns]
            if self.numeric_columns is not None
            else [c for c, dt in schema.items() if isinstance(dt, (T.LongType, T.IntegerType, T.DoubleType, T.FloatType))]
        )
        bool_cols = (
            [c for c in self.bool_columns if c in data.columns]
            if self.bool_columns is not None
            else [c for c, dt in schema.items() if isinstance(dt, T.BooleanType)]
        )
        date_cols = (
            [c for c in self.date_columns if c in data.columns]
            if self.date_columns is not None
            else [c for c, dt in schema.items() if isinstance(dt, (T.DateType, T.TimestampType))]
        )

        df = data
        for col in num_cols:
            df = df.withColumn(col, F.col(col).cast(T.DoubleType()))
        for col in bool_cols:
            df = df.withColumn(col, F.col(col).cast(T.BooleanType()))
        for col in date_cols:
            df = df.withColumn(col, F.to_timestamp(F.col(col)))
        return df

# ─────────────────────────────────────────────────────────────────────────────
# Steps: Numeric Cleaning & Imputation
# ─────────────────────────────────────────────────────────────────────────────

class FixNumericColumnsStep(AbstractFixNumericColumnsStep[SparkDataFrame]):
    """Spark: Clean numeric strings (regex) and impute NaN via native aggregation."""

    def __init__(
        self,
        data_frame: SparkDataFrame,
        strategy: ImputationStrategy = ImputationStrategy.MEDIAN,
        columns: Optional[List[str]] = None,
    ) -> None:
        super().__init__(data_frame)
        self.strategy = strategy
        self.columns  = columns

    def process(self, data: SparkDataFrame) -> SparkDataFrame:
        target_cols = self._resolve_columns(data)

        df = data
        for col in target_cols:
            # Clean: strip currency + non-numeric chars, replace comma→dot, cast
            cleaned = (
                F.regexp_replace(
                    F.regexp_replace(
                        F.regexp_replace(F.col(col).cast(T.StringType()), r"[$€£\s]", ""),
                        ",", "."
                    ),
                    r"[^\d.\-]", ""
                )
                .cast(T.DoubleType())
            )
            df = df.withColumn(col, cleaned)

        for col in target_cols:
            fill_val = self._compute_fill_value(df, col)
            if fill_val is not None:
                df = df.withColumn(
                    col,
                    F.when(F.col(col).isNull(), F.lit(fill_val)).otherwise(F.col(col))
                )
        return df

    def _resolve_columns(self, data: SparkDataFrame) -> List[str]:
        if self.columns is not None:
            return [c for c in self.columns if c in data.columns]
        return [
            c for c in data.columns
            if not c.endswith("_id") and c != "id"
            and isinstance(data.schema[c].dataType, (
                T.StringType, T.DoubleType, T.FloatType,
                T.IntegerType, T.LongType, T.ShortType
            ))
        ]

    def _compute_fill_value(self, df: SparkDataFrame, col: str) -> float | None:
        """[ACTION puntual]: compute scalar fill value via native Spark aggregation."""
        if self.strategy == ImputationStrategy.MEAN:
            row = df.agg(F.mean(F.col(col))).collect()[0]
            return float(row[0]) if row[0] is not None else None
        elif self.strategy == ImputationStrategy.MEDIAN:
            # approxQuantile avoids full distributed sort
            result = df.approxQuantile(col, [0.5], 0.01)
            return float(result[0]) if result else None
        else:  # MODE
            row = (
                df.where(F.col(col).isNotNull())
                .groupBy(col).count()
                .orderBy(F.desc("count"))
                .limit(1)
                .collect()
            )
            return float(row[0][col]) if row else None

# ─────────────────────────────────────────────────────────────────────────────
# Steps: Outlier Handling
# ─────────────────────────────────────────────────────────────────────────────

class IQROutlierStep(AbstractIQROutlierStep[SparkDataFrame]):
    """Spark: Clip outliers to Tukey IQR fence using approxQuantile."""

    def __init__(
        self,
        data_frame: SparkDataFrame,
        columns: Optional[List[str]] = None,
    ) -> None:
        super().__init__(data_frame)
        self.columns = columns

    def process(self, data: SparkDataFrame) -> SparkDataFrame:
        target_cols = self._resolve_columns(data)
        df = data
        for col in target_cols:
            # [ACTION puntual: approxQuantile para Q1/Q3]
            quantiles = df.approxQuantile(col, [0.25, 0.75], 0.01)
            if len(quantiles) < 2:
                continue
            q1, q3  = quantiles[0], quantiles[1]
            iqr     = q3 - q1
            lower   = q1 - 1.5 * iqr
            upper   = q3 + 1.5 * iqr
            df = df.withColumn(
                col,
                F.when(F.col(col) < lower, F.lit(lower))
                .when(F.col(col) > upper, F.lit(upper))
                .otherwise(F.col(col))
            )
        return df

    def _resolve_columns(self, data: SparkDataFrame) -> List[str]:
        if self.columns is not None:
            return [c for c in self.columns if c in data.columns]
        return [
            c for c in data.columns
            if isinstance(data.schema[c].dataType, (T.DoubleType, T.FloatType, T.IntegerType, T.LongType))
            and not c.endswith("_id") and c != "id"
        ]

class ZScoreOutlierStep(AbstractZScoreOutlierStep[SparkDataFrame]):
    """Spark: Nullify values beyond z_threshold standard deviations from mean."""

    _MIN_OBSERVATIONS: Final[int] = 4

    def __init__(
        self,
        data_frame: SparkDataFrame,
        columns: Optional[List[str]] = None,
        z_threshold: float = 3.0,
    ) -> None:
        super().__init__(data_frame)
        if z_threshold <= 0:
            raise ValueError(f"z_threshold must be strictly positive, got {z_threshold}.")
        self.columns     = columns
        self.z_threshold = z_threshold

    def process(self, data: SparkDataFrame) -> SparkDataFrame:
        target_cols = (
            [c for c in self.columns if c in data.columns]
            if self.columns is not None
            else [
                c for c in data.columns
                if isinstance(data.schema[c].dataType, (T.DoubleType, T.FloatType, T.IntegerType, T.LongType))
                and not c.endswith("_id") and c != "id"
            ]
        )
        df = data
        for col in target_cols:
            # [ACTION puntual: mean y stddev para calcular z-scores]
            stats_row = df.agg(
                F.mean(F.col(col)).alias("mean"),
                F.stddev(F.col(col)).alias("std"),
                F.count(F.col(col)).alias("n"),
            ).collect()[0]

            mean_val = stats_row["mean"]
            std_val  = stats_row["std"]
            n_obs    = stats_row["n"]

            if mean_val is None or std_val is None or std_val == 0 or n_obs < self._MIN_OBSERVATIONS:
                continue

            z_score_expr = F.abs((F.col(col) - F.lit(mean_val)) / F.lit(std_val))
            df = df.withColumn(
                col,
                F.when(z_score_expr > self.z_threshold, F.lit(None).cast(T.DoubleType()))
                .otherwise(F.col(col))
            )
        return df

class CapOutliersStep(AbstractCapOutliersStep[SparkDataFrame]):
    """Spark: Winsorize outliers at configurable percentile bounds (approxQuantile)."""

    def __init__(
        self,
        data_frame: SparkDataFrame,
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

    def process(self, data: SparkDataFrame) -> SparkDataFrame:
        target_cols = (
            [c for c in self.columns if c in data.columns]
            if self.columns is not None
            else [
                c for c in data.columns
                if isinstance(data.schema[c].dataType, (T.DoubleType, T.FloatType, T.IntegerType, T.LongType))
                and not c.endswith("_id") and c != "id"
            ]
        )
        df = data
        for col in target_cols:
            # [ACTION puntual: approxQuantile]
            quantiles = df.approxQuantile(col, [self.lower_percentile, self.upper_percentile], 0.01)
            if len(quantiles) < 2:
                continue
            lo, hi = quantiles[0], quantiles[1]
            df = df.withColumn(
                col,
                F.when(F.col(col) < lo, F.lit(lo))
                .when(F.col(col) > hi, F.lit(hi))
                .otherwise(F.col(col))
            )
        return df

# ─────────────────────────────────────────────────────────────────────────────
# Steps: Text Normalization
# ─────────────────────────────────────────────────────────────────────────────

class FixNotNumericColumnsStep(AbstractFixNotNumericColumnsStep[SparkDataFrame]):
    """Spark: Normalize text columns (strip, lowercase, underscores)."""

    def __init__(
        self,
        data_frame: SparkDataFrame,
        columns: Optional[List[str]] = None,
    ) -> None:
        super().__init__(data_frame)
        self.columns = columns

    def process(self, data: SparkDataFrame) -> SparkDataFrame:
        target_cols = (
            [c for c in self.columns if c in data.columns]
            if self.columns is not None
            else [c for c in data.columns if isinstance(data.schema[c].dataType, T.StringType)]
        )
        df = data
        for col in target_cols:
            df = df.withColumn(
                col,
                F.regexp_replace(
                    F.lower(F.trim(F.col(col))),
                    r"\s+", "_"
                )
            )
        return df

class NormalizeCategoriesStep(AbstractNormalizeCategoriesStep[SparkDataFrame]):
    """Spark: Unify category variants via F.when chains from synonym maps."""

    def __init__(
        self,
        data_frame: SparkDataFrame,
        mappings: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> None:
        super().__init__(data_frame)
        self.mappings: Dict[str, Dict[str, str]] = mappings or {}

    def process(self, data: SparkDataFrame) -> SparkDataFrame:
        df = data
        for col, mapping in self.mappings.items():
            if col not in df.columns or not mapping:
                continue
            normalized = F.lower(F.trim(F.col(col)))
            expr = normalized
            for alias, canonical in mapping.items():
                expr = F.when(normalized == F.lit(alias), F.lit(canonical)).otherwise(expr)
            df = df.withColumn(col, expr)
        return df

class TextStandardizationStep(AbstractTextStandardizationStep[SparkDataFrame]):
    """Spark: Text normalization — regex-based (NFKD via UDF, only justified exception).

    NOTE: Full NFKD diacritic removal requires a UDF since Spark has no native
    unicode normalization. This UDF is applied per-partition (not per-row collect).
    """

    _SPECIAL_CHAR_REGEX: Final[str] = r"[^\w\s\-]"
    _WHITESPACE_REGEX:   Final[str] = r"\s+"

    def __init__(
        self,
        data_frame: SparkDataFrame,
        columns: Optional[List[str]] = None,
    ) -> None:
        super().__init__(data_frame)
        self.columns = columns

    def process(self, data: SparkDataFrame) -> SparkDataFrame:
        target_cols = (
            self.columns
            if self.columns is not None
            else [c for c in data.columns if isinstance(data.schema[c].dataType, T.StringType)]
        )

        # UDF for NFKD normalization (applied per-partition, not a collect)
        @F.udf(returnType=T.StringType())
        def _nfkd_normalize(text: str | None) -> str | None:
            if text is None:
                return None
            text = (
                unicodedata.normalize("NFKD", text)
                .encode("ascii", "ignore")
                .decode("utf-8")
            )
            text = re.sub(r"[^\w\s\-]", " ", text)
            return re.sub(r"\s+", " ", text).strip().lower()

        df = data
        for col in (c for c in target_cols if c in df.columns):
            # Phase 1: Spark-native cleanup (fast, distributed)
            cleaned = F.regexp_replace(
                F.lower(F.trim(F.col(col))),
                self._WHITESPACE_REGEX, " "
            )
            # Phase 2: NFKD via UDF (per-partition, not a Driver collect)
            df = df.withColumn(col, _nfkd_normalize(cleaned))
        return df

# ─────────────────────────────────────────────────────────────────────────────
# Steps: Validation
# ─────────────────────────────────────────────────────────────────────────────

class ValidateDomainRulesStep(AbstractValidateDomainRulesStep[SparkDataFrame]):
    """Spark: Nullify per-column values violating domain bounds."""

    def __init__(
        self,
        data_frame: SparkDataFrame,
        rules: Optional[Dict[str, DomainBounds]] = None,
    ) -> None:
        super().__init__(data_frame)
        self.rules: Dict[str, DomainBounds] = rules or {}

    def process(self, data: SparkDataFrame) -> SparkDataFrame:
        df = data
        for col, bounds in self.rules.items():
            if col not in df.columns:
                continue
            numeric_col = F.col(col).cast(T.DoubleType())
            condition   = F.lit(False)
            if bounds.lower is not None:
                condition = condition | (numeric_col < F.lit(bounds.lower))
            if bounds.upper is not None:
                condition = condition | (numeric_col > F.lit(bounds.upper))
            df = df.withColumn(
                col,
                F.when(condition, F.lit(None)).otherwise(F.col(col))
            )
        return df

class CrossColumnValidationStep(AbstractCrossColumnValidationStep[SparkDataFrame]):
    """Spark: Validate cross-column rules using F.when native expressions."""

    def __init__(
        self,
        data_frame: SparkDataFrame,
        rules: Optional[List[CrossColumnRule]] = None,
    ) -> None:
        super().__init__(data_frame)
        self.rules: List[CrossColumnRule] = rules or []

    def process(self, data: SparkDataFrame) -> SparkDataFrame:
        df = data
        for rule in self.rules:
            if rule.if_col not in df.columns or rule.then_col not in df.columns:
                continue
            if rule.action == "set_nan":
                condition = (F.col(rule.if_col) == F.lit(rule.equals)) & F.col(rule.then_col).isNull()
                df = df.withColumn(
                    rule.if_col,
                    F.when(condition, F.lit(None)).otherwise(F.col(rule.if_col))
                )
        return df

# ─────────────────────────────────────────────────────────────────────────────
# Steps: Duplicate Handling
# ─────────────────────────────────────────────────────────────────────────────

class RemoveDuplicatesRowsStep(AbstractRemoveDuplicatesRowsStep[SparkDataFrame]):
    """Spark: Drop exact duplicate rows via native dropDuplicates()."""

    def __init__(self, data_frame: SparkDataFrame) -> None:
        super().__init__(data_frame)

    def process(self, data: SparkDataFrame) -> SparkDataFrame:
        return data.dropDuplicates()

# ─────────────────────────────────────────────────────────────────────────────
# Steps: Quality & Audit
# ─────────────────────────────────────────────────────────────────────────────

class FlagDataQualityStep(AbstractFlagDataQualityStep[SparkDataFrame]):
    """Spark: Append '_quality_score' column via native F.when sum of null indicators."""

    _QUALITY_COLUMN: Final[str] = "_quality_score"

    def __init__(self, data_frame: SparkDataFrame) -> None:
        super().__init__(data_frame)

    def process(self, data: SparkDataFrame) -> SparkDataFrame:
        n_cols = len(data.columns)
        if n_cols == 0:
            return data

        # Count non-null fields per row using sum of indicator columns
        null_indicators = [
            F.when(F.col(c).isNotNull(), F.lit(1)).otherwise(F.lit(0))
            for c in data.columns
        ]
        non_null_count = sum(null_indicators[1:], null_indicators[0])
        return data.withColumn(
            self._QUALITY_COLUMN,
            (non_null_count / F.lit(n_cols)).cast(T.DoubleType())
        )

class AddAuditColumnsStep(AbstractAddAuditColumnsStep[SparkDataFrame]):
    """Spark: Append lineage-tracking columns (_original_index, _cleaned_at)."""

    _ORIGINAL_INDEX_COL: Final[str] = "_original_index"
    _CLEANED_AT_COL:     Final[str] = "_cleaned_at"

    def __init__(self, data_frame: SparkDataFrame) -> None:
        super().__init__(data_frame)

    def process(self, data: SparkDataFrame) -> SparkDataFrame:
        df = data
        if self._ORIGINAL_INDEX_COL not in df.columns:
            df = df.withColumn(
                self._ORIGINAL_INDEX_COL,
                F.monotonically_increasing_id()
            )
        df = df.withColumn(self._CLEANED_AT_COL, F.current_timestamp())
        return df

# ─────────────────────────────────────────────────────────────────────────────
# Steps: Feature Scaling
# ─────────────────────────────────────────────────────────────────────────────

class StandardScalerStep(AbstractStandardScalerStep[SparkDataFrame]):
    """Spark: Z-score normalization with fit/transform separation.

    fit() computes μ and σ via Spark aggregation (one ACTION).
    transform() applies z = (x - μ) / σ as a pure withColumn expression (lazy).
    """

    def __init__(
        self,
        data_frame: SparkDataFrame,
        columns: Optional[List[str]] = None,
    ) -> None:
        super().__init__(data_frame)
        self.columns    = columns
        self._stats:     Dict[str, Dict[str, float]] = {}
        self._is_fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(self, data: SparkDataFrame) -> "StandardScalerStep":
        cols = self._resolve_columns(data)
        if not cols:
            self._is_fitted = True
            return self

        # [ACTION puntual: una sola agregación para todas las columnas]
        agg_exprs = []
        for col in cols:
            agg_exprs.append(F.mean(F.col(col)).alias(f"{col}__mean"))
            agg_exprs.append(F.stddev(F.col(col)).alias(f"{col}__std"))

        stats_row = data.agg(*agg_exprs).collect()[0].asDict()

        for col in cols:
            self._stats[col] = {
                "mean": float(stats_row.get(f"{col}__mean") or 0.0),
                "std":  float(stats_row.get(f"{col}__std")  or 0.0),
            }
        self._is_fitted = True
        return self

    def transform(self, data: SparkDataFrame) -> SparkDataFrame:
        if not self._is_fitted:
            raise RuntimeError(
                "StandardScalerStep.transform() called before fit(). "
                "Call fit(training_data) first."
            )
        df = data
        for col, stats in self._stats.items():
            if col not in df.columns:
                continue
            mean = stats["mean"]
            std  = stats["std"]
            if std > 0:
                df = df.withColumn(col, (F.col(col) - F.lit(mean)) / F.lit(std))
            else:
                df = df.withColumn(col, F.col(col) - F.lit(mean))
        return df

    def process(self, data: SparkDataFrame) -> SparkDataFrame:
        return self.fit(data).transform(data)

    def _resolve_columns(self, data: SparkDataFrame) -> List[str]:
        if self.columns is not None:
            return [c for c in self.columns if c in data.columns]
        return [
            c for c in data.columns
            if isinstance(data.schema[c].dataType, (T.DoubleType, T.FloatType, T.IntegerType, T.LongType))
        ]
