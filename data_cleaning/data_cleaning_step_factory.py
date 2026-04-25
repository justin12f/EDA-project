"""Module containing abstract base classes and step implementations for data cleaning."""

import pandas as pd

from data_cleaning.steps.base import BaseStep
from data_cleaning.steps.implementations import (
    ColumnScopedStep,
    ColumnsTitlesStep,
    EnforceSchemaStep,
    DropHighMissingColumnsStep,
    DropConstantColumnsStep,
    HandleSentinelValuesStep,
    ImputeCategoricalStep,
    SafeConversionStep,
    FixDatesColumnsStep,
    FixBoolsColumnsStep,
    FixColumnsTypesStep,
    FixNumericColumnsStep,
    IQROutlierStep,
    ZScoreOutlierStep,
    CapOutliersStep,
    StandardScalerStep,
)


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

    @classmethod
    @classmethod
    def create_scoped(
        cls,
        step_name: str,
        data_frame: pd.DataFrame,
        columns: list[str],
        **kwargs,
    ) -> BaseStep:
        """
        Instantiate a registered step wrapped in ColumnScopedStep.
        """
        inner_step = cls.create(step_name, data_frame, **kwargs)
        return ColumnScopedStep(
            inner_step=inner_step,
            data_frame=data_frame,
            columns=columns
        )

# ── CORE / ESTRUCTURA ─────────────────────────────────
DataCleaningStepFactory.register("fix_columns_titles", ColumnsTitlesStep)
DataCleaningStepFactory.register("enforce_schema", EnforceSchemaStep)
DataCleaningStepFactory.register("handle_sentinel_values", HandleSentinelValuesStep)
DataCleaningStepFactory.register("drop_high_missing_columns", DropHighMissingColumnsStep)
DataCleaningStepFactory.register("drop_constant_columns", DropConstantColumnsStep)

# ── CONVERSIÓN / TIPOS ────────────────────────────────
DataCleaningStepFactory.register("safe_conversion", SafeConversionStep)
DataCleaningStepFactory.register("fix_numeric_columns", FixNumericColumnsStep)
DataCleaningStepFactory.register("fix_bools_columns", FixBoolsColumnsStep)
DataCleaningStepFactory.register("fix_dates_columns", FixDatesColumnsStep)
DataCleaningStepFactory.register("fix_columns_types", FixColumnsTypesStep)

# ── IMPUTACIÓN ────────────────────────────────────────
DataCleaningStepFactory.register("impute_categorical", ImputeCategoricalStep)

# ── OUTLIERS ──────────────────────────────────────────
DataCleaningStepFactory.register("iqr_outlier", IQROutlierStep)
DataCleaningStepFactory.register("zscore_outlier", ZScoreOutlierStep)
DataCleaningStepFactory.register("cap_outliers", CapOutliersStep)

# ── ML / SCALING ──────────────────────────────────────
DataCleaningStepFactory.register("standard_scaler", StandardScalerStep)
