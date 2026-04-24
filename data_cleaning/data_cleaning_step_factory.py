"""Module containing abstract base classes and step implementations for data cleaning."""

import pandas as pd

from .steps.base import BaseStep
from .steps.implementations import (
    AddAuditColumnsStep,
    CapOutliersStep,
    ColumnsTitlesStep,
    CrossColumnValidationStep,
    DropConstantColumnsStep,
    DropHighMissingColumnsStep,
    EnforceSchemaStep,
    FixBoolsColumnsStep,
    FixColumnsTypesStep,
    FixDatesColumnsStep,
    FixNotNumericColumnsStep,
    FixNumericColumnsStep,
    FlagDataQualityStep,
    HandleOutliersStep,
    HandleSentinelValuesStep,
    ImputeCategoricalStep,
    NormalizeCategoriesStep,
    RemoveDuplicatesRowsStep,
    SafeConversionStep,
    TextStandardizationStep,
    ValidateDomainRulesStep,
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
DataCleaningStepFactory.register(
    "drop_high_missing_columns", DropHighMissingColumnsStep
)
DataCleaningStepFactory.register("drop_constant_columns", DropConstantColumnsStep)
DataCleaningStepFactory.register("text_standardization", TextStandardizationStep)
DataCleaningStepFactory.register("cap_outliers", CapOutliersStep)
DataCleaningStepFactory.register("impute_categorical", ImputeCategoricalStep)
DataCleaningStepFactory.register("enforce_schema", EnforceSchemaStep)
DataCleaningStepFactory.register("add_audit_columns", AddAuditColumnsStep)
