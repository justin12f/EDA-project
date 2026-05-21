"""
data_cleaning/steps/backends/__init__.py

Triple-backend package for data cleaning pipeline steps.

Usage:
    # Pandas (default)
    from data_cleaning.steps.backends.pandas_impl import ColumnsTitlesStep

    # Polars (local-first, lazy evaluation)
    from data_cleaning.steps.backends.polars_impl import ColumnsTitlesStep

    # PySpark (distributed enterprise)
    from data_cleaning.steps.backends.spark_impl import ColumnsTitlesStep

All three backends implement the same abstract contracts defined in
abstract_steps.py and are 100% interchangeable at the call site.
"""

from data_cleaning.steps.backends.abstract_steps import (
    AbstractBaseStep,
    AbstractColumnScopedStep,
    AbstractColumnsTitlesStep,
    AbstractEnforceSchemaStep,
    AbstractDropHighMissingColumnsStep,
    AbstractDropConstantColumnsStep,
    AbstractHandleSentinelValuesStep,
    AbstractImputeCategoricalStep,
    AbstractSafeConversionStep,
    AbstractFixDatesColumnsStep,
    AbstractFixBoolsColumnsStep,
    AbstractFixColumnsTypesStep,
    AbstractFixNumericColumnsStep,
    AbstractIQROutlierStep,
    AbstractZScoreOutlierStep,
    AbstractCapOutliersStep,
    AbstractFixNotNumericColumnsStep,
    AbstractNormalizeCategoriesStep,
    AbstractTextStandardizationStep,
    AbstractValidateDomainRulesStep,
    AbstractCrossColumnValidationStep,
    AbstractRemoveDuplicatesRowsStep,
    AbstractFlagDataQualityStep,
    AbstractAddAuditColumnsStep,
    AbstractStandardScalerStep,
)

__all__ = [
    "AbstractBaseStep",
    "AbstractColumnScopedStep",
    "AbstractColumnsTitlesStep",
    "AbstractEnforceSchemaStep",
    "AbstractDropHighMissingColumnsStep",
    "AbstractDropConstantColumnsStep",
    "AbstractHandleSentinelValuesStep",
    "AbstractImputeCategoricalStep",
    "AbstractSafeConversionStep",
    "AbstractFixDatesColumnsStep",
    "AbstractFixBoolsColumnsStep",
    "AbstractFixColumnsTypesStep",
    "AbstractFixNumericColumnsStep",
    "AbstractIQROutlierStep",
    "AbstractZScoreOutlierStep",
    "AbstractCapOutliersStep",
    "AbstractFixNotNumericColumnsStep",
    "AbstractNormalizeCategoriesStep",
    "AbstractTextStandardizationStep",
    "AbstractValidateDomainRulesStep",
    "AbstractCrossColumnValidationStep",
    "AbstractRemoveDuplicatesRowsStep",
    "AbstractFlagDataQualityStep",
    "AbstractAddAuditColumnsStep",
    "AbstractStandardScalerStep",
]
