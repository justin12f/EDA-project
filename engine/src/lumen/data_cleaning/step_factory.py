"""Backend-scoped data cleaning step factories."""

from __future__ import annotations

from typing import Any, Generic, TypeVar

from lumen.core.abstract_factory import RegistryFactory
from lumen.core.backend import Backend, validate_backend
from lumen.data_cleaning.steps.backends.abstract_steps import (
    AbstractBaseStep,
    AbstractColumnScopedStep,
)

T = TypeVar("T")


class AbstractDataCleaningStepFactory(RegistryFactory[str, AbstractBaseStep[Any]]):
    """Registry ``(step_name, backend)`` → step class. No pandas/polars/spark imports."""

    @classmethod
    def create(
        cls,
        step_name: str,
        data_frame: Any,
        backend: Backend | str | None = None,
        **kwargs: Any,
    ) -> AbstractBaseStep[Any]:
        if backend is not None:
            validate_backend(str(backend))
            return super().create(step_name, backend, data_frame, **kwargs)
        return super().create(step_name, cls._resolve_backend(data_frame), data_frame, **kwargs)

    @classmethod
    def create_scoped(
        cls,
        step_name: str,
        data_frame: Any,
        columns: list[str],
        backend: Backend | str | None = None,
        **kwargs: Any,
    ) -> AbstractBaseStep[Any]:
        active = backend or cls._resolve_backend(data_frame)
        inner = cls.create(step_name, data_frame, backend=active, **kwargs)
        scoped_cls = cls.get_class("column_scoped", active)
        return scoped_cls(inner_step=inner, data_frame=data_frame, columns=columns)

    @staticmethod
    def _resolve_backend(data_frame: Any) -> str:
        module = type(data_frame).__module__
        name = type(data_frame).__name__
        if "polars" in module:
            return "polars"
        if "pyspark" in module:
            return "spark"
        if "pandas" in module or name == "DataFrame":
            return "pandas"
        raise TypeError(
            f"Cannot infer backend from frame type {type(data_frame)!r}"
        )


class PandasDataCleaningStepFactory(AbstractDataCleaningStepFactory):
    """Pandas backend step factory."""


class PolarsDataCleaningStepFactory(AbstractDataCleaningStepFactory):
    """Polars backend step factory."""


class SparkDataCleaningStepFactory(AbstractDataCleaningStepFactory):
    """Spark backend step factory."""


def _register_backend_steps(
    factory: type[AbstractDataCleaningStepFactory],
    backend: Backend,
    module: Any,
) -> None:
    steps = {
        "column_scoped": module.ColumnScopedStep,
        "fix_columns_titles": module.ColumnsTitlesStep,
        "enforce_schema": module.EnforceSchemaStep,
        "handle_sentinel_values": module.HandleSentinelValuesStep,
        "drop_high_missing_columns": module.DropHighMissingColumnsStep,
        "drop_constant_columns": module.DropConstantColumnsStep,
        "safe_conversion": module.SafeConversionStep,
        "fix_numeric_columns": module.FixNumericColumnsStep,
        "fix_bools_columns": module.FixBoolsColumnsStep,
        "fix_dates_columns": module.FixDatesColumnsStep,
        "fix_columns_types": module.FixColumnsTypesStep,
        "impute_categorical": module.ImputeCategoricalStep,
        "iqr_outlier": module.IQROutlierStep,
        "zscore_outlier": module.ZScoreOutlierStep,
        "cap_outliers": module.CapOutliersStep,
        "standard_scaler": module.StandardScalerStep,
        "fix_not_numeric_columns": module.FixNotNumericColumnsStep,
        "normalize_categories": module.NormalizeCategoriesStep,
        "text_standardization": module.TextStandardizationStep,
        "validate_domain_rules": module.ValidateDomainRulesStep,
        "cross_column_validation": module.CrossColumnValidationStep,
        "remove_duplicates_rows": module.RemoveDuplicatesRowsStep,
        "flag_data_quality": module.FlagDataQualityStep,
        "add_audit_columns": module.AddAuditColumnsStep,
        "handle_outliers": module.ZScoreOutlierStep,
    }
    for name, step_cls in steps.items():
        factory.register(name, backend, step_cls)


def _bootstrap_registries() -> None:
    from lumen.data_cleaning.steps.backends import pandas_impl, polars_impl

    _register_backend_steps(AbstractDataCleaningStepFactory, "pandas", pandas_impl)
    _register_backend_steps(AbstractDataCleaningStepFactory, "polars", polars_impl)

    # PySpark is an optional dependency (the ``spark`` extra) — the API and
    # worker install ``lumen-engine`` without it. Requesting the spark
    # backend without PySpark installed raises a clear ValueError from
    # RegistryFactory.get_class() instead of a bare ModuleNotFoundError.
    try:
        from lumen.data_cleaning.steps.backends import spark_impl
    except ImportError:
        pass
    else:
        _register_backend_steps(AbstractDataCleaningStepFactory, "spark", spark_impl)


_bootstrap_registries()

# Unified registry (delegates to backend-specific factories)
DataCleaningStepFactory = AbstractDataCleaningStepFactory
