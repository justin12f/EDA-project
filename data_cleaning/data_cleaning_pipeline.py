"""Data Cleaning Pipeline — generic, column-agnostic modular pipeline."""

from typing import Optional

import pandas as pd

from data_cleaning.data_cleaning_report import DataCleaningReport
from data_cleaning.data_cleaning_step_factory import BaseStep, DataCleaningStepFactory
from data_cleaning.wrapper_steps_with_logger import wrapper_steps_with_logger
from data_cleaning.steps.implementations import ColumnScopedStep  


class DataCleaningPipeline:
    """Data Cleaning Pipeline"""

    def __init__(self, step_list: list[BaseStep]) -> None:
        self.report = DataCleaningReport()
        self._step_list = [
            wrapper_steps_with_logger(step, self.report) for step in step_list
        ]

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """Run the data cleaning pipeline and return the cleaned DataFrame."""
        data_frame = data.copy()
        for step in self._step_list:
            data_frame = step.process(data_frame)
        return data_frame


class PipelineBuilder:
    """
    Build a DataCleaningPipeline from a declarative configuration.
    """

    def __init__(self, data_frame: pd.DataFrame) -> None:
        self._data_frame = data_frame

    def build(
        self,
        configuration: list[dict[str, Optional[dict]]],
    ) -> DataCleaningPipeline:
        """Build and return a DataCleaningPipeline."""
        step_list = [self._build_step(entry) for entry in configuration]
        return DataCleaningPipeline(step_list)

    def _build_step(self, entry: dict[str, Optional[dict]]) -> BaseStep:
        """Build and return a BaseStep."""
        if len(entry) != 1:
            raise ValueError(
                f"Each entry must have exact one key. Received: {entry}"
            )

        step_name, kwargs = next(iter(entry.items()))
        kwargs = (kwargs or {}).copy()

        columns: list[str] | None = kwargs.pop("columns", None)

        step = DataCleaningStepFactory.create(step_name, self._data_frame, **kwargs)

        if columns is not None:
            return ColumnScopedStep(
                inner_step=step,
                data_frame=self._data_frame,
                columns=columns
            )

        return step

# ---------------------------------------------------------------------------
# Generic default configuration — NO hardcoded column names.
# Every step uses auto-detection (columns=None) so it works with any dataset.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Presets for Pipeline Configuration
# ---------------------------------------------------------------------------


preset_light: list[dict[str, Optional[dict]]] = [
    {"enforce_schema": None},
    {"fix_columns_titles": None},
    {"handle_sentinel_values": None},
    {"safe_conversion": None},
    {"drop_constant_columns": None},
    {"fix_not_numeric_columns": None},
    {"remove_duplicates_rows": None},
    {"fix_numeric_columns": {"fixcase": "median"}},
    {"fix_bools_columns": None},
    {"fix_dates_columns": None},
    {"flag_data_quality": None},
    {"fix_columns_types": None},
    {"add_audit_columns": None},
]

default_configuration: list[dict[str, Optional[dict]]] = [
    {"enforce_schema": None},
    {"drop_high_missing_columns": {"threshold": 0.8}},
    {"drop_constant_columns": None},
    {"fix_columns_titles": None},
    {"handle_sentinel_values": None},
    {"safe_conversion": None},
    {"text_standardization": None},
    {"fix_not_numeric_columns": None},
    {"remove_duplicates_rows": None},
    {"impute_categorical": {"strategy": "mode"}},
    {"fix_numeric_columns": {"fixcase": "median"}},
    {"cap_outliers": {"lower_percentile": 0.01, "upper_percentile": 0.99}},
    {"handle_outliers": {"z_threshold": 3.0}},
    {"fix_bools_columns": None},
    {"fix_dates_columns": None},
    {"flag_data_quality": None},
    {"fix_columns_types": None},
    {"add_audit_columns": None},
]

preset_strict: list[dict[str, Optional[dict]]] = [
    {"enforce_schema": {"min_rows": 5}},
    {"drop_high_missing_columns": {"threshold": 0.5}},
    {"drop_constant_columns": None},
    {"fix_columns_titles": None},
    {"handle_sentinel_values": None},
    {"safe_conversion": None},
    {"text_standardization": None},
    {"fix_not_numeric_columns": None},
    {"remove_duplicates_rows": None},
    {"impute_categorical": {"strategy": "fixed", "fill_value": "missing"}},
    {"fix_numeric_columns": {"fixcase": "mean"}},
    {"cap_outliers": {"lower_percentile": 0.05, "upper_percentile": 0.95}},
    {"handle_outliers": {"z_threshold": 2.5}},
    {"fix_bools_columns": None},
    {"fix_dates_columns": None},
    {"flag_data_quality": None},
    {"fix_columns_types": None},
    {"add_audit_columns": None},
]
