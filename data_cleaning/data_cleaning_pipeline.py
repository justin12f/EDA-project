"""Data Cleaning Pipeline — generic, column-agnostic modular pipeline."""

from typing import Optional
import pandas as pd
from data_cleaning.data_cleaning_step_factory import BaseStep, DataCleaningStepFactory
from data_cleaning.data_cleaning_report import DataCleaningReport
from data_cleaning.wrapper_steps_with_logger import wrapper_steps_with_logger


class DataCleaningPipeline:
    """Data Cleaning Pipeline"""

    def __init__(self, step_list: list[BaseStep]) -> None:
        self.report = DataCleaningReport()
        self._step_list = [
            wrapper_steps_with_logger(step, self.report)
            for step in step_list
        ]

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """Run the data cleaning pipeline and return the cleaned DataFrame."""
        data_frame = data.copy()
        for step in self._step_list:
            data_frame = step.process(data_frame)
        return data_frame


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


def build_pipeline(
    data_frame: pd.DataFrame,
    configuration: Optional[list[dict[str, Optional[dict]]]] = None,
) -> DataCleaningPipeline:
    """
    Build and return a DataCleaningPipeline.

    Pass a custom `configuration` list to override the defaults.
    Each item is a dict with one key (step name) and an optional kwargs dict.

    Example::

        pipeline = build_pipeline(df, configuration=[
            {"fix_columns_titles": None},
            {"validate_domain_rules": {"rules": {"age": [0, 120]}}},
            {"fix_numeric_columns": {"fixcase": "mean"}},
        ])
    """
    if configuration is None:
        configuration = default_configuration

    step_list: list[BaseStep] = []
    for step_entry in configuration:
        for step_name, kwargs in step_entry.items():
            if kwargs:
                step_list.append(
                    DataCleaningStepFactory.create(step_name, data_frame, **kwargs)
                )
            else:
                step_list.append(DataCleaningStepFactory.create(step_name, data_frame))

    return DataCleaningPipeline(step_list)

def build_pipeline_from_preset(
    data_frame: pd.DataFrame,
    preset: str = "default",
) -> DataCleaningPipeline:
    """Build a pipeline using one of the predefined configurations ('light', 'default', 'strict')."""
    presets = {
        "light": preset_light,
        "default": default_configuration,
        "strict": preset_strict,
    }
    config = presets.get(preset, default_configuration)
    return build_pipeline(data_frame, configuration=config)
