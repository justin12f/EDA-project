"""Data Cleaning Pipeline — backend-agnostic modular pipeline."""

from __future__ import annotations

from typing import Any, Optional

from lumen.data_cleaning.data_cleaning_report import DataCleaningReport
from lumen.data_cleaning.data_cleaning_step_factory import DataCleaningStepFactory
from lumen.data_cleaning.steps.backends.abstract_steps import AbstractBaseStep
from lumen.data_cleaning.wrapper_steps_with_logger import wrapper_steps_with_logger


class DataCleaningPipeline:
    """Runs a sequence of cleaning steps on a backend-native frame."""

    def __init__(self, step_list: list[AbstractBaseStep[Any]]) -> None:
        self.report = DataCleaningReport()
        self._step_list = [
            wrapper_steps_with_logger(step, self.report) for step in step_list
        ]

    def run(self, data: Any) -> Any:
        data_frame = data
        # Every step's own `process(self, data: pl.DataFrame)` is typed against
        # the eager frame — the polars backend's column-dtype checks and null
        # inspection index it directly (`data[col]`), which `LazyFrame` does
        # not support. A caller resolving a dataset handle gets a LazyFrame
        # back (materialize.py reads Parquet lazily), so this is where that
        # laziness is meant to end: once, before any step sees the frame,
        # rather than each step guessing whether to collect first.
        try:
            import polars as pl

            if isinstance(data_frame, pl.LazyFrame):
                data_frame = data_frame.collect()
        except ImportError:
            pass
        if hasattr(data_frame, "copy"):
            data_frame = data_frame.copy()
        for step in self._step_list:
            data_frame = step.process(data_frame)
        return data_frame


class PipelineBuilder:
    """Build a DataCleaningPipeline from declarative configuration."""

    def __init__(self, data_frame: Any) -> None:
        self._data_frame = data_frame

    def build(
        self,
        configuration: list[dict[str, Optional[dict]]],
    ) -> DataCleaningPipeline:
        step_list = [self._build_step(entry) for entry in configuration]
        return DataCleaningPipeline(step_list)

    def _build_step(self, entry: dict[str, Optional[dict]]) -> AbstractBaseStep[Any]:
        if len(entry) != 1:
            raise ValueError(
                f"Each entry must have exactly one key. Received: {entry}"
            )

        step_name, kwargs = next(iter(entry.items()))
        kwargs = (kwargs or {}).copy()
        columns: list[str] | None = kwargs.pop("columns", None)

        if columns is not None:
            return DataCleaningStepFactory.create_scoped(
                step_name, self._data_frame, columns, **kwargs
            )

        return DataCleaningStepFactory.create(step_name, self._data_frame, **kwargs)
