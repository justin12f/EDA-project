"""Wrapper that injects logging / metrics collection into any pipeline step."""

import time
from pandas import DataFrame
from data_cleaning.data_cleaning_report import DataCleaningReport, compare_metrics
from data_cleaning.data_cleaning_step_factory import BaseStep


def wrapper_steps_with_logger(
    step: BaseStep, report: DataCleaningReport
) -> BaseStep:  # BUG-07 fixed: was -> None
    """Wrap a step's process() to capture before/after metrics."""
    original_process = step.process

    def wrapped(data: DataFrame) -> DataFrame:
        start_time = time.time()
        before = data.copy()
        
        try:
            result = original_process(data)
        except Exception as e:
            raise RuntimeError(f"Error in step '{step.__class__.__name__}': {str(e)}") from e
            
        elapsed_ms = (time.time() - start_time) * 1000
        metrics = compare_metrics(before, result)
        metrics["elapsed_ms"] = elapsed_ms
        report.add_steps(step.__class__.__name__, metrics)
        return result

    step.process = wrapped
    return step
