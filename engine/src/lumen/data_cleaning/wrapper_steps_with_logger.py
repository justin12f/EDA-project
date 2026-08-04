"""Wrapper that injects logging / metrics collection into any pipeline step."""

from __future__ import annotations

import time
from typing import Any

from lumen.data_cleaning.data_cleaning_report import DataCleaningReport, compare_metrics
from lumen.data_cleaning.steps.backends.abstract_steps import AbstractBaseStep


def wrapper_steps_with_logger(
    step: AbstractBaseStep[Any], report: DataCleaningReport
) -> AbstractBaseStep[Any]:
    """Wrap a step's process() to capture before/after metrics."""
    original_process = step.process

    def wrapped(data: Any) -> Any:
        start_time = time.time()
        before = data.copy() if hasattr(data, "copy") else data

        try:
            result = original_process(data)
        except Exception as exc:
            raise RuntimeError(
                f"Error in step '{step.__class__.__name__}': {exc}"
            ) from exc

        elapsed_ms = (time.time() - start_time) * 1000
        metrics = compare_metrics(before, result)
        metrics["elapsed_ms"] = elapsed_ms
        report.add_steps(step.__class__.__name__, metrics)
        return result

    step.process = wrapped  # type: ignore[method-assign]
    return step
