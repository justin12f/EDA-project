"""Module for the data cleaning report and metrics comparison."""

# #[AI_CONTEXT_START]
# - CONFIGURACIÓN DE FACTORY: `DataCleaningReportFactory` por backend, inyectada vía `DataCleaningInyeccionDependency` desde la Factory Maestra.
# - ABSTRACCIÓN DEL DATO: Métricas before/after sobre el contenedor del backend, no `DataFrame` pandas fijo ni `np.ndarray` auxiliar.
# - REFACTOR NATIVO: Comparación de métricas con agregaciones nativas del backend activo.
# #[AI_CONTEXT_END]
import json
from typing import Any
import numpy as np
from pandas import DataFrame

class DataCleaningReport:
    """Cleaning report: tracks what changed in each pipeline step."""

    steps: list[dict[str, dict]]

    def __init__(self):
        self.steps = []

    def add_steps(self, name: str, metrics: dict):
        """Add a dictionary with name and metrics of the step to steps."""  # BUG fixed: dictionnary
        self.steps.append({"name": name, "metrics": metrics})

    def summary(self) -> DataFrame:
        """Return general info about the steps as a clean DataFrame."""
        summary_data = []
        for step in self.steps:
            name = step["name"]
            metrics = step["metrics"]
            rows_removed = metrics.get("rows_removed", 0)

            change_ratio = metrics.get("change_ratio", {})
            columns_changed = sum(1 for v in change_ratio.values() if v > 0)
            avg_change = sum(change_ratio.values()) / len(change_ratio) if change_ratio else 0.0

            summary_data.append(
                {
                    "step": name,
                    "rows_removed": rows_removed,
                    "columns_changed": columns_changed,
                    "avg_change": round(avg_change, 3),
                }
            )

        return DataFrame(summary_data)

    def detailed_summary(self) -> DataFrame:
        """Return highly detailed info about the steps, including null counts and timing."""
        summary_data = []
        for step in self.steps:
            name = step["name"]
            metrics = step["metrics"]
            rows_removed = metrics.get("rows_removed", 0)
            nulls_before = metrics.get("nulls_before", 0)
            nulls_after = metrics.get("nulls_after", 0)
            elapsed_ms = metrics.get("elapsed_ms", 0.0)

            change_ratio = metrics.get("change_ratio", {})
            columns_changed = sum(1 for v in change_ratio.values() if v > 0)
            avg_change = sum(change_ratio.values()) / len(change_ratio) if change_ratio else 0.0

            summary_data.append(
                {
                    "step": name,
                    "elapsed_ms": round(elapsed_ms, 2),
                    "rows_removed": rows_removed,
                    "columns_changed": columns_changed,
                    "nulls_before": nulls_before,
                    "nulls_after": nulls_after,
                    "null_diff": nulls_after - nulls_before,
                    "avg_change": round(avg_change, 3),
                }
            )

        return DataFrame(summary_data)

    def print_summary(self) -> None:
        """Print a human-readable table of the detailed summary."""
        df_summary = self.detailed_summary()
        if df_summary.empty:
            print("No steps recorded in the report.")
            return

        print("=" * 100)
        print(" PIPELINE EXECUTION REPORT ")
        print("=" * 100)
        print(df_summary.to_string(index=False))
        print("=" * 100)

    def to_json(self, path: str) -> None:
        """Serialize the report to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.steps, f, indent=4)

def _backend_of(frame: Any) -> str:
    module = type(frame).__module__
    if "polars" in module:
        return "polars"
    if "pyspark" in module:
        return "spark"
    return "pandas"


def compare_metrics(before: Any, after: Any) -> dict[str, dict]:
    """Compare two DataFrames and return change metrics.

    Dispatches on backend. Every branch returns the same keys — `rows_removed`,
    `nulls_before`, `nulls_after`, `changed_columns`, `change_ratio` — so the
    report renderer and every caller stay backend-blind.
    """
    backend = _backend_of(before)
    if backend == "polars":
        return _compare_polars(before, after)
    if backend == "spark":
        return _compare_spark(before, after)

    report: dict[str, dict] = {}
    report["rows_removed"] = len(before) - len(after)
    report["nulls_before"] = int(before.isna().sum().sum())
    report["nulls_after"] = int(after.isna().sum().sum())

    # Columns that exist in both (may have been added/removed by a step)
    changed_columns = [col for col in before.columns if col in after.columns]
    report["changed_columns"] = changed_columns

    col_changes: dict[str, float] = {}

    if len(before) == len(before.index.unique()) and len(after) == len(after.index.unique()):
        # Unique indices — compare by aligned index intersection
        common_index = before.index.intersection(after.index)
        for column in changed_columns:
            if len(common_index) == 0:
                col_changes[column] = 0.0
                continue
            before_vals = before.loc[common_index, column].to_numpy()
            after_vals = after.loc[common_index, column].to_numpy()
            with np.errstate(invalid="ignore"):
                different = before_vals != after_vals
                both_nan = _is_nan_array(before_vals) & _is_nan_array(after_vals)
                diff = (different & ~both_nan).sum()
            col_changes[column] = diff / len(common_index)
    else:
        # Duplicate indices — reset and compare by position up to min length
        b = before.reset_index(drop=True)
        a = after.reset_index(drop=True)
        n = min(len(b), len(a))
        for column in changed_columns:
            before_vals = b.loc[: n - 1, column].to_numpy()
            after_vals = a.loc[: n - 1, column].to_numpy()
            with np.errstate(invalid="ignore"):
                different = before_vals != after_vals
                both_nan = _is_nan_array(before_vals) & _is_nan_array(after_vals)
                diff = (different & ~both_nan).sum()
            col_changes[column] = diff / n

    report["change_ratio"] = col_changes
    return report

def _is_nan_array(arr) -> "np.ndarray":
    """Return a boolean array: True where the element is NaN/NaT/None."""
    try:
        return np.isnan(arr.astype(float))
    except (ValueError, TypeError):
        return np.array([x is None or (isinstance(x, float) and np.isnan(x)) for x in arr])


def _compare_polars(before: Any, after: Any) -> dict[str, Any]:
    """Native polars metrics — no pandas round trip, no materialisation of a
    LazyFrame beyond what the caller already collected."""
    import polars as pl

    before = before.collect() if isinstance(before, pl.LazyFrame) else before
    after = after.collect() if isinstance(after, pl.LazyFrame) else after

    changed_columns = [c for c in before.columns if c in after.columns]
    n = min(before.height, after.height)

    change_ratio: dict[str, float] = {}
    for column in changed_columns:
        if n == 0:
            change_ratio[column] = 0.0
            continue
        b = before[column].head(n)
        a = after[column].head(n)
        # ne_missing treats null != null as False, so an untouched null column
        # does not read as fully rewritten.
        change_ratio[column] = float(b.ne_missing(a).sum()) / n

    return {
        "rows_removed": before.height - after.height,
        "nulls_before": int(sum(before.null_count().row(0))),
        "nulls_after": int(sum(after.null_count().row(0))),
        "changed_columns": changed_columns,
        "change_ratio": change_ratio,
    }


def _compare_spark(before: Any, after: Any) -> dict[str, Any]:
    """Spark metrics. Row-level diffing is deliberately skipped: it would mean a
    shuffle-heavy join per step, and the report is diagnostics, not output."""
    from pyspark.sql import functions as F

    def null_total(frame: Any) -> int:
        if not frame.columns:
            return 0
        row = frame.select(
            [F.sum(F.col(c).isNull().cast("long")).alias(c) for c in frame.columns]
        ).collect()[0]
        return int(sum(v or 0 for v in row))

    changed_columns = [c for c in before.columns if c in after.columns]
    return {
        "rows_removed": before.count() - after.count(),
        "nulls_before": null_total(before),
        "nulls_after": null_total(after),
        "changed_columns": changed_columns,
        "change_ratio": {c: 0.0 for c in changed_columns},
    }
