"""Module for the data cleaning report and metrics comparison."""

import json
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
            avg_change = (
                sum(change_ratio.values()) / len(change_ratio) if change_ratio else 0.0
            )

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
            avg_change = (
                sum(change_ratio.values()) / len(change_ratio) if change_ratio else 0.0
            )

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


def compare_metrics(before: DataFrame, after: DataFrame) -> dict[str, dict]:
    """Compare two DataFrames and return change metrics."""
    report: dict[str, dict] = {}
    report["rows_removed"] = len(before) - len(after)
    report["nulls_before"] = int(before.isna().sum().sum())
    report["nulls_after"] = int(after.isna().sum().sum())

    # Columns that exist in both (may have been added/removed by a step)
    changed_columns = [col for col in before.columns if col in after.columns]
    report["changed_columns"] = changed_columns

    col_changes: dict[str, float] = {}

    if len(before) == len(before.index.unique()) and len(after) == len(
        after.index.unique()
    ):
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
        return np.array(
            [x is None or (isinstance(x, float) and np.isnan(x)) for x in arr]
        )
