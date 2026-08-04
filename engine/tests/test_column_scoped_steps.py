"""Column-scoped cleaning steps.

Every one of the 24 registered steps was unusable through a proposal until the
fix these tests guard. Three defects with one root cause: the abstract contracts
declared `@abstractmethod def __init__(...) -> None: ...`, meaning to document a
signature. Python treated the `...` as a real body that wins in the MRO, so:

1. `AbstractBaseStep.__init__` never ran and `_data_frame` was never assigned —
   silently, in 19 of 24 steps.
2. `AbstractColumnScopedStep` leads with `inner_step` rather than `data_frame`,
   so the concrete decorators' `super().__init__(data_frame)` raised, and every
   scoped construction failed.
3. Once construction worked, the report layer died on polars — the default
   backend — because `compare_metrics` was written against pandas.
"""

from __future__ import annotations

import pandas as pd
import polars as pl
import pytest

from lumen.data_cleaning.data_cleaning_pipeline import PipelineBuilder
from lumen.data_cleaning.data_cleaning_report import compare_metrics
from lumen.data_cleaning.step_factory import AbstractDataCleaningStepFactory as Factory

BACKENDS = ("pandas", "polars")


def frame(backend: str):
    data = {
        "country_code": ["DE", None, "US", None, "FR"],
        "email_hash": ["a1", "b2", "a1", "c3", "d4"],
        "untouched": [1, 2, 3, 4, 5],
    }
    return pd.DataFrame(data) if backend == "pandas" else pl.DataFrame(data)


def steps_for(backend: str) -> list[str]:
    return [
        key
        for key in sorted(Factory.registered_keys())
        if Factory.is_registered(key, backend) and key != "column_scoped"
    ]


# ── construction ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("backend", BACKENDS)
def test_every_registered_step_constructs(backend):
    for key in steps_for(backend):
        Factory.create(key, frame(backend), backend=backend)


@pytest.mark.parametrize("backend", BACKENDS)
def test_every_step_stores_the_frame_it_was_given(backend):
    """The silent half of the defect: construction succeeded, state did not."""
    offenders = [
        key
        for key in steps_for(backend)
        if not hasattr(Factory.create(key, frame(backend), backend=backend), "_data_frame")
    ]
    assert offenders == [], f"steps that never stored _data_frame: {offenders}"


@pytest.mark.parametrize("backend", BACKENDS)
def test_every_registered_step_constructs_scoped(backend):
    """The loud half: this raised for all 24 steps on both backends."""
    offenders = []
    for key in steps_for(backend):
        try:
            Factory.create_scoped(key, frame(backend), ["country_code"], backend=backend)
        except Exception as exc:  # noqa: BLE001
            offenders.append(f"{key}: {type(exc).__name__}")
    assert offenders == [], f"scoped construction failed for: {offenders}"


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_scoped_step_rejects_an_empty_column_list(backend):
    with pytest.raises(ValueError, match="non-empty"):
        Factory.create_scoped("impute_categorical", frame(backend), [], backend=backend)


# ── behaviour ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_scoped_pipeline_cleans_only_its_columns(backend):
    source = frame(backend)
    result = PipelineBuilder(source).build(
        [{"impute_categorical": {"columns": ["country_code"], "strategy": "mode"}}]
    ).run(source)

    if backend == "polars":
        assert result["country_code"].null_count() == 0
        assert result["untouched"].to_list() == [1, 2, 3, 4, 5]
        assert result.columns == source.columns
        assert source["country_code"].null_count() == 2, "input must not be mutated"
    else:
        assert result["country_code"].isna().sum() == 0
        assert result["untouched"].tolist() == [1, 2, 3, 4, 5]
        assert list(result.columns) == list(source.columns)
        assert source["country_code"].isna().sum() == 2, "input must not be mutated"


@pytest.mark.parametrize("backend", BACKENDS)
def test_an_unscoped_step_still_runs(backend):
    source = frame(backend)
    result = PipelineBuilder(source).build([{"remove_duplicates_rows": {}}]).run(source)
    height = result.height if backend == "polars" else len(result)
    assert height == 5, "no two rows are identical, so none should be removed"


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_multi_step_pipeline_composes(backend):
    source = frame(backend)
    result = PipelineBuilder(source).build(
        [
            {"impute_categorical": {"columns": ["country_code"], "strategy": "mode"}},
            {"remove_duplicates_rows": {}},
        ]
    ).run(source)

    nulls = (
        result["country_code"].null_count()
        if backend == "polars"
        else result["country_code"].isna().sum()
    )
    assert nulls == 0


def test_an_unknown_step_name_is_rejected_before_anything_runs():
    with pytest.raises(ValueError, match="summon_daemon"):
        PipelineBuilder(frame("polars")).build([{"summon_daemon": {}}])


# ── the report layer ────────────────────────────────────────────────────────


@pytest.mark.parametrize("backend", BACKENDS)
def test_compare_metrics_works_on_both_backends(backend):
    """It ran for every step through the logging wrapper, and was pandas-only."""
    before = frame(backend)
    after = PipelineBuilder(before).build(
        [{"impute_categorical": {"columns": ["country_code"], "strategy": "mode"}}]
    ).run(before)

    metrics = compare_metrics(before, after)

    assert set(metrics) == {
        "rows_removed",
        "nulls_before",
        "nulls_after",
        "changed_columns",
        "change_ratio",
    }
    assert metrics["rows_removed"] == 0
    assert metrics["nulls_before"] == 2
    assert metrics["nulls_after"] == 0
    assert "untouched" in metrics["changed_columns"]
    assert metrics["change_ratio"]["untouched"] == 0.0
    assert metrics["change_ratio"]["country_code"] > 0


def test_compare_metrics_agrees_across_backends():
    results = {}
    for backend in BACKENDS:
        before = frame(backend)
        after = PipelineBuilder(before).build(
            [{"impute_categorical": {"columns": ["country_code"], "strategy": "mode"}}]
        ).run(before)
        results[backend] = compare_metrics(before, after)

    for key in ("rows_removed", "nulls_before", "nulls_after"):
        assert results["pandas"][key] == results["polars"][key], key


def test_compare_metrics_handles_an_empty_frame():
    empty = pl.DataFrame({"a": [], "b": []})
    metrics = compare_metrics(empty, empty)
    assert metrics["rows_removed"] == 0
    assert metrics["change_ratio"] == {"a": 0.0, "b": 0.0}
