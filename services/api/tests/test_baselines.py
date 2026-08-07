"""The pure serialization/parsing helpers in lumen_api/baselines.py — no DB.
The functions that actually read/write column_baselines and
source_baselines (`prepare_and_update_column_baselines`,
`compute_and_remember_source_baseline`) are exercised live in
test_baselines_learning.py instead.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from lumen.sentinel import CategoricalBaseline, NumericBaseline
from lumen_api.baselines import (
    _categorical_from_json,
    _categorical_to_json,
    _lag_seconds,
    _numeric_from_json,
    _numeric_to_json,
)


class TestNumericJson:
    def test_round_trips_through_json(self):
        baseline = NumericBaseline().updated(10.0).updated(20.0).updated(30.0)
        assert _numeric_from_json(_numeric_to_json(baseline)) == baseline

    def test_a_missing_or_empty_blob_is_a_fresh_baseline(self):
        assert _numeric_from_json(None) == NumericBaseline()
        assert _numeric_from_json({}) == NumericBaseline()


class TestCategoricalJson:
    def test_round_trips_through_json(self):
        baseline = CategoricalBaseline().updated({"a": 3, "b": 7}, 10)
        assert _categorical_from_json(_categorical_to_json(baseline)) == baseline

    def test_a_missing_or_empty_blob_is_a_fresh_baseline(self):
        assert _categorical_from_json(None) == CategoricalBaseline()
        assert _categorical_from_json({}) == CategoricalBaseline()


class TestLagSeconds:
    def test_computes_the_gap_to_now(self):
        now = datetime(2026, 1, 2, tzinfo=timezone.utc)
        assert _lag_seconds("2026-01-01 00:00:00+00:00", now) == pytest.approx(86_400.0)

    def test_a_naive_timestamp_is_treated_as_utc(self):
        now = datetime(2026, 1, 1, 1, 0, 0, tzinfo=timezone.utc)
        assert _lag_seconds("2026-01-01 00:00:00", now) == pytest.approx(3600.0)

    def test_an_unparseable_value_returns_none_not_a_crash(self):
        assert _lag_seconds("not-a-real-timestamp", datetime.now(timezone.utc)) is None

    def test_lag_is_never_negative(self):
        # A clock skew, or a future-dated value, floors at zero rather than
        # reporting a nonsensical negative lag.
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        assert _lag_seconds("2026-01-02 00:00:00+00:00", now) == 0.0
