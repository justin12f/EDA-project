"""Per-column self-calibrating baselines (ADR-0013) — every case here must
resolve with no model call, same discipline as test_sentinel_drift.py.
"""

from __future__ import annotations

import pytest

from lumen.sentinel import (
    MIN_SAMPLE_SIZE,
    CategoricalBaseline,
    NumericBaseline,
    categorical_shift,
    classify_baseline_kind,
)


class TestClassifyBaselineKind:
    @pytest.mark.parametrize("dtype", ["int64", "Int32", "float64", "double", "Decimal(10,2)"])
    def test_numeric_dtypes_are_numeric(self, dtype):
        assert classify_baseline_kind(dtype, non_null_count=100, distinct_count=50) == "numeric"

    @pytest.mark.parametrize("dtype", ["datetime64[ns]", "Date", "Timestamp", "time"])
    def test_datetime_dtypes_are_datetime_even_with_high_cardinality(self, dtype):
        # A timestamp column is normally near-100% unique — that must not
        # push it into "freeform_string" the way a high-cardinality string
        # would; dtype wins over the cardinality heuristic.
        assert classify_baseline_kind(dtype, non_null_count=1000, distinct_count=999) == "datetime"

    def test_a_low_cardinality_string_is_categorical(self):
        # 5 distinct values across 1000 non-null rows: a country code.
        assert classify_baseline_kind("string", non_null_count=1000, distinct_count=5) == "categorical"

    def test_a_high_cardinality_string_is_freeform(self):
        # 950 distinct values across 1000 non-null rows: an email address.
        assert classify_baseline_kind("string", non_null_count=1000, distinct_count=950) == "freeform_string"

    def test_an_empty_string_column_defaults_to_categorical_not_a_crash(self):
        assert classify_baseline_kind("string", non_null_count=0, distinct_count=0) == "categorical"

    def test_the_cardinality_boundary_is_exact(self):
        # Exactly at the 0.2 ratio: categorical (the check is <=, not <).
        assert classify_baseline_kind("string", non_null_count=100, distinct_count=20) == "categorical"
        assert classify_baseline_kind("string", non_null_count=100, distinct_count=21) == "freeform_string"


class TestNumericBaseline:
    def test_a_fresh_baseline_is_not_calibrated(self):
        assert NumericBaseline().is_calibrated() is False

    def test_cold_start_never_flags_a_value_as_out_of_band(self):
        baseline = NumericBaseline()
        for _ in range(MIN_SAMPLE_SIZE - 1):
            baseline = baseline.updated(10.0)
        # Wildly different from the (short) history so far — still not
        # flagged, because there isn't enough history to have an opinion.
        assert baseline.is_within_band(10_000.0) is True

    def test_becomes_calibrated_at_exactly_the_minimum_sample_size(self):
        baseline = NumericBaseline()
        for _ in range(MIN_SAMPLE_SIZE - 1):
            baseline = baseline.updated(10.0)
            assert baseline.is_calibrated() is False
        baseline = baseline.updated(10.0)
        assert baseline.is_calibrated() is True

    def test_a_stable_series_flags_a_sharp_one_off_deviation(self):
        baseline = NumericBaseline()
        for _ in range(20):
            baseline = baseline.updated(100.0)
        assert baseline.stdev == pytest.approx(0.0)
        assert baseline.is_within_band(100.0) is True
        assert baseline.is_within_band(500.0) is False

    def test_a_naturally_volatile_series_tolerates_its_own_normal_swing(self):
        # Alternating 90/110 around a mean of 100 - the swing itself is
        # what "normal" looks like for this column, unlike the stable case
        # above. The whole point of a per-column band (ADR-0013's own
        # `discount_pct` example): the same absolute value (110) must read
        # differently depending on which column it showed up in.
        baseline = NumericBaseline()
        for i in range(20):
            baseline = baseline.updated(90.0 if i % 2 == 0 else 110.0)
        assert baseline.is_within_band(110.0) is True
        assert baseline.is_within_band(90.0) is True

    def test_adapts_to_a_sustained_genuine_shift(self):
        # Action Item 7's first fixture: a real, sustained business change
        # (e.g. a pricing change moving a discount column's typical level)
        # must eventually stop being flagged as it becomes the new normal.
        baseline = NumericBaseline()
        for _ in range(20):
            baseline = baseline.updated(100.0)
        assert baseline.is_within_band(140.0) is False  # not yet adapted

        for _ in range(30):
            baseline = baseline.updated(140.0)
        assert baseline.is_within_band(140.0) is True  # adapted

    def test_still_catches_a_short_sharp_incident_after_a_long_history(self):
        # Action Item 7's second fixture: a long, stable history must not
        # make the gate numb to a real incident — the opposite failure mode
        # from a lifetime average that a hundred more scans barely moves.
        baseline = NumericBaseline()
        for _ in range(200):
            baseline = baseline.updated(100.0)
        assert baseline.is_within_band(100.0) is True
        assert baseline.is_within_band(1000.0) is False

    def test_control_limits_widen_with_a_larger_z(self):
        baseline = NumericBaseline()
        for i in range(20):
            baseline = baseline.updated(100.0 + (i % 2))
        narrow = baseline.control_limits(z=1.0)
        wide = baseline.control_limits(z=6.0)
        assert wide[0] < narrow[0] < narrow[1] < wide[1]


class TestCategoricalBaseline:
    def test_a_fresh_baseline_is_not_calibrated(self):
        assert CategoricalBaseline().is_calibrated() is False

    def test_an_empty_scan_does_not_advance_the_baseline(self):
        baseline = CategoricalBaseline()
        updated = baseline.updated({}, 0)
        assert updated == baseline

    def test_converges_toward_the_observed_distribution(self):
        baseline = CategoricalBaseline()
        for _ in range(30):
            baseline = baseline.updated({"US": 80, "DE": 20}, 100)
        assert baseline.frequencies["US"] == pytest.approx(0.8, abs=0.02)
        assert baseline.frequencies["DE"] == pytest.approx(0.2, abs=0.02)


class TestCategoricalShift:
    def _calibrated(self) -> CategoricalBaseline:
        baseline = CategoricalBaseline()
        for _ in range(10):
            baseline = baseline.updated({"US": 70, "DE": 20, "FR": 10}, 100)
        return baseline

    def test_cold_start_reports_nothing(self):
        baseline = CategoricalBaseline().updated({"US": 100}, 100)
        assert categorical_shift(baseline, {"US": 50, "DE": 50}, 100) == []

    def test_a_stable_distribution_reports_nothing(self):
        baseline = self._calibrated()
        assert categorical_shift(baseline, {"US": 70, "DE": 20, "FR": 10}, 100) == []

    def test_a_large_share_move_is_reported(self):
        baseline = self._calibrated()
        # FR moves 10% -> 50%; since shares sum to 100%, US necessarily
        # drops too (70% -> 30%) - both are genuine >=15-point moves and
        # both must be reported, not just the one the test happens to name.
        findings = {f.category: f for f in categorical_shift(baseline, {"US": 30, "DE": 20, "FR": 50}, 100)}
        assert set(findings) == {"US", "FR"}
        assert findings["FR"].kind == "share_shift"
        assert findings["FR"].baseline_share == pytest.approx(0.10)
        assert findings["FR"].current_share == pytest.approx(0.50)
        assert findings["US"].baseline_share == pytest.approx(0.70)
        assert findings["US"].current_share == pytest.approx(0.30)

    def test_a_new_category_at_volume_is_reported(self):
        baseline = self._calibrated()
        findings = categorical_shift(baseline, {"US": 60, "DE": 20, "FR": 10, "BR": 10}, 100)
        assert len(findings) == 1
        assert findings[0].category == "BR"
        assert findings[0].kind == "new_category"
        assert findings[0].baseline_share is None

    def test_a_negligible_new_category_is_not_reported(self):
        baseline = self._calibrated()
        findings = categorical_shift(baseline, {"US": 69, "DE": 20, "FR": 10, "JP": 1}, 100)
        assert findings == []

    def test_a_category_vanishing_entirely_is_reported_as_a_share_shift(self):
        # DE (baseline 20%) does not appear in this scan's counts at all -
        # a 20%-to-0% move, "far below its historical share" (ADR-0013 §1)
        # even though it is absent, not merely reduced.
        baseline = self._calibrated()
        findings = categorical_shift(baseline, {"US": 80, "FR": 20}, 100)
        vanished = [f for f in findings if f.category == "DE"]
        assert len(vanished) == 1
        assert vanished[0].kind == "share_shift"
        assert vanished[0].current_share == 0.0
        assert vanished[0].baseline_share == pytest.approx(0.20)
