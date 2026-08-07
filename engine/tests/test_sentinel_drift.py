"""Deterministic drift detection (ADR-0008) — every case here must resolve
with no model call, because that is the whole point of the deterministic gate.
"""

from __future__ import annotations

import pytest

from lumen.sentinel import NumericBaseline, detect_drift, diff_schema, null_rate_deltas


def test_an_unchanged_schema_has_no_diff():
    schema = {"id": "int64", "email": "string", "country": "string"}
    assert diff_schema(schema, schema) == []


def test_a_new_column_is_reported_as_added():
    old = {"id": "int64", "email": "string"}
    new = {"id": "int64", "email": "string", "phone": "string"}
    changes = diff_schema(old, new)
    assert len(changes) == 1
    assert changes[0].kind == "added"
    assert changes[0].column == "phone"


def test_a_missing_column_is_reported_as_removed():
    old = {"id": "int64", "email": "string", "fax": "string"}
    new = {"id": "int64", "email": "string"}
    changes = diff_schema(old, new)
    assert len(changes) == 1
    assert changes[0].kind == "removed"
    assert changes[0].column == "fax"


def test_a_similar_name_swap_reads_as_a_rename_not_add_plus_remove():
    old = {"id": "int64", "country": "string"}
    new = {"id": "int64", "country_code": "string"}
    changes = diff_schema(old, new)
    assert len(changes) == 1
    assert changes[0].kind == "renamed"
    assert changes[0].previous_column == "country"
    assert changes[0].column == "country_code"


def test_an_unrelated_add_and_remove_is_not_forced_into_a_rename():
    old = {"id": "int64", "legacy_flag": "bool"}
    new = {"id": "int64", "revenue_usd": "float64"}
    changes = diff_schema(old, new)
    kinds = {c.kind for c in changes}
    assert kinds == {"added", "removed"}, "unrelated names must not pair up as a rename"


def test_a_type_change_is_reported_on_an_otherwise_stable_column():
    old = {"id": "int64", "amount": "string"}
    new = {"id": "int64", "amount": "float64"}
    changes = diff_schema(old, new)
    assert len(changes) == 1
    assert changes[0].kind == "type_changed"
    assert changes[0].column == "amount"


def test_null_rate_deltas_ignores_columns_below_threshold():
    old = {"email": 0.10, "country": 0.05}
    new = {"email": 0.12, "country": 0.05}  # 2-point move, under the 5-point default
    assert null_rate_deltas(old, new) == []


def test_null_rate_deltas_reports_a_real_shift_either_direction():
    old = {"email": 0.10, "country": 0.05}
    new = {"email": 0.40, "country": 0.00}  # up 30pt, down 5pt
    shifts = {s.column: s.delta for s in null_rate_deltas(old, new)}
    assert shifts["email"] == pytest.approx(0.30)
    assert shifts["country"] == pytest.approx(-0.05)


def test_null_rate_deltas_skips_a_column_that_no_longer_exists():
    """A vanished column is diff_schema's finding, not this one's — it cannot
    have "drifted null" if there is no current value to compare."""
    old = {"email": 0.10, "fax": 0.90}
    new = {"email": 0.10}
    assert null_rate_deltas(old, new) == []


def test_detect_drift_returns_none_when_nothing_moved():
    schema = {"id": "int64", "email": "string"}
    rates = {"id": 0.0, "email": 0.1}
    assert detect_drift(schema, schema, rates, rates) is None


def test_detect_drift_classifies_schema_change_over_statistical_shift():
    old_schema = {"id": "int64", "country": "string"}
    new_schema = {"id": "int64", "country_code": "string"}
    rates = {"id": 0.0, "country": 0.05}
    # Only country -> country_code moved; null rates are stable for what's
    # comparable (id). A schema change alongside no statistical shift must
    # still classify as schema_change, not fall through to "nothing found".
    result = detect_drift(old_schema, new_schema, rates, {"id": 0.0, "country_code": 0.05})
    assert result is not None
    assert result.kind == "schema_change"
    assert result.severity > 0


def test_detect_drift_classifies_statistical_shift_when_schema_is_stable():
    schema = {"id": "int64", "email": "string"}
    old_rates = {"id": 0.0, "email": 0.05}
    new_rates = {"id": 0.0, "email": 0.55}
    result = detect_drift(schema, schema, old_rates, new_rates)
    assert result is not None
    assert result.kind == "statistical_shift"


def test_severity_scales_with_how_much_changed_not_flat_on_any_change():
    schema = {"id": "int64", "email": "string"}
    small_shift = detect_drift(
        schema, schema, {"id": 0.0, "email": 0.10}, {"id": 0.0, "email": 0.16}
    )
    large_shift = detect_drift(
        schema, schema, {"id": 0.0, "email": 0.10}, {"id": 0.0, "email": 0.90}
    )
    assert small_shift.severity < large_shift.severity


def test_as_details_is_plain_json_safe_data():
    schema = {"id": "int64", "email": "string"}
    result = detect_drift(schema, {"id": "int64"}, {"id": 0.0, "email": 0.1}, {"id": 0.0})
    details = result.as_details()
    assert isinstance(details["schema_changes"], list)
    assert isinstance(details["schema_changes"][0], dict)


# ── calibrated null-rate baselines (ADR-0013) ───────────────────────────────


def _calibrated(*values: float) -> NumericBaseline:
    baseline = NumericBaseline()
    for value in values:
        baseline = baseline.updated(value)
    return baseline


def test_a_stable_columns_baseline_flags_a_move_smaller_than_the_flat_threshold():
    # A column that has always sat at ~2% null: a jump to 8% is a real
    # deviation for *this* column even though it is under the flat 0.05
    # (5-point) default the un-calibrated path would have ignored.
    baseline = _calibrated(0.02, 0.02, 0.02, 0.02, 0.02, 0.02)
    shifts = null_rate_deltas(
        {"email": 0.02}, {"email": 0.08}, baselines={"email": baseline}
    )
    assert len(shifts) == 1
    assert shifts[0].column == "email"


def test_a_volatile_columns_baseline_tolerates_a_move_larger_than_the_flat_threshold():
    # A column whose null rate has always swung between roughly 10% and
    # 30% (e.g. an optional field only some sources populate): a reading of
    # 30% is well past the flat 5-point threshold from a 15% starting
    # point, but it is exactly the swing this column has always shown.
    baseline = _calibrated(0.10, 0.30, 0.10, 0.30, 0.10, 0.30, 0.15)
    shifts = null_rate_deltas(
        {"discount_pct": 0.15}, {"discount_pct": 0.30}, baselines={"discount_pct": baseline}
    )
    assert shifts == []


def test_a_column_without_a_baseline_keeps_the_flat_threshold_fallback():
    baseline = _calibrated(0.02, 0.02)  # below MIN_SAMPLE_SIZE - not yet calibrated
    shifts = null_rate_deltas(
        {"email": 0.02, "country": 0.05},
        {"email": 0.08, "country": 0.055},
        baselines={"email": baseline},
    )
    # 'email': baseline exists but isn't calibrated yet -> flat threshold
    # applies, and a 6-point move clears the default 0.05.
    # 'country': no baseline entry at all -> flat threshold, 0.5-point move
    # does not clear it.
    assert [s.column for s in shifts] == ["email"]


def test_detect_drift_threads_null_rate_baselines_through():
    schema = {"id": "int64", "email": "string"}
    baseline = _calibrated(0.02, 0.02, 0.02, 0.02, 0.02, 0.02)
    result = detect_drift(
        schema,
        schema,
        {"id": 0.0, "email": 0.02},
        {"id": 0.0, "email": 0.08},
        null_rate_baselines={"email": baseline},
    )
    assert result is not None
    assert result.kind == "statistical_shift"
    assert result.null_rate_shifts[0].column == "email"


def test_detect_drift_with_no_baselines_argument_is_unchanged():
    # The exact scenario test_null_rate_deltas_ignores_columns_below_threshold
    # exercises directly - detect_drift's own call site must still see it.
    schema = {"id": "int64"}
    result = detect_drift(schema, schema, {"id": 0.10}, {"id": 0.12})
    assert result is None
