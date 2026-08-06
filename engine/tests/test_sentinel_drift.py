"""Deterministic drift detection (ADR-0008) — every case here must resolve
with no model call, because that is the whole point of the deterministic gate.
"""

from __future__ import annotations

import pytest

from lumen.sentinel import detect_drift, diff_schema, null_rate_deltas


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
