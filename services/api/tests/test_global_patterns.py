"""drift_signature() and is_corroborated() (ADR-0012) — both pure and
database-free, same reason structural_shape() and wilson_lower_bound()
(ADR-0011) run in the default suite rather than gated behind `-m
integration`: a function whose answer depends only on its arguments is the
whole point of ADR-0012's own action item #2.
"""

from __future__ import annotations

import pytest

from lumen_api.global_patterns import (
    MIN_CORROBORATION_OUTCOMES,
    MIN_SUCCESS_RATE,
    GlobalPatternMatch,
    drift_signature,
    is_corroborated,
)


class TestDriftSignature:
    def test_a_single_kind_of_schema_change_is_named_by_its_kind(self):
        details = {"schema_changes": [{"kind": "added", "column": "x"}]}
        assert drift_signature("schema_change", details) == "schema_change:added"

    def test_two_added_columns_are_still_one_shape(self):
        details = {
            "schema_changes": [
                {"kind": "added", "column": "x"},
                {"kind": "added", "column": "y"},
            ]
        }
        assert drift_signature("schema_change", details) == "schema_change:added"

    def test_mixed_schema_change_kinds_are_mixed_and_sorted(self):
        details = {
            "schema_changes": [
                {"kind": "type_changed", "column": "x"},
                {"kind": "added", "column": "y"},
            ]
        }
        # sorted regardless of the order detect_drift happened to report —
        # two orgs whose schema changed the same *way* must fingerprint
        # identically even if diff_schema visited their columns differently.
        assert drift_signature("schema_change", details) == "schema_change:mixed:added+type_changed"
        reordered = {
            "schema_changes": [
                {"kind": "added", "column": "y"},
                {"kind": "type_changed", "column": "x"},
            ]
        }
        assert drift_signature("schema_change", reordered) == drift_signature("schema_change", details)

    def test_a_schema_change_with_no_schema_changes_listed_is_unclassified(self):
        assert drift_signature("schema_change", {}) == "schema_change:unclassified"

    def test_an_increase_only_null_rate_shift_is_named_increase(self):
        details = {"null_rate_shifts": [{"column": "x", "delta": 0.1}]}
        assert drift_signature("statistical_shift", details) == "statistical_shift:null_rate_increase"

    def test_a_decrease_only_null_rate_shift_is_named_decrease(self):
        details = {"null_rate_shifts": [{"column": "x", "delta": -0.1}]}
        assert drift_signature("statistical_shift", details) == "statistical_shift:null_rate_decrease"

    def test_shifts_in_both_directions_are_mixed(self):
        details = {
            "null_rate_shifts": [
                {"column": "x", "delta": 0.1},
                {"column": "y", "delta": -0.2},
            ]
        }
        assert drift_signature("statistical_shift", details) == "statistical_shift:null_rate_mixed"

    def test_a_statistical_shift_with_no_shifts_listed_is_unclassified(self):
        assert drift_signature("statistical_shift", {}) == "statistical_shift:unclassified"

    def test_an_unknown_kind_does_not_raise(self):
        assert drift_signature("some_future_kind", {}) == "some_future_kind:unclassified"

    def test_the_signature_never_contains_a_literal_column_name(self):
        # The whole point (ADR-0012 §1, extending ADR-0011 SS1 to the drift
        # side): comparable across orgs, never the literal content that
        # would make two orgs' patterns incomparable or leak a column name.
        details = {"schema_changes": [{"kind": "added", "column": "customer_ssn"}]}
        signature = drift_signature("schema_change", details)
        assert "customer_ssn" not in signature
        details = {"null_rate_shifts": [{"column": "email_address", "delta": 0.3}]}
        signature = drift_signature("statistical_shift", details)
        assert "email_address" not in signature


class TestIsCorroborated:
    def test_none_is_never_corroborated(self):
        assert is_corroborated(None) is False

    def test_below_the_outcome_floor_is_not_corroborated_even_at_perfect_success(self):
        match = GlobalPatternMatch(
            occurrences=100, applied_count=MIN_CORROBORATION_OUTCOMES - 1, rejected_count=0, success_rate=1.0
        )
        assert is_corroborated(match) is False

    def test_below_the_success_rate_floor_is_not_corroborated_even_with_enough_outcomes(self):
        match = GlobalPatternMatch(
            occurrences=100,
            applied_count=MIN_CORROBORATION_OUTCOMES,
            rejected_count=50,
            success_rate=MIN_SUCCESS_RATE - 0.01,
        )
        assert is_corroborated(match) is False

    def test_meeting_both_floors_exactly_is_corroborated(self):
        match = GlobalPatternMatch(
            occurrences=MIN_CORROBORATION_OUTCOMES,
            applied_count=MIN_CORROBORATION_OUTCOMES,
            rejected_count=0,
            success_rate=MIN_SUCCESS_RATE,
        )
        assert is_corroborated(match) is True

    def test_high_occurrences_alone_never_substitutes_for_decided_outcomes(self):
        # occurrences counts *proposals*, not outcomes — a pattern nobody
        # has decided on yet proves nothing about whether it works.
        match = GlobalPatternMatch(occurrences=10_000, applied_count=0, rejected_count=0, success_rate=0.0)
        assert is_corroborated(match) is False

    @pytest.mark.parametrize("applied,rejected", [(3, 0), (10, 2), (50, 5)])
    def test_a_variety_of_corroborated_shapes_all_pass(self, applied, rejected):
        total = applied + rejected
        if total < MIN_CORROBORATION_OUTCOMES:
            pytest.skip("not enough outcomes to be a meaningful corroboration case")
        match = GlobalPatternMatch(
            occurrences=total, applied_count=applied, rejected_count=rejected, success_rate=applied / total
        )
        assert is_corroborated(match) == (applied / total >= MIN_SUCCESS_RATE)
