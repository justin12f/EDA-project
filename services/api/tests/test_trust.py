"""structural_shape() and wilson_lower_bound() (ADR-0011) — both pure and
database-free, so unlike almost everything else added this session, these
run in the default suite, not gated behind `-m integration`. That is the
entire point of keeping them pure (ADR-0011's own action item #2).
"""

from __future__ import annotations

import pytest

from lumen_api.trust import structural_shape, wilson_lower_bound


class TestStructuralShape:
    def test_a_single_step_pipeline_is_named_by_its_bucket(self):
        spec = {"steps": [{"impute_categorical": {"columns": ["x"]}}]}
        assert structural_shape("pipeline_patch", spec) == "pipeline_patch:imputation"

    def test_two_steps_in_the_same_bucket_are_distinguished_from_one(self):
        spec = {
            "steps": [
                {"impute_categorical": {"columns": ["x"]}},
                {"handle_sentinel_values": {"columns": ["y"]}},
            ]
        }
        assert structural_shape("pipeline_patch", spec) == "pipeline_patch:imputation_multi"

    def test_steps_spanning_different_buckets_are_mixed_and_sorted(self):
        spec = {"steps": [{"remove_duplicates_rows": {}}, {"impute_categorical": {}}]}
        # sorted alphabetically regardless of the order steps were given in —
        # two proposals with the same two step *types* in a different order
        # must fingerprint identically.
        assert structural_shape("pipeline_patch", spec) == "pipeline_patch:mixed:deduplication+imputation"
        reordered = {"steps": [{"impute_categorical": {}}, {"remove_duplicates_rows": {}}]}
        assert structural_shape("pipeline_patch", reordered) == structural_shape("pipeline_patch", spec)

    def test_an_unregistered_step_name_falls_back_to_other_instead_of_raising(self):
        spec = {"steps": [{"some_future_step_kind": {}}]}
        assert structural_shape("cleaning_pipeline", spec) == "cleaning_pipeline:other"

    def test_an_empty_step_list_is_its_own_shape(self):
        assert structural_shape("cleaning_pipeline", {"steps": []}) == "cleaning_pipeline:empty"

    def test_cleaning_pipeline_and_pipeline_patch_are_distinct_kinds_of_the_same_shape(self):
        spec = {"steps": [{"impute_categorical": {}}]}
        assert structural_shape("cleaning_pipeline", spec) != structural_shape("pipeline_patch", spec)

    def test_entity_mapping_is_classified_by_member_count_not_which_sources(self):
        two = {"members": [{"source_id": "a", "column": "x"}, {"source_id": "b", "column": "y"}]}
        three = {"members": [{"source_id": "a", "column": "x"}] * 3}
        assert structural_shape("entity_mapping", two) == "entity_mapping:2_source_merge"
        assert structural_shape("entity_mapping", three) == "entity_mapping:3_source_merge"

    def test_the_signature_never_contains_a_literal_column_name(self):
        # The whole point (ADR-0011 SS1): comparable across sources, never
        # the literal content that would make two orgs' patterns
        # incomparable or leak a column name into a cross-org-shaped string.
        spec = {"steps": [{"impute_categorical": {"columns": ["ssn", "email_address"]}}]}
        signature = structural_shape("pipeline_patch", spec)
        assert "ssn" not in signature
        assert "email_address" not in signature

    def test_plan_change_and_member_role_change_get_fixed_signatures(self):
        assert structural_shape("plan_change", {"plan_code": "pro"}) == "plan_change:tier_change"
        assert (
            structural_shape("member_role_change", {"role": "admin"})
            == "member_role_change:role_change"
        )

    def test_an_unknown_kind_does_not_raise(self):
        assert structural_shape("some_future_kind", {}) == "some_future_kind:unclassified"


class TestWilsonLowerBound:
    def test_no_decisions_is_zero(self):
        assert wilson_lower_bound(0, 0) == 0.0

    def test_matches_the_known_reference_value_at_n_equals_one(self):
        # Commonly-cited reference: Wilson lower bound for 1/1 approvals at
        # z=1.96 is ~0.206 — the number that makes "two approvals in a row
        # reads as 100% trusted" (Option B, rejected by the ADR) impossible.
        assert wilson_lower_bound(1, 0) == pytest.approx(0.2065, abs=1e-3)

    def test_a_short_perfect_streak_scores_below_a_long_one(self):
        short = wilson_lower_bound(2, 0)
        long_streak = wilson_lower_bound(200, 0)
        assert 0 < short < long_streak < 1

    def test_one_rejection_measurably_lowers_the_score_even_after_a_long_streak(self):
        clean = wilson_lower_bound(20, 0)
        one_miss = wilson_lower_bound(19, 1)
        assert one_miss < clean

    def test_the_score_never_exceeds_the_raw_ratio(self):
        # The lower bound is, definitionally, never optimistic relative to
        # the naive approvals/(approvals+rejections) ratio Option B used.
        for approvals, rejections in ((3, 0), (10, 2), (50, 5)):
            ratio = approvals / (approvals + rejections)
            assert wilson_lower_bound(approvals, rejections) <= ratio

    def test_the_score_is_bounded_between_zero_and_one(self):
        for approvals, rejections in ((0, 5), (5, 0), (1, 1), (1000, 1)):
            score = wilson_lower_bound(approvals, rejections)
            assert 0.0 <= score <= 1.0


# ── ADR-0024 schema kinds ───────────────────────────────────────────────


class TestSchemaShapes:
    def test_a_single_table_design_is_named_by_its_count(self):
        spec = {"tables": [{"name": "orders"}]}
        assert structural_shape("schema_design", spec) == "schema_design:1_table"

    def test_several_tables_pluralise(self):
        spec = {"tables": [{"name": "orders"}, {"name": "customers"}]}
        assert structural_shape("schema_design", spec) == "schema_design:2_tables"

    def test_a_design_with_no_tables_is_its_own_shape(self):
        assert structural_shape("schema_design", {"tables": []}) == "schema_design:empty"

    def test_an_additive_migration(self):
        spec = {"steps": [{"kind": "add_column", "reversible": True}]}
        assert structural_shape("schema_migration", spec) == "schema_migration:additive"

    def test_a_widening_migration(self):
        spec = {"steps": [{"kind": "widen_type", "reversible": True}]}
        assert structural_shape("schema_migration", spec) == "schema_migration:type_widening"

    def test_any_irreversible_step_makes_the_whole_migration_destructive(self):
        """One narrowing step among ten additive ones is still destructive.
        Trust is granted per shape, so a shape that can hide an irreversible
        step would let one be auto-applied on the strength of the others."""
        spec = {
            "steps": [
                {"kind": "add_column", "reversible": True},
                {"kind": "narrow_type", "reversible": False},
            ]
        }
        assert structural_shape("schema_migration", spec) == "schema_migration:destructive"

    def test_a_mixed_reversible_migration_reports_the_widening(self):
        spec = {
            "steps": [
                {"kind": "add_column", "reversible": True},
                {"kind": "widen_type", "reversible": True},
            ]
        }
        assert structural_shape("schema_migration", spec) == "schema_migration:type_widening"

    def test_the_shape_never_contains_a_table_or_column_name(self):
        spec = {"tables": [{"name": "customer_ssns"}, {"name": "salaries"}]}
        shape = structural_shape("schema_design", spec)
        assert "customer_ssns" not in shape
        assert "salaries" not in shape
