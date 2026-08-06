"""Deterministic drift detection — ADR-0008.

No model call anywhere in this module, on purpose: a scheduled tick has to be
cheap enough to run against every source on every org's schedule regardless of
whether anything actually changed. The reasoning-tier model is reserved for
*diagnosing* a change this module already found, never for noticing one.

PSI / distribution-shift — the third check ADR-0008 describes — is
deliberately not here. `profile_source` captures a column's dtype and null
rate, not its value distribution, and population stability index needs a
distribution to compare against. Schema drift and null-rate drift are both
real today, both computable from what profiling already captures, and both
ship in this file; distribution-shift is the next thing to add once profiling
captures a histogram worth diffing — not before.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Literal

# A rename guess below this similarity is more often noise than signal —
# lower it and unrelated columns start pairing up; raise it and an obvious
# rename (a dropped trailing underscore, a case change) stops being caught.
_RENAME_SIMILARITY = 0.6


@dataclass(frozen=True)
class SchemaChange:
    kind: Literal["added", "removed", "renamed", "type_changed"]
    column: str
    detail: str
    previous_column: str | None = None


def diff_schema(old: dict[str, str], new: dict[str, str]) -> list[SchemaChange]:
    """What changed between two `frame_schema()` snapshots of the same source.

    Rename detection is a heuristic (name similarity only, not the column's
    position — position is not a fact a dict schema actually carries) and is
    reported the same way a genuine add+remove would be if it guesses wrong.
    That is deliberate: this function's output becomes a *proposal* a human
    reviews, not an automatic action, so a wrong guess costs a moment of
    review, not a bad decision made unattended.
    """
    old_cols, new_cols = set(old), set(new)
    added, removed, common = new_cols - old_cols, old_cols - new_cols, old_cols & new_cols

    changes: list[SchemaChange] = []
    matched_added: set[str] = set()
    matched_removed: set[str] = set()

    for removed_col in sorted(removed):
        best_match, best_score = None, 0.0
        for added_col in sorted(added - matched_added):
            score = SequenceMatcher(None, removed_col.lower(), added_col.lower()).ratio()
            if score > best_score:
                best_match, best_score = added_col, score
        if best_match is not None and best_score >= _RENAME_SIMILARITY:
            changes.append(
                SchemaChange(
                    kind="renamed",
                    column=best_match,
                    previous_column=removed_col,
                    detail=f"'{removed_col}' appears renamed to '{best_match}' ({best_score:.0%} similar)",
                )
            )
            matched_added.add(best_match)
            matched_removed.add(removed_col)

    for col in sorted(added - matched_added):
        changes.append(SchemaChange(kind="added", column=col, detail=f"new column '{col}' ({new[col]})"))
    for col in sorted(removed - matched_removed):
        changes.append(SchemaChange(kind="removed", column=col, detail=f"column '{col}' is gone"))
    for col in sorted(common):
        if old[col] != new[col]:
            changes.append(
                SchemaChange(
                    kind="type_changed",
                    column=col,
                    detail=f"'{col}' changed type from {old[col]} to {new[col]}",
                )
            )
    return changes


@dataclass(frozen=True)
class NullRateShift:
    column: str
    previous: float
    current: float
    delta: float


def null_rate_deltas(
    old: dict[str, float], new: dict[str, float], threshold: float = 0.05
) -> list[NullRateShift]:
    """Columns whose null rate moved by more than `threshold` (absolute),
    either direction. Restricted to columns present in both profiles — a
    column that appeared or disappeared is `diff_schema`'s finding, not this
    one's; a column cannot have "drifted null" if it did not exist before."""
    shifts = []
    for col in sorted(set(old) & set(new)):
        delta = new[col] - old[col]
        if abs(delta) >= threshold:
            shifts.append(NullRateShift(column=col, previous=old[col], current=new[col], delta=delta))
    return shifts


@dataclass(frozen=True)
class DriftResult:
    kind: Literal["schema_change", "statistical_shift"]
    severity: float  # 0..1
    schema_changes: list[SchemaChange] = field(default_factory=list)
    null_rate_shifts: list[NullRateShift] = field(default_factory=list)

    def as_details(self) -> dict:
        return {
            "schema_changes": [vars(change) for change in self.schema_changes],
            "null_rate_shifts": [vars(shift) for shift in self.null_rate_shifts],
        }


def detect_drift(
    old_schema: dict[str, str],
    new_schema: dict[str, str],
    old_null_rates: dict[str, float],
    new_null_rates: dict[str, float],
    *,
    null_rate_threshold: float = 0.05,
) -> DriftResult | None:
    """The one entry point a scheduled tick calls. `None` means the tick found
    nothing worth a `DriftEvent` row — the common case, and the reason this
    whole module has to be free: most ticks against a healthy source end here."""
    schema_changes = diff_schema(old_schema, new_schema)
    null_shifts = null_rate_deltas(old_null_rates, new_null_rates, null_rate_threshold)

    if not schema_changes and not null_shifts:
        return None

    # Schema drift is weighted as the more consequential class — a cleaning
    # step references a column by name, and a rename or removal can break one
    # outright, where a null-rate wobble usually just changes an outcome's
    # accuracy. Severity scales with how much changed, not a flat 1.0 for any
    # change: three columns disappearing should not read the same as one
    # column's null rate moving six points.
    schema_severity = min(1.0, len(schema_changes) * 0.4)
    stat_severity = min(1.0, max((abs(shift.delta) for shift in null_shifts), default=0.0) * 2)
    severity = max(schema_severity, stat_severity)
    kind = "schema_change" if schema_changes else "statistical_shift"

    return DriftResult(kind=kind, severity=severity, schema_changes=schema_changes, null_rate_shifts=null_shifts)
