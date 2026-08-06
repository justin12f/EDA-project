"""Collective immune memory — ADR-0012. A structural signature shared across
every org that opts in, never a sample of an org's actual data:
`global_patterns` has no `org_id` column at all, so there is no query,
correct or buggy, that can reconnect a shared pattern to the organization
that produced it. That is a property of the schema, not a promise made by
the code that writes to it.

Two signatures, both pure and database-free — the same requirement ADR-0011's
own action item put on `structural_shape()`, extended here to the drift side.
`drift_signature()` is the shape of the *problem* (what kind of drift this
is); `structural_shape()` (ADR-0011, reused as-is via proposals.py/sentinel.py
— nothing about it needed to change) is the shape of the *fix*. A
`GlobalPattern` row is keyed on both together, since "this fix works" only
means something for a specific problem shape.

This codebase has no "revert an applied proposal" feature. The ADR's own
prose talks about a fix being "rejected or reverted" — the only real negative
signal that exists here is a `pipeline_patch` proposal decided `reject`
(the same accept/reject `decide_proposal` already handles for ADR-0011).
`rejected_count` below tracks exactly that, not a separate undo action this
product doesn't have.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Annotated, Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from lumen_api.auth.dependencies import Identity, require_role
from lumen_api.db.session import service_session, user_session

# Neither number comes from the ADR's own text — §4 describes the property
# ("corroboration across a minimum number of distinct occurrences") without
# naming one. Five distinct orgs' decided outcomes, at least 60% of them a
# keep, is a deliberately modest bar: strict enough that one or two orgs'
# experience can never alone promote a pattern (each org can contribute at
# most one outcome per signature pair — see `_claim`), loose enough that a
# genuinely common fix surfaces well before the platform has hundreds of
# tenants on it.
MIN_CORROBORATION_OUTCOMES = 5
MIN_SUCCESS_RATE = 0.6


def drift_signature(kind: str, details: dict[str, Any]) -> str:
    """The structural shape of a `DriftEvent` (engine's `lumen.sentinel.drift`)
    — kind plus what changed, never a column name or a null-rate value. Mirrors
    `structural_shape()`'s bucket-then-sort approach: two orgs whose schema
    changed in the same *way* must fingerprint identically regardless of which
    columns or how many were involved.
    """
    if kind == "schema_change":
        kinds = sorted({c.get("kind", "") for c in (details.get("schema_changes") or []) if c})
        if not kinds:
            return "schema_change:unclassified"
        if len(kinds) == 1:
            return f"schema_change:{kinds[0]}"
        return "schema_change:mixed:" + "+".join(kinds)
    if kind == "statistical_shift":
        deltas = [s.get("delta", 0) for s in (details.get("null_rate_shifts") or []) if s]
        if not deltas:
            return "statistical_shift:unclassified"
        increased, decreased = any(d > 0 for d in deltas), any(d < 0 for d in deltas)
        if increased and decreased:
            return "statistical_shift:null_rate_mixed"
        return f"statistical_shift:null_rate_{'increase' if increased else 'decrease'}"
    return f"{kind}:unclassified"


@dataclass(frozen=True)
class GlobalPatternMatch:
    occurrences: int
    applied_count: int
    rejected_count: int
    success_rate: float


def is_corroborated(match: GlobalPatternMatch | None) -> bool:
    """The poisoning-resistance bar (§4): a pattern needs decided outcomes
    from enough distinct contributions, most of them a keep, before it is
    trusted as a confidence prior. `occurrences` alone (how many times a fix
    of this shape was *proposed*, anywhere) proves nothing about whether it
    *worked* — only decided outcomes do, so only those gate this.
    """
    if match is None:
        return False
    total = match.applied_count + match.rejected_count
    return total >= MIN_CORROBORATION_OUTCOMES and match.success_rate >= MIN_SUCCESS_RATE


async def lookup(pattern_signature: str, fix_signature: str) -> GlobalPatternMatch | None:
    """Consuming the shared library is free for every org regardless of
    whether it contributes (ADR-0012 §2). This reads `global_patterns`
    through `service_session()` because the table has no `org_id` column to
    scope a `user_session`'s RLS against in the first place — nothing routes
    a client directly to this table; this module is the only caller, and a
    match becomes a prior baked into the Sentinel's own confidence
    classification, never a raw row exposed through any endpoint.
    """
    async with service_session() as db:
        row = (
            await db.execute(
                text(
                    "select occurrences, applied_count, rejected_count, success_rate "
                    "from public.global_patterns "
                    "where pattern_signature = :pattern and fix_signature = :fix"
                ),
                {"pattern": pattern_signature, "fix": fix_signature},
            )
        ).mappings().first()
    if row is None:
        return None
    return GlobalPatternMatch(
        occurrences=row["occurrences"],
        applied_count=row["applied_count"],
        rejected_count=row["rejected_count"],
        success_rate=float(row["success_rate"]),
    )


async def contribution_enabled(db: AsyncSession, org_id: uuid.UUID) -> bool:
    row = (
        await db.execute(
            text("select pattern_contribution_enabled from public.organizations where id = :org"),
            {"org": org_id},
        )
    ).mappings().first()
    return bool(row and row["pattern_contribution_enabled"])


async def record_occurrence(
    db: AsyncSession, org_id: uuid.UUID, pattern_signature: str, fix_signature: str
) -> None:
    """A fix of this shape was proposed for a problem of this shape. Counted
    at most once per (org, pattern, fix) ever, via the `_claim` ledger — an
    org whose Sentinel re-encounters the same drift shape a hundred times
    contributes one occurrence to the shared count, not a hundred.

    `db` is expected to be an open `user_session(user_id)` for `org_id` —
    the caller already has one (this is called from inside the same
    transaction that just wrote the proposal); only the aggregate write
    below needs a different session, since `global_patterns` has no RLS
    policy a user_session could ever satisfy.
    """
    if not await contribution_enabled(db, org_id):
        return
    if not await _claim(db, org_id, pattern_signature, fix_signature, column="contributed_occurrence"):
        return
    async with service_session() as global_db:
        await global_db.execute(
            text(
                "insert into public.global_patterns "
                "(pattern_signature, fix_signature, occurrences, last_seen_at) "
                "values (:pattern, :fix, 1, now()) "
                "on conflict (pattern_signature, fix_signature) do update set "
                "  occurrences = public.global_patterns.occurrences + 1, last_seen_at = now()"
            ),
            {"pattern": pattern_signature, "fix": fix_signature},
        )


async def record_outcome(
    db: AsyncSession,
    org_id: uuid.UUID,
    pattern_signature: str,
    fix_signature: str,
    *,
    applied: bool,
) -> None:
    """A previously proposed fix of this shape was accepted or rejected.
    Same one-contribution-per-org ledger claim as `record_occurrence`, on the
    independent `contributed_outcome` flag — an org's occurrence and outcome
    for the same signature pair are claimed separately, since a proposal is
    always created before it is decided, sometimes much later.
    """
    if not await contribution_enabled(db, org_id):
        return
    if not await _claim(db, org_id, pattern_signature, fix_signature, column="contributed_outcome"):
        return
    async with service_session() as global_db:
        row = (
            await global_db.execute(
                text(
                    "insert into public.global_patterns "
                    "(pattern_signature, fix_signature, applied_count, rejected_count, last_seen_at) "
                    "values (:pattern, :fix, :applied_inc, :rejected_inc, now()) "
                    "on conflict (pattern_signature, fix_signature) do update set "
                    "  applied_count = public.global_patterns.applied_count + :applied_inc, "
                    "  rejected_count = public.global_patterns.rejected_count + :rejected_inc, "
                    "  last_seen_at = now() "
                    "returning applied_count, rejected_count"
                ),
                {
                    "pattern": pattern_signature,
                    "fix": fix_signature,
                    "applied_inc": 1 if applied else 0,
                    "rejected_inc": 0 if applied else 1,
                },
            )
        ).mappings().one()
        total = row["applied_count"] + row["rejected_count"]
        await global_db.execute(
            text(
                "update public.global_patterns set success_rate = :rate "
                "where pattern_signature = :pattern and fix_signature = :fix"
            ),
            {
                "rate": row["applied_count"] / total if total else 0.0,
                "pattern": pattern_signature,
                "fix": fix_signature,
            },
        )


async def _claim(
    db: AsyncSession, org_id: uuid.UUID, pattern_signature: str, fix_signature: str, *, column: str
) -> bool:
    """Atomically mark this org's first-ever contribution of `column` for this
    signature pair, or discover it already made one. A single upsert, so two
    concurrent calls for the same org (a schedule double-firing) cannot both
    win the claim — Postgres serializes the conflicting writes.

    `column` is always one of this module's own two literal strings, never
    request input, so interpolating it into the query text is safe (`text()`
    has no placeholder syntax for a column name).

    If the caller's `global_patterns` write fails after this claim commits,
    this org can never contribute this signature pair again — a narrow,
    self-limiting gap (this org's one data point is lost, not duplicated or
    corrupted), the same shape of trade-off proposals.py already accepts for
    ADR-0011's trust write landing in a separate transaction from the apply.
    """
    row = (
        await db.execute(
            text(
                f"insert into public.org_pattern_contributions "
                f"(org_id, pattern_signature, fix_signature, {column}) "
                f"values (:org, :pattern, :fix, true) "
                f"on conflict (org_id, pattern_signature, fix_signature) do update "
                f"  set {column} = true "
                f"  where public.org_pattern_contributions.{column} = false "
                f"returning 1"
            ),
            {"org": org_id, "pattern": pattern_signature, "fix": fix_signature},
        )
    ).first()
    return row is not None


# ── the opt-in setting (ADR-0012 §2, §5) ─────────────────────────────────
#
# "Visible next to the trust panel ADR-0011 already introduced, not buried
# in account settings." Owner-only, same bar as trust.py's panel: this
# decides whether this org's own decision history feeds a platform-wide
# aggregate, which is at least as sensitive as trust.py's auto-apply toggle.
# Consuming the shared library needs no setting at all — every org already
# gets that, opted in or not.

router = APIRouter(prefix="/v1/pattern-contribution", tags=["global-patterns"])


class ContributionUpdate(BaseModel):
    enabled: bool


@router.get("")
async def get_contribution_setting(
    identity: Annotated[Identity, Depends(require_role("owner"))],
) -> dict[str, Any]:
    async with user_session(identity.user_id) as db:
        enabled = await contribution_enabled(db, identity.org_id)
    return {"enabled": enabled}


@router.put("")
async def set_contribution_setting(
    body: ContributionUpdate,
    identity: Annotated[Identity, Depends(require_role("owner"))],
) -> dict[str, Any]:
    async with user_session(identity.user_id) as db:
        await db.execute(
            text("update public.organizations set pattern_contribution_enabled = :enabled where id = :org"),
            {"enabled": body.enabled, "org": identity.org_id},
        )
    return {"enabled": body.enabled}
