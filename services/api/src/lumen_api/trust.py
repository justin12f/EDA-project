"""Progressive trust: confidence learned per (org, pattern) from accept/reject
history (ADR-0011) — a second, independent gate stacked on top of ADR-0008's
structural confidence ladder, never a replacement for it.

`structural_shape()` is pure and database-free on purpose (ADR-0011's own
action item #2 calls for it to be unit-tested — a function whose answer
depends only on its arguments is what makes that possible without a live
database). Everything else here reads/writes `pattern_trust_scores` and
needs a session.
"""

from __future__ import annotations

import math
import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from lumen_api.auth.dependencies import Identity, require_role
from lumen_api.db.session import user_session
from lumen_api.errors import NotFound

# Below this many *consecutive* approvals (since the last rejection, if any
# — see the migration's own note on why this is a streak, not a lifetime
# count), a pattern has not earned unattended trust regardless of how good
# its Wilson score looks. The ADR's own illustrative default.
DEFAULT_MIN_STREAK = 20

# Which roles' decisions build trust. Not every accept should count the same
# (ADR-0011 §"Harder": "a proposal rubber-stamped by someone without the
# authority to make that call shouldn't build trust the org owner would
# endorse") — a `member` can still accept or reject normally, unchanged;
# their decision just does not feed the score. Matches the bar `_OWNER_ONLY_
# KINDS` already sets for money/permission-changing proposals in
# proposals.py, one step down: trust-building is admin-or-above, not
# owner-only, because unlike a plan change it never mutates anything by
# itself.
TRUST_ELIGIBLE_ROLES = frozenset({"admin", "owner"})

_STEP_BUCKETS: dict[str, str] = {
    "impute_categorical": "imputation",
    "handle_sentinel_values": "imputation",
    "remove_duplicates_rows": "deduplication",
    "fix_columns_types": "type_coercion",
    "fix_numeric_columns": "type_coercion",
    "fix_dates_columns": "type_coercion",
    "fix_not_numeric_columns": "type_coercion",
    "fix_bools_columns": "type_coercion",
    "safe_conversion": "type_coercion",
    "enforce_schema": "type_coercion",
    "cap_outliers": "outlier_handling",
    "handle_outliers": "outlier_handling",
    "iqr_outlier": "outlier_handling",
    "zscore_outlier": "outlier_handling",
    "standard_scaler": "outlier_handling",
    "drop_constant_columns": "column_pruning",
    "drop_high_missing_columns": "column_pruning",
    "normalize_categories": "normalization",
    "text_standardization": "normalization",
    "fix_columns_titles": "normalization",
    "cross_column_validation": "validation",
    "validate_domain_rules": "validation",
    "flag_data_quality": "validation",
    "add_audit_columns": "audit",
}


def structural_shape(kind: str, spec: dict[str, Any]) -> str:
    """A signature comparable across sources within an org — never a column
    name or a value (ADR-0011 §1), only "what kind of change is this."
    """
    if kind in ("cleaning_pipeline", "pipeline_patch"):
        return f"{kind}:{_pipeline_shape(spec.get('steps') or [])}"
    if kind == "entity_mapping":
        members = spec.get("members") or []
        return f"{kind}:{len(members)}_source_merge"
    if kind == "plan_change":
        return "plan_change:tier_change"
    if kind == "member_role_change":
        return "member_role_change:role_change"
    return f"{kind}:unclassified"


def _pipeline_shape(steps: list[dict[str, Any]]) -> str:
    if not steps:
        return "empty"
    buckets = sorted(
        {_STEP_BUCKETS.get(next(iter(step), ""), "other") for step in steps if step}
    )
    if len(buckets) == 1:
        return buckets[0] if len(steps) == 1 else f"{buckets[0]}_multi"
    return "mixed:" + "+".join(buckets)


def wilson_lower_bound(approvals: int, rejections: int, z: float = 1.96) -> float:
    """Conservative estimate of "how often has this org approved this
    pattern" — deliberately not the raw ratio (ADR-0011 §2 rejects that:
    two approvals out of two must not read as "100% trusted"). Standard
    Wilson score interval lower bound: near 0 at low n, converges toward
    the raw ratio as n grows. z=1.96 is the 95% confidence level.
    """
    n = approvals + rejections
    if n == 0:
        return 0.0
    p_hat = approvals / n
    z2 = z * z
    denominator = 1 + z2 / n
    center = p_hat + z2 / (2 * n)
    adjustment = z * math.sqrt((p_hat * (1 - p_hat) + z2 / (4 * n)) / n)
    return max(0.0, (center - adjustment) / denominator)


async def record_decision(
    db: AsyncSession, org_id: uuid.UUID, pattern_signature: str, *, approved: bool
) -> None:
    """Update (org, pattern)'s trust after a decision. `approvals` /
    `rejections` are lifetime and drive `score`, the descriptive Wilson
    estimate an owner reviews (ADR-0011 §5); `consecutive_approvals` resets
    to 0 on any rejection and is what the auto-apply gate actually reads
    (§4) — see the migration's own note on why both exist.
    """
    row = (
        await db.execute(
            text(
                "select approvals, rejections from public.pattern_trust_scores "
                "where org_id = :org and pattern_signature = :pattern"
            ),
            {"org": org_id, "pattern": pattern_signature},
        )
    ).mappings().first()
    approvals = (row["approvals"] if row else 0) + (1 if approved else 0)
    rejections = (row["rejections"] if row else 0) + (0 if approved else 1)
    score = wilson_lower_bound(approvals, rejections)

    await db.execute(
        text(
            "insert into public.pattern_trust_scores "
            "(org_id, pattern_signature, approvals, rejections, consecutive_approvals, "
            " score, last_decided_at) "
            "values (:org, :pattern, :approvals, :rejections, "
            "        case when :approved then 1 else 0 end, :score, now()) "
            "on conflict (org_id, pattern_signature) do update set "
            "  approvals = :approvals, rejections = :rejections, "
            "  consecutive_approvals = case when :approved "
            "    then public.pattern_trust_scores.consecutive_approvals + 1 else 0 end, "
            "  score = :score, last_decided_at = now()"
        ),
        {
            "org": org_id,
            "pattern": pattern_signature,
            "approvals": approvals,
            "rejections": rejections,
            "approved": approved,
            "score": score,
        },
    )


async def is_auto_apply_eligible(
    db: AsyncSession,
    org_id: uuid.UUID,
    pattern_signature: str,
    *,
    min_streak: int = DEFAULT_MIN_STREAK,
) -> bool:
    """The second gate ADR-0011 §3 adds: structural confidence (ADR-0008,
    checked by the caller) is necessary but no longer sufficient on its own.
    A pattern never decided on by an eligible role has no row at all —
    absence reads as "no earned trust yet," not as an error.
    """
    row = (
        await db.execute(
            text(
                "select consecutive_approvals, auto_apply_enabled "
                "from public.pattern_trust_scores where org_id = :org and pattern_signature = :pattern"
            ),
            {"org": org_id, "pattern": pattern_signature},
        )
    ).mappings().first()
    if row is None:
        return False
    return bool(row["auto_apply_enabled"]) and row["consecutive_approvals"] >= min_streak


# ── the trust panel (ADR-0011 §5) ────────────────────────────────────────
#
# "Trust here is not a black box the org discovers by watching what
# happens — it is a setting they can inspect and override." Owner-only,
# same bar as api_keys.py: this doesn't move money or spend quota by
# itself, but it does decide whether an agent is allowed to change tenant
# data unattended, which is at least as sensitive.

router = APIRouter(prefix="/v1/trust-patterns", tags=["trust"])


class AutoApplyUpdate(BaseModel):
    enabled: bool


@router.get("")
async def list_patterns(
    identity: Annotated[Identity, Depends(require_role("owner"))],
) -> dict[str, Any]:
    async with user_session(identity.user_id) as db:
        rows = (
            await db.execute(
                text(
                    "select id, pattern_signature, approvals, rejections, consecutive_approvals, "
                    "       score, auto_apply_enabled, last_decided_at "
                    "from public.pattern_trust_scores order by last_decided_at desc"
                )
            )
        ).mappings().all()
    return {
        "patterns": [
            {
                **dict(row),
                "id": str(row["id"]),
                # What the auto-apply gate would actually decide right now —
                # the raw counts alone leave a person doing the >= 20 math
                # themselves, which is exactly the "black box" §5 rejects.
                "eligible": bool(row["auto_apply_enabled"])
                and row["consecutive_approvals"] >= DEFAULT_MIN_STREAK,
            }
            for row in rows
        ],
        "min_streak": DEFAULT_MIN_STREAK,
    }


@router.put("/{pattern_id}/auto-apply")
async def set_auto_apply(
    pattern_id: uuid.UUID,
    body: AutoApplyUpdate,
    identity: Annotated[Identity, Depends(require_role("owner"))],
) -> dict[str, Any]:
    async with user_session(identity.user_id) as db:
        result = await db.execute(
            text("update public.pattern_trust_scores set auto_apply_enabled = :enabled where id = :id"),
            {"enabled": body.enabled, "id": pattern_id},
        )
    if result.rowcount == 0:
        raise NotFound("No trust pattern with that id")
    return {"id": str(pattern_id), "auto_apply_enabled": body.enabled}
