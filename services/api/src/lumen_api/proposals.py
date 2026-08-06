"""Proposals: what the agent asked to change, and the person's answer.

A human accepting a proposal here still runs the pipeline synchronously, in
this request — a cleaning pipeline over the dataset sizes this product
handles today runs in well under the request timeout, and moving it behind a
queue is a matter of swapping this function's body for `enqueue_job(...)`
once a run is large enough that a person shouldn't wait on it, not a redesign
of the schema. `services/worker` exists now (ADR-0008), but its first job is
the Sentinel's scheduled tick, not this path — a worker existing does not by
itself mean every synchronous thing should move onto it.
"""

from __future__ import annotations

import json
import uuid
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from lumen_api.apply_pipeline import apply_cleaning_pipeline
from lumen_api.auth.dependencies import Identity, current_identity, require_role
from lumen_api.billing.stripe_client import checkout_url_for_plan_change
from lumen_api.db.session import user_session
from lumen_api.errors import BadRequest, Conflict, Forbidden, NotFound
from lumen_api.lineage import record_entity_dependency
from lumen_api.trust import TRUST_ELIGIBLE_ROLES, record_decision, structural_shape

# Kinds only a workspace owner may decide — spending money or changing who can
# do what is a stricter gate than "any member can review a cleaning step".
_OWNER_ONLY_KINDS = frozenset({"plan_change", "member_role_change"})

# cleaning_pipeline (a human or CleaningAgent proposed it from scratch) and
# pipeline_patch (the Sentinel proposed a revision — ADR-0008) apply the same
# way: both specs are {rid, steps}, and "how a pipeline gets run" does not
# depend on who or what decided it should change.
_PIPELINE_KINDS = frozenset({"cleaning_pipeline", "pipeline_patch"})

router = APIRouter(prefix="/v1/proposals", tags=["proposals"])

_SELECT = (
    "select p.id, p.org_id, p.run_id, p.thread_id, p.author_agent, p.kind, p.status, p.spec, "
    "       p.rationale, p.estimate, p.decided_by, p.decided_at, p.applied_run_id, p.created_at, "
    # Already attached, never fetched separately (ADR-0010 §4's Option D) —
    # null for every kind that never gets one (plan_change,
    # member_role_change, entity_mapping) or hasn't finished computing yet.
    "       ir.dependents_checked as impact_dependents_checked, "
    "       ir.dependents_total as impact_dependents_total, "
    "       ir.findings as impact_findings, ir.summary as impact_summary, "
    "       ir.computed_at as impact_computed_at "
    "from public.proposals p "
    "left join public.impact_reports ir on ir.proposal_id = p.id"
)


class DecisionRequest(BaseModel):
    decision: Literal["accept", "reject"]


def _shape(row: Any) -> dict[str, Any]:
    data = dict(row)
    computed_at = data.pop("impact_computed_at")
    impact = None
    if computed_at is not None:
        impact = {
            "dependents_checked": data.pop("impact_dependents_checked"),
            "dependents_total": data.pop("impact_dependents_total"),
            "findings": data.pop("impact_findings"),
            "summary": data.pop("impact_summary"),
        }
    else:
        for key in ("impact_dependents_checked", "impact_dependents_total", "impact_findings", "impact_summary"):
            data.pop(key, None)
    data["impact"] = impact
    return data


@router.get("")
async def list_proposals(
    identity: Annotated[Identity, Depends(current_identity)],
    thread_id: uuid.UUID | None = None,
    status: str | None = None,
) -> dict[str, Any]:
    clauses, params = [], {}
    if thread_id is not None:
        clauses.append("p.thread_id = :thread")
        params["thread"] = thread_id
    if status is not None:
        clauses.append("p.status = cast(:status as public.proposal_status)")
        params["status"] = status
    where = f" where {' and '.join(clauses)}" if clauses else ""

    async with user_session(identity.user_id) as db:
        rows = (
            await db.execute(text(f"{_SELECT}{where} order by p.created_at desc"), params)
        ).mappings().all()
    return {"proposals": [_shape(row) for row in rows]}


@router.get("/{proposal_id}")
async def get_proposal(
    proposal_id: uuid.UUID,
    identity: Annotated[Identity, Depends(current_identity)],
) -> dict[str, Any]:
    async with user_session(identity.user_id) as db:
        row = (
            await db.execute(text(f"{_SELECT} where p.id = :id"), {"id": proposal_id})
        ).mappings().first()
    if row is None:
        raise NotFound(f"No proposal with id {proposal_id}")
    return _shape(row)


@router.post("/{proposal_id}/decide")
async def decide_proposal(
    proposal_id: uuid.UUID,
    body: DecisionRequest,
    # A viewer can watch an agent work but not authorize what it changes.
    identity: Annotated[Identity, Depends(require_role("member"))],
) -> dict[str, Any]:
    async with user_session(identity.user_id) as db:
        row = (
            await db.execute(text(f"{_SELECT} where p.id = :id"), {"id": proposal_id})
        ).mappings().first()
    if row is None:
        raise NotFound(f"No proposal with id {proposal_id}")
    if row["status"] != "awaiting_review":
        raise Conflict(f"Proposal is already '{row['status']}', not awaiting review")
    if row["kind"] in _OWNER_ONLY_KINDS and identity.role != "owner":
        raise Forbidden(f"Only a workspace owner can decide a '{row['kind']}' proposal")

    pattern = structural_shape(row["kind"], dict(row["spec"] or {}))

    if body.decision == "reject":
        async with user_session(identity.user_id) as db:
            await db.execute(
                text(
                    "update public.proposals set status = 'rejected', "
                    "       decided_by = :user, decided_at = now() where id = :id"
                ),
                {"user": identity.user_id, "id": proposal_id},
            )
            if identity.role in TRUST_ELIGIBLE_ROLES:
                await record_decision(db, identity.org_id, pattern, approved=False)
        return {"id": str(proposal_id), "status": "rejected"}

    if row["kind"] == "plan_change":
        result = await _apply_plan_change(proposal_id, row, identity)
    elif row["kind"] == "member_role_change":
        result = await _apply_member_role_change(proposal_id, row, identity)
    elif row["kind"] == "entity_mapping":
        result = await _apply_entity_mapping(proposal_id, row, identity)
    elif row["kind"] in _PIPELINE_KINDS:
        result = await _apply_cleaning_pipeline(proposal_id, row, identity)
    else:
        raise BadRequest(f"No apply handler for proposal kind '{row['kind']}'")

    # Separate transaction from whichever _apply_* above just ran, not the
    # same one — ADR-0011 §2 asks for "the same transaction as the
    # decision," which would mean threading a shared session into four
    # handlers with otherwise no reason to share one. If this write fails
    # after a successful apply, the proposal is still correctly applied;
    # only this org's trust score misses one data point, which is a
    # low-severity, self-correcting gap, not a decision left half-made.
    if identity.role in TRUST_ELIGIBLE_ROLES:
        async with user_session(identity.user_id) as db:
            await record_decision(db, identity.org_id, pattern, approved=True)
    return result


async def _apply_cleaning_pipeline(
    proposal_id: uuid.UUID, row: Any, identity: Identity
) -> dict[str, Any]:
    spec = dict(row["spec"] or {})
    rid, steps = spec.get("rid"), spec.get("steps")
    if not rid or not steps:
        raise BadRequest("This proposal's spec is missing 'rid' or 'steps'")

    outcome = await apply_cleaning_pipeline(
        identity.org_id,
        identity.user_id,
        thread_id=row["thread_id"],
        rid=rid,
        steps=steps,
        rationale=row["rationale"],
        proposal_id=proposal_id,
    )

    async with user_session(identity.user_id) as db:
        await db.execute(
            text(
                "update public.proposals set status = 'applied', decided_by = :user, "
                "       decided_at = now(), applied_run_id = :run where id = :id"
            ),
            {"user": identity.user_id, "run": outcome["applied_run_id"], "id": proposal_id},
        )

    return {
        "id": str(proposal_id),
        "status": "applied",
        "applied_run_id": str(outcome["applied_run_id"]),
        "result": outcome["result"],
        "report": outcome["report"],
    }


async def _apply_plan_change(proposal_id: uuid.UUID, row: Any, identity: Identity) -> dict[str, Any]:
    spec = dict(row["spec"] or {})
    plan_code = spec.get("plan_code")
    if not plan_code:
        raise BadRequest("This proposal's spec is missing 'plan_code'")

    async with user_session(identity.user_id) as db:
        # Both copies, in the same transaction: organizations.plan_code is the
        # fast-read denormalized value current_identity() serves;
        # subscriptions is the billing-detail record. They must never disagree.
        await db.execute(
            text("update public.organizations set plan_code = :plan where id = :org"),
            {"plan": plan_code, "org": identity.org_id},
        )
        await db.execute(
            text(
                "update public.subscriptions set plan_code = :plan, updated_at = now() "
                "where org_id = :org"
            ),
            {"plan": plan_code, "org": identity.org_id},
        )
        await db.execute(
            text(
                "update public.proposals set status = 'applied', decided_by = :user, "
                "       decided_at = now() where id = :id"
            ),
            {"user": identity.user_id, "id": proposal_id},
        )

    checkout_url = checkout_url_for_plan_change(
        identity.org_id, plan_code, int(spec.get("price_cents") or 0)
    )
    return {
        "id": str(proposal_id),
        "status": "applied",
        "plan_code": plan_code,
        "checkout_url": checkout_url,
        "note": (
            None
            if checkout_url
            else "Stripe is not configured — the plan changed in the app; no payment was collected."
        ),
    }


async def _apply_member_role_change(
    proposal_id: uuid.UUID, row: Any, identity: Identity
) -> dict[str, Any]:
    spec = dict(row["spec"] or {})
    target_user_id, role = spec.get("user_id"), spec.get("role")
    if not target_user_id or not role:
        raise BadRequest("This proposal's spec is missing 'user_id' or 'role'")

    async with user_session(identity.user_id) as db:
        result = await db.execute(
            text(
                "update public.memberships set role = cast(:role as public.org_role) "
                "where org_id = :org and user_id = :target"
            ),
            {"role": role, "org": identity.org_id, "target": uuid.UUID(target_user_id)},
        )
        if result.rowcount == 0:
            raise NotFound("That member is no longer part of this workspace.")
        await db.execute(
            text(
                "update public.proposals set status = 'applied', decided_by = :user, "
                "       decided_at = now() where id = :id"
            ),
            {"user": identity.user_id, "id": proposal_id},
        )

    return {"id": str(proposal_id), "status": "applied", "user_id": target_user_id, "role": role}


async def _apply_entity_mapping(proposal_id: uuid.UUID, row: Any, identity: Identity) -> dict[str, Any]:
    """No pipeline runs here — accepting just writes the `CanonicalEntity` a
    person already reviewed. `status='approved'` directly: the enum's
    'proposed' value describes the state the `Proposal` itself already held,
    not a second pending state this row ever passes through (ADR-0009 §2).
    """
    spec = dict(row["spec"] or {})
    name = spec.get("canonical_name")
    entity_type = spec.get("canonical_type")
    members = spec.get("members") or []
    reconciliation_rule = spec.get("reconciliation_rule") or {}
    if not name or not entity_type or len(members) < 2:
        raise BadRequest(
            "This proposal's spec is missing 'canonical_name', 'canonical_type', "
            "or at least two 'members'"
        )

    try:
        async with user_session(identity.user_id) as db:
            entity_id = (
                await db.execute(
                    text(
                        "insert into public.canonical_entities "
                        "(org_id, name, entity_type, reconciliation_rule, status, proposal_id, created_by) "
                        "values (:org, :name, :type, cast(:rule as jsonb), 'approved', :proposal, :user) "
                        "returning id"
                    ),
                    {
                        "org": identity.org_id,
                        "name": name,
                        "type": entity_type,
                        "rule": json.dumps(reconciliation_rule),
                        "proposal": proposal_id,
                        "user": identity.user_id,
                    },
                )
            ).scalar_one()

            for member in members:
                source_uuid = uuid.UUID(member["source_id"])
                await db.execute(
                    text(
                        "insert into public.canonical_entity_members (entity_id, source_id, column_name) "
                        "values (:entity, :source, :column)"
                    ),
                    {"entity": entity_id, "source": source_uuid, "column": member["column"]},
                )
                await record_entity_dependency(
                    db, identity.org_id, entity_id, source_uuid, member["column"]
                )

            await db.execute(
                text(
                    "update public.proposals set status = 'applied', decided_by = :user, "
                    "       decided_at = now() where id = :id"
                ),
                {"user": identity.user_id, "id": proposal_id},
            )
    except IntegrityError as exc:
        # Either the name or a member column is already claimed by an entity
        # accepted from a different proposal since this one was created — a
        # real conflict for a person to resolve, not a 500.
        raise Conflict(
            "This entity's name or one of its member columns is already mapped "
            "to a different canonical entity."
        ) from exc

    return {
        "id": str(proposal_id),
        "status": "applied",
        "entity": {
            "id": str(entity_id),
            "name": name,
            "entity_type": entity_type,
            "members": members,
        },
    }
