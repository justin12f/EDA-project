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
    "select id, org_id, run_id, thread_id, author_agent, kind, status, spec, "
    "       rationale, estimate, decided_by, decided_at, applied_run_id, created_at "
    "from public.proposals"
)


class DecisionRequest(BaseModel):
    decision: Literal["accept", "reject"]


@router.get("")
async def list_proposals(
    identity: Annotated[Identity, Depends(current_identity)],
    thread_id: uuid.UUID | None = None,
    status: str | None = None,
) -> dict[str, Any]:
    clauses, params = [], {}
    if thread_id is not None:
        clauses.append("thread_id = :thread")
        params["thread"] = thread_id
    if status is not None:
        clauses.append("status = cast(:status as public.proposal_status)")
        params["status"] = status
    where = f" where {' and '.join(clauses)}" if clauses else ""

    async with user_session(identity.user_id) as db:
        rows = (
            await db.execute(text(f"{_SELECT}{where} order by created_at desc"), params)
        ).mappings().all()
    return {"proposals": [dict(row) for row in rows]}


@router.get("/{proposal_id}")
async def get_proposal(
    proposal_id: uuid.UUID,
    identity: Annotated[Identity, Depends(current_identity)],
) -> dict[str, Any]:
    async with user_session(identity.user_id) as db:
        row = (
            await db.execute(text(f"{_SELECT} where id = :id"), {"id": proposal_id})
        ).mappings().first()
    if row is None:
        raise NotFound(f"No proposal with id {proposal_id}")
    return dict(row)


@router.post("/{proposal_id}/decide")
async def decide_proposal(
    proposal_id: uuid.UUID,
    body: DecisionRequest,
    # A viewer can watch an agent work but not authorize what it changes.
    identity: Annotated[Identity, Depends(require_role("member"))],
) -> dict[str, Any]:
    async with user_session(identity.user_id) as db:
        row = (
            await db.execute(text(f"{_SELECT} where id = :id"), {"id": proposal_id})
        ).mappings().first()
    if row is None:
        raise NotFound(f"No proposal with id {proposal_id}")
    if row["status"] != "awaiting_review":
        raise Conflict(f"Proposal is already '{row['status']}', not awaiting review")
    if row["kind"] in _OWNER_ONLY_KINDS and identity.role != "owner":
        raise Forbidden(f"Only a workspace owner can decide a '{row['kind']}' proposal")

    if body.decision == "reject":
        async with user_session(identity.user_id) as db:
            await db.execute(
                text(
                    "update public.proposals set status = 'rejected', "
                    "       decided_by = :user, decided_at = now() where id = :id"
                ),
                {"user": identity.user_id, "id": proposal_id},
            )
        return {"id": str(proposal_id), "status": "rejected"}

    if row["kind"] == "plan_change":
        return await _apply_plan_change(proposal_id, row, identity)
    if row["kind"] == "member_role_change":
        return await _apply_member_role_change(proposal_id, row, identity)
    if row["kind"] == "entity_mapping":
        return await _apply_entity_mapping(proposal_id, row, identity)
    if row["kind"] not in _PIPELINE_KINDS:
        raise BadRequest(f"No apply handler for proposal kind '{row['kind']}'")
    return await _apply_cleaning_pipeline(proposal_id, row, identity)


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
                await db.execute(
                    text(
                        "insert into public.canonical_entity_members (entity_id, source_id, column_name) "
                        "values (:entity, :source, :column)"
                    ),
                    {
                        "entity": entity_id,
                        "source": uuid.UUID(member["source_id"]),
                        "column": member["column"],
                    },
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
