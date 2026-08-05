"""Proposals: what the agent asked to change, and the person's answer.

Accepting runs the pipeline synchronously, in this request. There is no arq
worker yet — `services/worker` is still empty — so this is the honest
implementation for what exists today rather than a stub waiting for one: a
cleaning pipeline over the dataset sizes this product handles runs in well
under the request timeout. Moving it behind a queue is a matter of swapping
this function's body for `enqueue_job(...)` once a run is large enough that a
person shouldn't wait on it — not a redesign of the schema, which already has
`runs.status` and `agent_events` built for exactly that.
"""

from __future__ import annotations

import uuid
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import text

from lumen.data_cleaning.data_cleaning_pipeline import PipelineBuilder
from lumen_api.auth.dependencies import Identity, current_identity, require_role
from lumen_api.context.store import ContextEntry, ContextStore, Kind
from lumen_api.datasets.store import HandleStore
from lumen_api.db.session import user_session
from lumen_api.errors import BadRequest, Conflict, NotFound
from lumen_api.jsonable import jsonable

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

    spec = dict(row["spec"] or {})
    rid, steps = spec.get("rid"), spec.get("steps")
    if not rid or not steps:
        raise BadRequest("This proposal's spec is missing 'rid' or 'steps'")

    store = HandleStore(identity.org_id, identity.user_id)
    handle = await store.get(rid)
    frame = await store.resolve(rid)

    pipeline = PipelineBuilder(frame).build(steps)
    cleaned = pipeline.run(frame)
    new_handle = await store.put(
        cleaned, backend=handle.backend, label=f"cleaned: {handle.label or rid}"
    )

    async with user_session(identity.user_id) as db:
        new_run_id = (
            await db.execute(
                text(
                    "insert into public.runs (org_id, thread_id, kind, status, backend, "
                    "                         created_by, finished_at) "
                    "values (:org, :thread, 'apply_pipeline', 'succeeded', :backend, "
                    "        :user, now()) returning id"
                ),
                {
                    "org": identity.org_id,
                    "thread": row["thread_id"],
                    "backend": handle.backend,
                    "user": identity.user_id,
                },
            )
        ).scalar_one()

        await db.execute(
            text(
                "update public.proposals set status = 'applied', decided_by = :user, "
                "       decided_at = now(), applied_run_id = :run where id = :id"
            ),
            {"user": identity.user_id, "run": new_run_id, "id": proposal_id},
        )

    summary = _report_summary(pipeline.report)
    try:
        await ContextStore(identity.org_id, identity.user_id).remember(
            ContextEntry(
                kind=Kind.RATIONALE,
                title=f"Cleaning applied to {handle.label or rid}",
                content=f"{row['rationale']}\n\n{summary}",
                rid=new_handle.rid,
                run_id=new_run_id,
                metadata={"steps": jsonable(pipeline.report.steps)},
            )
        )
    except Exception:  # noqa: BLE001 — the pipeline already ran; memory is best-effort
        pass

    return {
        "id": str(proposal_id),
        "status": "applied",
        "applied_run_id": str(new_run_id),
        "result": {
            "rid": new_handle.rid,
            "row_count": new_handle.row_count,
            "schema": new_handle.schema,
        },
        "report": summary,
    }


def _report_summary(report: Any) -> str:
    lines: list[str] = []
    for step in report.steps:
        metrics = step.get("metrics", {})
        removed = metrics.get("rows_removed", 0)
        changed = {c: r for c, r in metrics.get("change_ratio", {}).items() if r > 0}
        parts = [f"{removed} rows removed"] if removed else []
        if changed:
            parts.append(
                "changed " + ", ".join(f"{c} {r * 100:.1f}%" for c, r in sorted(changed.items()))
            )
        lines.append(f"- {step['name']}: {'; '.join(parts) or 'no measurable change'}")
    return "\n".join(lines) if lines else "No steps recorded."
