"""Whether a source can be trusted right now (ADR-0009 §3) — computed fresh
on every read, never stored, so there is only ever one place this can
disagree with itself. `certification_for` is shared by the authenticated
endpoint below and the public trust API (`GET /v1/public/certification`),
which is the entire point of a *computed* view: the same query answers the
same question for a signed-in member and for an external system holding an
API key, because it is derived from `DriftEvent`/`Proposal` state either way,
never a second ledger.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Annotated, Any

from fastapi import APIRouter, Depends
from sqlalchemy import text

from lumen_api.auth.dependencies import Identity, current_identity
from lumen_api.context.store import ContextStore
from lumen_api.db.session import user_session

router = APIRouter(prefix="/v1/sources/{source_id}", tags=["certification"])

# A canonical entity's member column with more than this fraction null has
# drifted enough to distrust, even if the shift never crossed detect_drift's
# own threshold in isolation — a column being load-bearing for a business
# term is a stricter bar than "nothing changed enough to alert on."
_ANOMALY_NULL_RATE = 0.5


@dataclass(frozen=True)
class Certification:
    certified: bool
    open_drift_count: int
    pending_pipeline_review: bool
    unresolved_entity_columns: list[str]
    last_checked_at: str | None
    checked_by: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "certified": self.certified,
            "open_drift_count": self.open_drift_count,
            "pending_pipeline_review": self.pending_pipeline_review,
            "unresolved_entity_columns": self.unresolved_entity_columns,
            "last_checked_at": self.last_checked_at,
            "checked_by": self.checked_by,
        }


async def certification_for(
    org_id: uuid.UUID, user_id: uuid.UUID, source_id: uuid.UUID
) -> Certification:
    async with user_session(user_id) as db:
        open_drift = (
            await db.execute(
                text(
                    "select count(*) from public.drift_events "
                    "where source_id = :source and status not in ('resolved', 'dismissed')"
                ),
                {"source": source_id},
            )
        ).scalar_one()

        # "the source's active pipeline spec is the currently accepted
        # Proposal" — a pipeline proposal still awaiting a decision means
        # what's currently running has not been through the review this
        # product's whole trust model rests on, whatever it currently is.
        pending = (
            await db.execute(
                text(
                    "select count(*) from public.proposals p "
                    "join public.runs r on r.id = p.run_id "
                    "where r.source_id = :source "
                    "  and p.kind in ('cleaning_pipeline', 'pipeline_patch') "
                    "  and p.status = 'awaiting_review'"
                ),
                {"source": source_id},
            )
        ).scalar_one()

        members = (
            await db.execute(
                text(
                    "select m.column_name from public.canonical_entity_members m "
                    "join public.canonical_entities e on e.id = m.entity_id "
                    "where m.source_id = :source and e.status = 'approved'"
                ),
                {"source": source_id},
            )
        ).scalars().all()

        schedule = (
            await db.execute(
                text("select last_run_at from public.data_source_schedules where source_id = :source"),
                {"source": source_id},
            )
        ).mappings().first()

    baseline = await ContextStore(org_id, user_id).latest_profile(source_id)
    schema = (baseline or {}).get("schema", {})
    null_rates = (baseline or {}).get("null_rates", {})
    unresolved = [
        column
        for column in members
        if column not in schema or null_rates.get(column, 0.0) > _ANOMALY_NULL_RATE
    ]

    last_checked_at: str | None = None
    checked_by: str | None = None
    if schedule and schedule["last_run_at"]:
        last_checked_at = schedule["last_run_at"].isoformat()
        checked_by = "scheduled_tick"
    elif baseline is not None:
        checked_by = "manual_profile"

    return Certification(
        # Never checked at all is not "nothing wrong" — it is "nothing
        # known" — so it can never read as certified regardless of the
        # other three conditions all trivially passing on no evidence.
        certified=checked_by is not None and open_drift == 0 and pending == 0 and not unresolved,
        open_drift_count=int(open_drift),
        pending_pipeline_review=bool(pending),
        unresolved_entity_columns=unresolved,
        last_checked_at=last_checked_at,
        checked_by=checked_by,
    )


@router.get("/certification")
async def get_certification(
    source_id: uuid.UUID, identity: Annotated[Identity, Depends(current_identity)]
) -> dict[str, Any]:
    result = await certification_for(identity.org_id, identity.user_id, source_id)
    return {"source_id": str(source_id), **result.as_dict()}
