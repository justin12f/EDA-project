"""Per-source schedule toggle and drift feed — ADR-0008's own account of what
the frontend needs: "the only new surface is a feed of `drift_events` and a
schedule toggle per source." Detection and diagnosis already run entirely off
the request path (`services/worker`); this router only lets a person turn a
source's scheduled scan on, opt into auto-apply, and see what the Sentinel
has found.
"""

from __future__ import annotations

import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import text

from lumen_api.auth.dependencies import Identity, current_identity, require_role
from lumen_api.db.session import user_session

router = APIRouter(prefix="/v1/sources/{source_id}", tags=["sentinel"])

_SCHEDULE_COLUMNS = "id, cron, enabled, auto_apply_enabled, last_run_at, next_run_at"


class ScheduleUpdate(BaseModel):
    enabled: bool
    auto_apply_enabled: bool = False


@router.get("/schedule")
async def get_schedule(
    source_id: uuid.UUID, identity: Annotated[Identity, Depends(current_identity)]
) -> dict[str, Any]:
    async with user_session(identity.user_id) as db:
        row = (
            await db.execute(
                text(
                    f"select {_SCHEDULE_COLUMNS} from public.data_source_schedules "
                    "where source_id = :source"
                ),
                {"source": source_id},
            )
        ).mappings().first()
    if row is None:
        # No schedule row has ever been created for this source — off, same
        # shape as an explicit disable, so the frontend renders one way either
        # way instead of a separate "not yet configured" state.
        return {
            "id": None,
            "source_id": str(source_id),
            "enabled": False,
            "auto_apply_enabled": False,
            "cron": None,
            "last_run_at": None,
            "next_run_at": None,
        }
    return {"source_id": str(source_id), **dict(row)}


@router.put("/schedule")
async def set_schedule(
    source_id: uuid.UUID,
    body: ScheduleUpdate,
    # Opting into unattended writes (auto-apply) is a step above "any member
    # can review a proposal" — the same bar decide_proposal already holds
    # cleaning_pipeline and pipeline_patch to, just moved earlier: before a
    # patch exists, rather than at the moment of accepting one.
    identity: Annotated[Identity, Depends(require_role("member"))],
) -> dict[str, Any]:
    async with user_session(identity.user_id) as db:
        row = (
            await db.execute(
                text(
                    "insert into public.data_source_schedules "
                    "(org_id, source_id, enabled, auto_apply_enabled, created_by) "
                    "values (:org, :source, :enabled, :auto, :user) "
                    "on conflict (source_id) do update set "
                    "  enabled = excluded.enabled, "
                    "  auto_apply_enabled = excluded.auto_apply_enabled "
                    f"returning {_SCHEDULE_COLUMNS}"
                ),
                {
                    "org": identity.org_id,
                    "source": source_id,
                    "enabled": body.enabled,
                    "auto": body.auto_apply_enabled,
                    "user": identity.user_id,
                },
            )
        ).mappings().first()
    return {"source_id": str(source_id), **dict(row)}


@router.get("/drift-events")
async def list_drift_events(
    source_id: uuid.UUID,
    identity: Annotated[Identity, Depends(current_identity)],
    limit: int = 20,
) -> dict[str, Any]:
    async with user_session(identity.user_id) as db:
        rows = (
            await db.execute(
                text(
                    "select id, kind, severity, details, status, proposal_id, occurred_at "
                    "from public.drift_events where source_id = :source "
                    "order by occurred_at desc limit :limit"
                ),
                {"source": source_id, "limit": limit},
            )
        ).mappings().all()
    return {"events": [dict(row) for row in rows]}
