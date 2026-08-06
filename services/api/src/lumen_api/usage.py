"""What this workspace is spending, and against what.

The frontend already had a plan badge and a token count with nothing behind
them (ADR-0004's own motivating complaint). This is that backing: the plan
row and each metered metric's status from the same `QuotaGate` a run is
admitted or refused against, so the number a person sees is the number that
actually governs them.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends
from sqlalchemy import text

from lumen_api.auth.dependencies import Identity, current_identity
from lumen_api.billing.quota import QuotaGate
from lumen_api.db.session import user_session

router = APIRouter(prefix="/v1/usage", tags=["usage"])

_METRICS = ("llm_input_tokens", "llm_output_tokens", "agent_run")


@router.get("")
async def usage(identity: Annotated[Identity, Depends(current_identity)]) -> dict[str, Any]:
    async with user_session(identity.user_id) as db:
        plan = (
            await db.execute(
                text("select code, name, price_cents from public.plans where code = :code"),
                {"code": identity.plan_code},
            )
        ).mappings().first()

    gate = QuotaGate(identity.org_id, identity.user_id)
    statuses = [await gate.status(metric) for metric in _METRICS]

    return {
        "plan": dict(plan) if plan else {"code": identity.plan_code, "name": identity.plan_code, "price_cents": 0},
        "usage": [status.as_dict() for status in statuses],
    }
