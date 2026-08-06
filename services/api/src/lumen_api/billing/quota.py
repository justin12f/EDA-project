"""QuotaGate — admission control against a plan's limits (ADR-0004).

Reads Postgres directly rather than a Redis-cached counter. Redis is already
provisioned for when the query volume needs the cache; querying the
authoritative table first is correctness-first, not a regression against a
caching layer that has no load to justify it yet — the same call the
usage-metering migration's own header note makes about `usage_rollups`.

Two tiers, not the ADR's three: `allow` below 80%, `warn` from 80–100%
(work proceeds, the caller surfaces the warning), `deny` at 100% and above.
The ADR's 120% "kill" tier — cancelling an *already-running* job mid-flight —
needs a cancellation token threaded through the engine's `AgentLoop`, which
does not exist yet; admission-time `deny` is what's implemented today. Noted
here rather than silently narrowed.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from sqlalchemy import text

from lumen_api.db.session import user_session

SOFT = 0.8
HARD = 1.0

Metric = Literal[
    "llm_input_tokens", "llm_output_tokens", "compute_seconds",
    "rows_scanned", "storage_bytes_day", "agent_run",
]

_LIMIT_COLUMN: dict[str, str] = {
    "llm_input_tokens": "llm_input_tokens_limit",
    "llm_output_tokens": "llm_output_tokens_limit",
    "agent_run": "agent_runs_limit",
    "storage_bytes_day": "storage_bytes_limit",
}


class Decision(StrEnum):
    ALLOW = "allow"
    WARN = "warn"
    DENY = "deny"


@dataclass(frozen=True)
class QuotaStatus:
    metric: str
    used: float
    limit: float | None
    ratio: float | None  # None when limit is None (unlimited)
    decision: Decision

    def as_dict(self) -> dict[str, object]:
        return {
            "metric": self.metric,
            "used": self.used,
            "limit": self.limit,
            "ratio": round(self.ratio, 4) if self.ratio is not None else None,
            "decision": str(self.decision),
        }


class QuotaGate:
    def __init__(self, org_id: uuid.UUID, user_id: uuid.UUID) -> None:
        self._org_id = org_id
        self._user_id = user_id

    async def status(self, metric: Metric) -> QuotaStatus:
        column = _LIMIT_COLUMN.get(metric)
        if column is None:
            # compute_seconds / rows_scanned have no plan limit column yet —
            # metered (ADR-0004 item 4, tied to the worker) but not gated.
            return QuotaStatus(metric=metric, used=0.0, limit=None, ratio=None, decision=Decision.ALLOW)

        async with user_session(self._user_id) as db:
            row = (
                await db.execute(
                    text(
                        f"select p.{column} as limit_value, coalesce(u.quantity, 0) as used "
                        "from public.organizations o "
                        "join public.plans p on p.code = o.plan_code "
                        # usage_current_period casts metric to text (see the
                        # migration's note) — compare as text, not the enum.
                        "left join public.usage_current_period u "
                        "  on u.org_id = o.id and u.metric = :metric "
                        "where o.id = :org"
                    ),
                    {"org": self._org_id, "metric": metric},
                )
            ).mappings().first()

        if row is None or row["limit_value"] is None:
            used = float(row["used"]) if row else 0.0
            return QuotaStatus(metric=metric, used=used, limit=None, ratio=None, decision=Decision.ALLOW)

        limit = float(row["limit_value"])
        used = float(row["used"])
        ratio = used / limit if limit > 0 else float("inf")
        decision = (
            Decision.DENY if ratio >= HARD else Decision.WARN if ratio >= SOFT else Decision.ALLOW
        )
        return QuotaStatus(metric=metric, used=used, limit=limit, ratio=ratio, decision=decision)

    async def remaining(self, metric: Metric) -> float | None:
        """How much of this metric is left in the current period. None = unlimited."""
        status = await self.status(metric)
        if status.limit is None:
            return None
        return max(0.0, status.limit - status.used)

    async def record(
        self,
        db,
        *,
        metric: Metric,
        quantity: float,
        agent: str,
        run_id: uuid.UUID | None = None,
        metadata: dict | None = None,
    ) -> None:
        """Write one usage row, on the caller's own already-open session.

        Takes the session rather than opening one, so the write lands in the
        same transaction as whatever else that session is doing — a run's own
        insert, a run's own status update — matching the "same transaction as
        the work" rule this table exists to enforce.
        """
        import json

        await db.execute(
            text(
                "insert into public.usage_records (org_id, run_id, agent, metric, quantity, meta) "
                "values (:org, :run, :agent, :metric, :quantity, "
                "        cast(:metadata as jsonb))"
            ),
            {
                "org": self._org_id,
                "run": run_id,
                "agent": agent,
                "metric": metric,
                "quantity": quantity,
                "metadata": json.dumps(metadata or {}),
            },
        )
