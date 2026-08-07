"""ADR-0008: the scheduled tick.

Two functions, a dispatcher and a worker, because arq's own `cron()` is a
static, Python-level schedule (fixed hour/minute sets baked into
`WorkerSettings` at import time) — it cannot be populated per-row from a
database at runtime. `dispatch_due_schedules` is the only thing actually
registered as a cron job; it runs frequently and cheaply (one query, no data
touched) and enqueues `process_schedule` once per source whose `next_run_at`
has passed. That job does the real work — re-profile, diff, maybe diagnose —
so a slow source's profiling never blocks the dispatch of every other org's
tick behind it.

No model call anywhere in `process_schedule`. Detection is `lumen.sentinel`'s
deterministic check; a `DriftEvent` it writes is *diagnosed* — the model call,
`diagnose_drift` below — by a second job this one enqueues, never inline.
Keeping the tick itself model-free is what makes it safe to run against every
scheduled source regardless of whether anything changed; keeping diagnosis a
separate job is what stops a slow or quota-denied diagnosis from holding up
that tick, or another source's, behind it.
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from datetime import timedelta
from typing import Any, Literal

from sqlalchemy import text

from lumen.agents.loop import AgentLoop
from lumen.agents.master_factory import AgentMasterFactory
from lumen.llm.base import ToolSpec
from lumen.sentinel import detect_drift
from lumen_api.agents.registry import Tool, ToolRegistry, registered_steps
from lumen_api.agents.runner import StreamingSink
from lumen_api.apply_pipeline import apply_cleaning_pipeline
from lumen_api.baselines import compute_and_remember_source_baseline, prepare_and_update_column_baselines
from lumen_api.billing.quota import Decision, QuotaGate
from lumen_api.context.store import ContextStore
from lumen_api.datasets.store import HandleStore, SupabaseStorage
from lumen_api.db.session import service_session, user_session
from lumen_api.global_patterns import drift_signature, is_corroborated, lookup, record_occurrence
from lumen_api.impact import compute_impact_report
from lumen_api.llm import provider
from lumen_api.profiling import profile_and_remember
from lumen_api.shadow_run import ShadowRunResult, shadow_run
from lumen_api.trust import is_auto_apply_eligible, structural_shape
from lumen_worker.glossary import detect_and_enqueue_clusters

# The `cron` column is a small named interval, not crontab syntax — arq has no
# use for a stored crontab string (its own schedule is fixed at import time),
# so the column only ever has to answer "how long until the next tick", which
# a lookup table answers without a parsing dependency. An unrecognized value
# (or none) falls back to daily rather than erroring a tick over a typo.
_INTERVALS: dict[str, timedelta] = {
    "hourly": timedelta(hours=1),
    "daily": timedelta(days=1),
    "weekly": timedelta(weeks=1),
}


def _interval_for(cron: str | None) -> timedelta:
    return _INTERVALS.get(cron or "daily", _INTERVALS["daily"])


async def dispatch_due_schedules(ctx: dict[str, Any]) -> int:
    """The cron job. Finds every enabled schedule whose `next_run_at` has
    passed, across every org — this runs with no signed-in person behind it,
    so it reads through `service_session()`, exactly the cross-tenant
    maintenance that entry point's own docstring reserves itself for.
    """
    async with service_session() as db:
        due = (
            await db.execute(
                text(
                    "select id, org_id, source_id, created_by "
                    "from public.data_source_schedules "
                    "where enabled and next_run_at <= now()"
                )
            )
        ).mappings().all()

    for row in due:
        await ctx["redis"].enqueue_job(
            "process_schedule",
            str(row["id"]),
            str(row["org_id"]),
            str(row["source_id"]),
            str(row["created_by"]) if row["created_by"] else None,
        )
    return len(due)


async def process_schedule(
    ctx: dict[str, Any],
    schedule_id: str,
    org_id: str,
    source_id: str,
    acting_user_id: str | None,
) -> dict[str, Any]:
    """One source, one tick. Re-profiles it exactly the way a chat turn's
    `read_source` + `profile_source` would, diffs the result against the last
    stored profile, and writes a `DriftEvent` if `lumen.sentinel.detect_drift`
    finds something. Advances `next_run_at` regardless of outcome — a source
    with nothing wrong still needs its next check scheduled.
    """
    schedule_uuid = uuid.UUID(schedule_id)
    org_uuid = uuid.UUID(org_id)
    source_uuid = uuid.UUID(source_id)

    async with service_session() as db:
        source = (
            await db.execute(
                text("select object_path, name from public.data_sources where id = :id"),
                {"id": source_uuid},
            )
        ).mappings().first()
        schedule = (
            await db.execute(
                text("select cron from public.data_source_schedules where id = :id"),
                {"id": schedule_uuid},
            )
        ).mappings().first()

    cron = schedule["cron"] if schedule else None

    if source is None or not source["object_path"]:
        await _mark_run(schedule_uuid, cron)
        return {"status": "skipped", "reason": "source has no uploaded file yet"}

    # There is no "system" identity in this schema, and every read of tenant
    # data goes through a real member's RLS session — scheduled tick or not.
    # A schedule always has a creator (the migration's `created_by` column),
    # but a defensive check here is cheaper than a confusing RLS-empty result.
    if acting_user_id is None:
        await _mark_run(schedule_uuid, cron)
        return {"status": "skipped", "reason": "schedule has no creator to act as"}
    user_uuid = uuid.UUID(acting_user_id)

    store = HandleStore(org_uuid, user_uuid)
    memory = ContextStore(org_uuid, user_uuid)
    baseline = await memory.latest_profile(source_uuid)

    payload = await SupabaseStorage().download(source["object_path"])
    directory = tempfile.mkdtemp(prefix="lumen-sentinel-")
    suffix = os.path.splitext(source["object_path"])[1] or ".csv"
    local = os.path.join(directory, f"source{suffix}")
    with open(local, "wb") as file:
        file.write(payload)

    frame = AgentMasterFactory("polars").readers().read(local)
    handle = await store.put(frame, label=source["name"], source_id=source_uuid)
    # The same call the chat tool's `profile_source` makes (`profiling.py`'s own
    # docstring on why this must be one implementation, not two): it both
    # computes this tick's null rates AND persists them as the profile the
    # *next* tick reads back as `baseline`. A tick that only computed and never
    # wrote one would never have anything to diff against, no matter how many
    # times it ran.
    profiled = await profile_and_remember(memory, handle, frame, handle.rid)

    await _mark_run(schedule_uuid, cron)

    cold_start = baseline is None or "schema" not in baseline

    # ADR-0013: every column's baseline updates every tick, cold start or
    # not — a column needs its first observation recorded before it can
    # ever become calibrated (nothing here looks anomalous until then;
    # `NumericBaseline.is_within_band`/`categorical_shift` both no-op below
    # `MIN_SAMPLE_SIZE`). `null_rate_baselines` is this scan's state
    # *before* this tick's null rates fold in — exactly what `detect_drift`
    # needs below, so a comparison is never made against a band this same
    # scan just widened to include itself.
    async with user_session(user_uuid) as db:
        null_rate_baselines, value_findings = await prepare_and_update_column_baselines(
            db, org_uuid, source_uuid, frame, handle.backend, handle.schema, handle.row_count, profiled.null_rates,
        )
        volume_finding = await compute_and_remember_source_baseline(db, org_uuid, source_uuid, handle.row_count)

        # Alerts a person triages, not a shape diagnose_drift's patch-
        # proposing prompt knows how to address — there is rarely a
        # cleaning step for "this column's typical value moved" or "the
        # table stopped growing." Recorded for the drift feed at the
        # default status='detected' and never enqueued to diagnose_drift
        # the way a schema/null-rate event is below.
        if value_findings:
            await db.execute(
                text(
                    "insert into public.drift_events "
                    "(org_id, source_id, schedule_id, kind, severity, details) "
                    "values (:org, :source, :schedule, 'value_drift', :severity, cast(:details as jsonb))"
                ),
                {
                    "org": org_uuid,
                    "source": source_uuid,
                    "schedule": schedule_uuid,
                    "severity": min(1.0, len(value_findings) * 0.3),
                    "details": json.dumps({"findings": value_findings}, default=str),
                },
            )
        if volume_finding:
            await db.execute(
                text(
                    "insert into public.drift_events "
                    "(org_id, source_id, schedule_id, kind, severity, details) "
                    "values (:org, :source, :schedule, 'volume_freshness', :severity, cast(:details as jsonb))"
                ),
                {
                    "org": org_uuid,
                    "source": source_uuid,
                    "schedule": schedule_uuid,
                    "severity": _volume_severity(volume_finding),
                    "details": json.dumps(volume_finding, default=str),
                },
            )

    result = None if cold_start else detect_drift(
        baseline["schema"],
        handle.schema,
        baseline.get("null_rates", {}),
        profiled.null_rates,
        null_rate_baselines=null_rate_baselines,
    )

    # ADR-0009: whatever is new or changed about this schema is also what the
    # glossary's column clustering cares about — a cold start means every
    # column is new; otherwise only what detect_drift's own schema diff
    # already found (added/renamed/type_changed — a removed column has
    # nothing new to embed).
    changed_columns = (
        list(handle.schema)
        if cold_start
        else [
            change.column
            for change in (result.schema_changes if result else [])
            if change.kind in ("added", "renamed", "type_changed")
        ]
    )
    if changed_columns:
        await detect_and_enqueue_clusters(
            ctx, memory, handle, org_uuid, source_uuid, acting_user_id, changed_columns
        )

    if cold_start:
        # First scan of this source — nothing to diff against yet. Not
        # drift; a cold start. The profile just written above is what the
        # *next* tick will read back as `baseline`.
        return {"status": "baseline_recorded"}
    if result is None:
        return {"status": "no_drift"}

    async with user_session(user_uuid) as db:
        event_id = (
            await db.execute(
                text(
                    "insert into public.drift_events "
                    "(org_id, source_id, schedule_id, kind, severity, details) "
                    "values (:org, :source, :schedule, :kind, :severity, cast(:details as jsonb)) "
                    "returning id"
                ),
                {
                    "org": org_uuid,
                    "source": source_uuid,
                    "schedule": schedule_uuid,
                    "kind": result.kind,
                    "severity": result.severity,
                    "details": json.dumps(result.as_details(), default=str),
                },
            )
        ).scalar_one()

    # Detection stays model-free; diagnosis is its own job so a slow or
    # quota-denied diagnosis never holds up this tick, or another source's
    # behind it.
    await ctx["redis"].enqueue_job("diagnose_drift", str(event_id), org_id, acting_user_id)
    return {"status": "drift_detected", "drift_event_id": str(event_id), "severity": result.severity}


async def _mark_run(schedule_id: uuid.UUID, cron: str | None) -> None:
    async with service_session() as db:
        await db.execute(
            text(
                "update public.data_source_schedules set last_run_at = now(), "
                "       next_run_at = now() + :interval where id = :id"
            ),
            {"interval": _interval_for(cron), "id": schedule_id},
        )


SENTINEL_SYSTEM_PROMPT = """\
You are Lumen's Data Sentinel. A scheduled tick found drift on a data source — \
a deterministic schema or null-rate check flagged it, not a hunch — and your one \
job is to decide whether a cleaning-pipeline patch would address it.

This is unattended: nobody is here to answer a question or catch a bad guess \
before it ships. Be conservative.

- Propose a patch only for a mechanical, obvious correction to the drift shown \
  below — a renamed column an existing step can be repointed at, a new column \
  that plainly needs the same treatment as its siblings, a null-rate jump an \
  imputation step already covers.
- If the drift looks like a real change to the data itself — a column gone \
  because a source system stopped sending it, a null-rate jump with no evident \
  cause — call `cannot_diagnose`. A wrong guess applied unattended costs more \
  than an alert with no proposal.
- Call exactly one tool. Do not ask questions; nobody is here to answer them.
"""


def _drift_briefing(
    source_name: str,
    kind: str,
    severity: float,
    details: dict[str, Any],
    schema: dict[str, str],
    backend: str,
) -> str:
    lines = [f"Source: {source_name}", f"Drift kind: {kind} (severity {severity:.2f})"]
    for change in details.get("schema_changes", []):
        lines.append(f"- schema: {change.get('detail')}")
    for shift in details.get("null_rate_shifts", []):
        lines.append(
            f"- nulls: '{shift.get('column')}' moved from {shift.get('previous', 0):.1%} "
            f"to {shift.get('current', 0):.1%}"
        )
    lines.append(f"\nCurrent schema: {json.dumps(schema)}")
    lines.append(f"Valid cleaning-pipeline step names: {', '.join(registered_steps(backend))}")
    return "\n".join(lines)


def _volume_severity(finding: dict[str, Any]) -> float:
    """How far past its own band this scan's row-count delta landed, scaled
    to 0..1. `DEFAULT_Z` (3.0) is the band edge, so a delta that just barely
    breached reads as low severity; one many multiples of the column's own
    stdev past it reads as high. A zero-stdev baseline (every prior scan
    saw the exact same delta) has no scale to measure against — a flat
    mid-severity rather than a division by zero.
    """
    stdev = finding.get("baseline_stdev") or 0.0
    if stdev <= 0:
        return 0.5
    z_actual = abs(finding["observed_delta"] - finding["baseline_mean"]) / stdev
    return min(1.0, z_actual / 10.0)


def _classify_confidence(
    kind: str, schema_changes: list[dict[str, Any]], shadow: ShadowRunResult
) -> Literal["high", "medium", "low"]:
    """Deterministic, not the model's self-report — matching every other
    confidence signal in this codebase (drift severity itself, forecaster
    selection). High is reserved for exactly the case
    `data_source_schedules.auto_apply_enabled`'s own migration comment
    describes: additive-only schema drift a shadow run confirms barely moves
    the data. A row-count swing over half the dataset is not a small
    correction regardless of how reasonable the model's rationale reads.
    """
    if not shadow.ok:
        return "low"
    before = max(shadow.row_count_before, 1)
    row_delta_ratio = abs(shadow.row_count_after - shadow.row_count_before) / before
    if row_delta_ratio > 0.5:
        return "low"
    additive_only = kind == "schema_change" and schema_changes and all(
        c.get("kind") == "added" for c in schema_changes
    )
    if additive_only and row_delta_ratio < 0.05:
        return "high"
    return "medium"


async def diagnose_drift(
    ctx: dict[str, Any], drift_event_id: str, org_id: str, acting_user_id: str
) -> dict[str, Any]:
    """The model call `process_schedule` deliberately does not make. Given one
    drift event, decide whether a cleaning-pipeline patch addresses it and how
    confidently (`_classify_confidence`), propose it, and auto-apply only the
    high-confidence case on a schedule that opted in.
    """
    event_uuid, org_uuid, user_uuid = (
        uuid.UUID(drift_event_id), uuid.UUID(org_id), uuid.UUID(acting_user_id),
    )

    async with service_session() as db:
        event = (
            await db.execute(
                text(
                    "select e.source_id, e.schedule_id, e.kind, e.severity, e.details, "
                    "       s.name as source_name, coalesce(sch.auto_apply_enabled, false) as auto_apply_enabled "
                    "from public.drift_events e "
                    "join public.data_sources s on s.id = e.source_id "
                    "left join public.data_source_schedules sch on sch.id = e.schedule_id "
                    "where e.id = :id"
                ),
                {"id": event_uuid},
            )
        ).mappings().first()
    if event is None:
        return {"status": "skipped", "reason": "drift event no longer exists"}

    gate = QuotaGate(org_uuid, user_uuid)
    run_quota, input_quota, output_quota = (
        await gate.status("agent_run"),
        await gate.status("llm_input_tokens"),
        await gate.status("llm_output_tokens"),
    )
    if Decision.DENY in (run_quota.decision, input_quota.decision, output_quota.decision):
        # Left at 'detected', not 'dismissed' — this org's quota, not the
        # drift itself, is why nothing happened. A person upgrading the plan
        # should still find this waiting, not silently discarded.
        return {"status": "skipped", "reason": "quota exceeded"}

    store = HandleStore(org_uuid, user_uuid)
    handle = await store.latest_for_source(event["source_id"])
    if handle is None:
        return {"status": "skipped", "reason": "source has no materialised data yet"}

    run_id = uuid.uuid4()
    async with user_session(user_uuid) as db:
        await db.execute(
            text("update public.drift_events set status = 'diagnosing' where id = :id"),
            {"id": event_uuid},
        )
        await db.execute(
            text(
                "insert into public.runs (id, org_id, source_id, thread_id, kind, status, backend, created_by) "
                "values (:id, :org, :source, :id, 'sentinel_diagnose', 'running', :backend, :user)"
            ),
            {
                "id": run_id,
                "org": org_uuid,
                "source": event["source_id"],
                "backend": handle.backend,
                "user": user_uuid,
            },
        )
        await gate.record(db, metric="agent_run", quantity=1, agent="sentinel", run_id=run_id)

    outcome: dict[str, Any] = {"decision": None}

    async def propose_patch(steps: list[dict[str, Any]], rationale: str) -> dict[str, Any]:
        shadow = await shadow_run(store, handle.rid, steps)
        if not shadow.ok:
            return {"ok": False, "error": f"Shadow run failed: {shadow.error}"}

        confidence = _classify_confidence(
            event["kind"], event["details"].get("schema_changes", []), shadow
        )

        # ADR-0012: a signature comparable across every org that has opted
        # in, not just this one's own history. Computed regardless of this
        # org's own contribution setting — consuming the shared library is
        # free (§2) — and consulted only when structural confidence alone
        # left this at 'low': a corroborated global pattern can raise the
        # starting point to 'medium', never straight to 'high', and never
        # skips the shadow run that already ran above it.
        pattern_signature = drift_signature(event["kind"], event["details"])
        fix_signature = structural_shape("pipeline_patch", {"steps": steps})
        if confidence == "low" and is_corroborated(await lookup(pattern_signature, fix_signature)):
            confidence = "medium"

        if confidence == "low":
            outcome["decision"] = "low_confidence"
            async with user_session(user_uuid) as db:
                await db.execute(
                    text(
                        "update public.drift_events set status = 'dismissed', "
                        "       details = details || cast(:note as jsonb) where id = :id"
                    ),
                    {
                        "note": json.dumps(
                            {
                                "diagnosis": "low_confidence",
                                "rationale": rationale,
                                "shadow_run": shadow.as_dict(),
                            },
                            default=str,
                        ),
                        "id": event_uuid,
                    },
                )
            return {
                "ok": True,
                "data": {
                    "confidence": "low",
                    "proposal_created": False,
                    "reason": (
                        "This patch's effect is too large or uncertain to propose "
                        "automatically; the drift is left for manual review instead."
                    ),
                },
            }

        async with user_session(user_uuid) as db:
            proposal_id = (
                await db.execute(
                    text(
                        "insert into public.proposals "
                        "(org_id, run_id, thread_id, author_agent, kind, spec, rationale) "
                        "values (:org, :run, :run, 'sentinel', 'pipeline_patch', "
                        "        cast(:spec as jsonb), :rationale) returning id"
                    ),
                    {
                        "org": org_uuid,
                        "run": run_id,
                        "spec": json.dumps(
                            {
                                "rid": handle.rid,
                                "steps": steps,
                                "drift_event_id": drift_event_id,
                                # ADR-0012: stamped in now so decide_proposal can
                                # record this fix's eventual accept/reject against
                                # the same problem signature without a second
                                # drift_events lookup.
                                "drift_pattern_signature": pattern_signature,
                            }
                        ),
                        "rationale": rationale,
                    },
                )
            ).scalar_one()
            await db.execute(
                text(
                    "update public.drift_events set status = 'proposed', proposal_id = :proposal "
                    "where id = :id"
                ),
                {"proposal": proposal_id, "id": event_uuid},
            )
            # ADR-0012: this occurrence — a fix of this shape was proposed
            # for a problem of this shape — is a no-op unless this org has
            # opted into contribution; see record_occurrence's own gate.
            await record_occurrence(db, org_uuid, pattern_signature, fix_signature)

        # Synchronous, on purpose (ADR-0010 §4's Option D): this is exactly
        # the case that motivates it — a Sentinel patch appearing at 6am with
        # nobody who remembers the pipeline's assumptions in the room, so the
        # report has to already be there when a person finally opens it.
        await compute_impact_report(
            org_uuid, user_uuid,
            proposal_id=proposal_id, source_id=event["source_id"], rid=handle.rid, steps=steps,
        )

        # ADR-0011's second, independent gate: structural confidence alone
        # is no longer sufficient. "Never decided on before" and "explicitly
        # earned trust" must not be confused — a pattern with no row in
        # pattern_trust_scores is exactly as ineligible as one with a row
        # and too short a streak (is_auto_apply_eligible's own contract).
        pattern = structural_shape("pipeline_patch", {"steps": steps})
        async with user_session(user_uuid) as db:
            trusted = await is_auto_apply_eligible(db, org_uuid, pattern)

        applied = False
        if confidence == "high" and event["auto_apply_enabled"] and trusted:
            applied_outcome = await apply_cleaning_pipeline(
                org_uuid,
                user_uuid,
                thread_id=run_id,
                rid=handle.rid,
                steps=steps,
                rationale=f"Auto-applied by the Data Sentinel (high confidence): {rationale}",
                run_kind="sentinel_auto_apply",
                proposal_id=proposal_id,
            )
            async with user_session(user_uuid) as db:
                await db.execute(
                    text(
                        "update public.proposals set status = 'applied', decided_by = :user, "
                        "       decided_at = now(), applied_run_id = :run where id = :id"
                    ),
                    {
                        "user": user_uuid,
                        "run": applied_outcome["applied_run_id"],
                        "id": proposal_id,
                    },
                )
                await db.execute(
                    text("update public.drift_events set status = 'resolved' where id = :id"),
                    {"id": event_uuid},
                )
            applied = True

        outcome["decision"] = "applied" if applied else "proposed"
        return {
            "ok": True,
            "data": {
                "confidence": confidence,
                "proposal_id": str(proposal_id),
                "proposal_created": True,
                "auto_applied": applied,
            },
        }

    async def cannot_diagnose(reason: str) -> dict[str, Any]:
        outcome["decision"] = "cannot_diagnose"
        async with user_session(user_uuid) as db:
            await db.execute(
                text(
                    "update public.drift_events set status = 'dismissed', "
                    "       details = details || cast(:note as jsonb) where id = :id"
                ),
                {
                    "note": json.dumps({"diagnosis": "cannot_diagnose", "reason": reason}),
                    "id": event_uuid,
                },
            )
        return {"ok": True, "data": {"dismissed": True, "reason": reason}}

    registry = ToolRegistry(
        [
            Tool(
                ToolSpec(
                    name="propose_patch",
                    description=(
                        "Propose a cleaning-pipeline patch for this drift. Steps are "
                        "shadow-run against the source's current data before anything is "
                        "decided — an invalid or too-large-impact patch never reaches a "
                        "person as a proposal claiming more confidence than it has."
                    ),
                    input_schema={
                        "type": "object",
                        "properties": {
                            "steps": {"type": "array", "items": {"type": "object"}},
                            "rationale": {
                                "type": "string",
                                "description": "Why this addresses the drift shown to you.",
                            },
                        },
                        "required": ["steps", "rationale"],
                    },
                ),
                propose_patch,
            ),
            Tool(
                ToolSpec(
                    name="cannot_diagnose",
                    description=(
                        "No cleaning-pipeline patch confidently addresses this drift. "
                        "Leaves it for a person to look at instead of guessing."
                    ),
                    input_schema={
                        "type": "object",
                        "properties": {"reason": {"type": "string"}},
                        "required": ["reason"],
                    },
                ),
                cannot_diagnose,
            ),
        ]
    )

    sink = StreamingSink(org_uuid, user_uuid, run_id)
    loop = AgentLoop(
        provider(tier="specialist"),
        registry,
        agent_name="sentinel",
        sink=sink,
        # One bounded decision over one drift event, not an open-ended chat —
        # the analyst's much larger bounds (settings.agent_*) would only let
        # a confused run spend more before stopping.
        max_iterations=6,
        deadline_seconds=90.0,
        max_total_tokens=20_000,
    )

    briefing = _drift_briefing(
        event["source_name"],
        event["kind"],
        event["severity"],
        event["details"],
        handle.schema,
        handle.backend,
    )
    error: str | None = None
    stop_reason = "failed"
    try:
        result = await loop.run(SENTINEL_SYSTEM_PROMPT, briefing)
        stop_reason = result.stop_reason
    except Exception as exc:  # noqa: BLE001 — a failed diagnosis is a result, not a crashed worker
        error = f"{type(exc).__name__}: {exc}"

    if outcome["decision"] is None:
        # Neither tool committed a terminal status — the model talked instead
        # of deciding, or ran out of iterations/budget/deadline/raised. Left
        # at 'diagnosing' forever this event would never resurface: the next
        # tick only re-detects drift from a fresh baseline, it does not retry
        # a stuck diagnosis.
        async with user_session(user_uuid) as db:
            await db.execute(
                text(
                    "update public.drift_events set status = 'dismissed', "
                    "       details = details || cast(:note as jsonb) where id = :id"
                ),
                {
                    "note": json.dumps({"diagnosis": "no_decision", "stop_reason": stop_reason}),
                    "id": event_uuid,
                },
            )

    async with user_session(user_uuid) as db:
        await db.execute(
            text(
                "update public.runs set status = cast(:status as public.run_status), "
                "       finished_at = now(), error = :error where id = :id"
            ),
            {"status": "failed" if error else "succeeded", "error": error, "id": run_id},
        )
        if sink.input_tokens:
            await gate.record(
                db, metric="llm_input_tokens", quantity=sink.input_tokens,
                agent="sentinel", run_id=run_id,
            )
        if sink.output_tokens:
            await gate.record(
                db, metric="llm_output_tokens", quantity=sink.output_tokens,
                agent="sentinel", run_id=run_id,
            )

    return {"status": outcome["decision"] or "no_decision", "run_id": str(run_id), "stop_reason": stop_reason}
