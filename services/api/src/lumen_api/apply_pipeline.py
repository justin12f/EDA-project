"""Actually run a cleaning pipeline and materialize the result.

One implementation, two callers: a human accepting a `cleaning_pipeline` or
`pipeline_patch` proposal (`proposals.py`), and the Sentinel auto-applying a
high-confidence patch a human never had to review (`agents/sentinel.py`,
ADR-0008 §4). Same steps, same spec shape, same everything except who decided
it and why — which is exactly the part this function does not need to know.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import text

from lumen.data_cleaning.data_cleaning_pipeline import PipelineBuilder
from lumen_api.context.store import ContextEntry, ContextStore, Kind
from lumen_api.datasets.store import HandleStore
from lumen_api.db.session import user_session
from lumen_api.jsonable import jsonable


async def apply_cleaning_pipeline(
    org_id: uuid.UUID,
    user_id: uuid.UUID,
    *,
    thread_id: uuid.UUID,
    rid: str,
    steps: list[dict[str, Any]],
    rationale: str,
    run_kind: str = "apply_pipeline",
) -> dict[str, Any]:
    """Runs `steps` against `rid`, writes the cleaned frame as a new handle,
    records a `runs` row, and remembers the outcome as an org-scoped fact.
    Returns the new handle's facts and a human-readable report — everything a
    caller needs to close out whatever proposal asked for this, without this
    function knowing what a proposal is.
    """
    store = HandleStore(org_id, user_id)
    handle = await store.get(rid)
    frame = await store.resolve(rid)

    pipeline = PipelineBuilder(frame).build(steps)
    cleaned = pipeline.run(frame)
    new_handle = await store.put(
        cleaned, backend=handle.backend, label=f"cleaned: {handle.label or rid}"
    )

    async with user_session(user_id) as db:
        new_run_id = (
            await db.execute(
                text(
                    "insert into public.runs (org_id, source_id, thread_id, kind, status, backend, "
                    "                         created_by, finished_at) "
                    "values (:org, :source, :thread, :kind, 'succeeded', :backend, :user, now()) "
                    "returning id"
                ),
                {
                    "org": org_id,
                    # The *original* rid's source — a certification check
                    # (ADR-0009) needs "the currently accepted pipeline run
                    # for source X" to be findable by source_id, whichever
                    # path applied it.
                    "source": handle.source_id,
                    "thread": thread_id,
                    "kind": run_kind,
                    "backend": handle.backend,
                    "user": user_id,
                },
            )
        ).scalar_one()

    summary = _report_summary(pipeline.report)
    try:
        await ContextStore(org_id, user_id).remember(
            ContextEntry(
                kind=Kind.RATIONALE,
                title=f"Cleaning applied to {handle.label or rid}",
                content=f"{rationale}\n\n{summary}",
                rid=new_handle.rid,
                run_id=new_run_id,
                metadata={"steps": jsonable(pipeline.report.steps)},
            )
        )
    except Exception:  # noqa: BLE001 — the pipeline already ran; memory is best-effort
        pass

    return {
        "applied_run_id": new_run_id,
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
