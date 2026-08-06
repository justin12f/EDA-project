"""ImpactReport: what depends on this change, computed before a human ever
sees the proposal (ADR-0010).

Synchronous, not a background job — the ADR's own chosen path (§4's Option
D): a proposal is created with its impact report already attached, so
nobody watches a loading spinner on the review screen. Only meaningful for
a proposal that changes data (`cleaning_pipeline` / `pipeline_patch` — the
kinds with `steps` to replay); `entity_mapping` has no "before vs after
data" to diff, so `proposals.py` and `agents/registry.py` only call this
for the two pipeline kinds.

Only `canonical_entity` dependents produce a finding today. A `pipeline`
dependent on the *same* source is, in this product, always the pipeline
this proposal would itself replace — pipelines are strictly single-source,
so there is no case yet where one pipeline's change affects a *different*
pipeline. `analysis_widget` has no writer at all (see `lineage.py`'s own
note). Both are still queried and excluded/skipped explicitly rather than
never considered, so this does not have to change shape the day either
stops being true.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from sqlalchemy import text

from lumen.datasets.materialize import column_values
from lumen.llm.base import ChatMessage
from lumen_api.datasets.store import HandleStore
from lumen_api.db.session import user_session
from lumen_api.llm import provider
from lumen_api.settings import get_settings
from lumen_api.shadow_run import replay_frame

# Bounded, not exhaustive (ADR-0010 §4): a well-connected source gets the N
# most-recently-declared dependents and an honest "N of M" when the cap is
# hit — a silently truncated report is worse than no report, because it
# reads as complete when it is not.
MAX_DEPENDENTS = 10

# Below this, a match-rate shift reads as noise, not a finding worth a
# person's attention — the same shape as ADR-0008's null-rate drift
# threshold, on the same reasoning: a flat cutoff over a computed signal,
# not a guess.
CONSEQUENTIAL_DELTA = 0.02


async def compute_impact_report(
    org_id: uuid.UUID,
    user_id: uuid.UUID,
    *,
    proposal_id: uuid.UUID,
    source_id: uuid.UUID,
    rid: str,
    steps: list[dict[str, Any]],
) -> dict[str, Any]:
    async with user_session(user_id) as db:
        total = (
            await db.execute(
                text(
                    "select count(*) from public.artifact_dependencies "
                    "where org_id = :org and source_id = :source and artifact_kind != 'pipeline'"
                ),
                {"org": org_id, "source": source_id},
            )
        ).scalar_one()
        dependents = (
            await db.execute(
                text(
                    "select artifact_kind, artifact_id, columns from public.artifact_dependencies "
                    "where org_id = :org and source_id = :source and artifact_kind != 'pipeline' "
                    "order by created_at desc limit :limit"
                ),
                {"org": org_id, "source": source_id, "limit": MAX_DEPENDENTS},
            )
        ).mappings().all()

    findings: list[dict[str, Any]] = []
    if dependents:
        store = HandleStore(org_id, user_id)
        try:
            before_frame, after_frame = await replay_frame(store, rid, steps)
        except Exception:  # noqa: BLE001 — an unsimulable pipeline must not block proposal creation
            before_frame = after_frame = None

        if before_frame is not None:
            handle = await store.get(rid)
            for dependent in dependents:
                if dependent["artifact_kind"] != "canonical_entity":
                    continue
                finding = await _entity_finding(
                    org_id, user_id, handle.backend, before_frame, after_frame,
                    dependent, source_id,
                )
                if finding is not None:
                    findings.append(finding)

    summary = await _summarize(findings, len(dependents), int(total))

    async with user_session(user_id) as db:
        await db.execute(
            text(
                "insert into public.impact_reports "
                "(proposal_id, org_id, dependents_checked, dependents_total, findings, summary) "
                "values (:proposal, :org, :checked, :total, cast(:findings as jsonb), :summary)"
            ),
            {
                "proposal": proposal_id,
                "org": org_id,
                "checked": len(dependents),
                "total": int(total),
                "findings": json.dumps(findings),
                "summary": summary,
            },
        )

    return {
        "dependents_checked": len(dependents),
        "dependents_total": int(total),
        "findings": findings,
        "summary": summary,
    }


async def _entity_finding(
    org_id: uuid.UUID,
    user_id: uuid.UUID,
    backend: str,
    before_frame: Any,
    after_frame: Any,
    dependent: Any,
    source_id: uuid.UUID,
) -> dict[str, Any] | None:
    entity_id = dependent["artifact_id"]
    columns = dependent["columns"] or []
    if not columns:
        return None
    column = columns[0]

    async with user_session(user_id) as db:
        entity = (
            await db.execute(
                text("select reconciliation_rule from public.canonical_entities where id = :id"),
                {"id": entity_id},
            )
        ).mappings().first()
        other_members = (
            await db.execute(
                text(
                    "select source_id, column_name from public.canonical_entity_members "
                    "where entity_id = :entity and source_id != :source"
                ),
                {"entity": entity_id, "source": source_id},
            )
        ).mappings().all()
    if entity is None or not other_members:
        return None  # a single-source "entity" has nothing to match against

    # unique(source_id, column_name) means at most one member per other
    # source — the first is the strongest available signal, not an
    # arbitrary pick among several.
    other = other_members[0]
    store = HandleStore(org_id, user_id)
    other_handle = await store.latest_for_source(other["source_id"])
    if other_handle is None:
        return None
    other_frame = await store.resolve(other_handle.rid)

    rule = dict(entity["reconciliation_rule"] or {})
    other_values = _normalize_all(column_values(other_frame, other_handle.backend, other["column_name"]), rule)
    if not other_values:
        return None

    before_rate = _match_rate(_normalize_all(column_values(before_frame, backend, column), rule), other_values)
    after_rate = _match_rate(_normalize_all(column_values(after_frame, backend, column), rule), other_values)
    delta = after_rate - before_rate
    if abs(delta) < CONSEQUENTIAL_DELTA:
        return None

    return {
        "artifact_kind": "canonical_entity",
        "artifact_id": str(entity_id),
        "metric": "match_rate",
        "before": round(before_rate, 4),
        "after": round(after_rate, 4),
        "delta_pct": round(delta, 4),
    }


def _normalize(value: str, rule: dict[str, Any]) -> str | None:
    text_ = value
    strip_chars = rule.get("strip")
    if strip_chars:
        for char in strip_chars:
            text_ = text_.replace(char, "")
    case = rule.get("case")
    if case == "upper":
        text_ = text_.upper()
    elif case == "lower":
        text_ = text_.lower()
    text_ = text_.strip()
    return text_ or None


def _normalize_all(values: set[str], rule: dict[str, Any]) -> set[str]:
    normalized = {_normalize(v, rule) for v in values}
    normalized.discard(None)
    return normalized  # type: ignore[return-value]


def _match_rate(values: set[str], other: set[str]) -> float:
    if not values:
        return 0.0
    return len(values & other) / len(values)


async def _summarize(findings: list[dict[str, Any]], checked: int, total: int) -> str:
    fallback = _fallback_summary(findings, checked, total)
    if not findings or get_settings().resolved_llm_mode == "mock":
        # Nothing to summarize, or no real model configured worth spending a
        # call on — the deterministic sentence already says everything a
        # model would only be paraphrasing.
        return fallback

    prompt = (
        "Summarize these findings in one short, plain sentence for someone about "
        "to approve a data-cleaning change:\n"
        + "\n".join(
            f"- {f['artifact_kind']} {f['artifact_id']}: {f['metric']} moved from "
            f"{f['before']:.2f} to {f['after']:.2f} ({f['delta_pct']:+.1%})"
            for f in findings
        )
    )
    try:
        # Presentation of numbers the deterministic simulation already
        # produced, not a decision — the fast tier, not the reasoning one
        # (ADR-0008 applies the same cost discipline to drift diagnosis).
        response = await provider(tier="fast").complete(
            [ChatMessage(role="user", content=prompt)], [], max_tokens=200
        )
        return (response.text or "").strip() or fallback
    except Exception:  # noqa: BLE001 — a missing/failed summary must not block proposal creation
        return fallback


def _fallback_summary(findings: list[dict[str, Any]], checked: int, total: int) -> str:
    coverage = f"{checked} of {total}" if total > checked else str(checked)
    if not findings:
        if checked == 0:
            return "No other artifacts depend on this source and these columns."
        return f"Checked {coverage} dependent(s); none showed a consequential change."
    lines = [
        f"{f['artifact_kind']} {f['artifact_id']}: {f['metric']} {f['before']:.2f} -> "
        f"{f['after']:.2f} ({f['delta_pct']:+.1%})"
        for f in findings
    ]
    return "; ".join(lines) + f". ({coverage} dependents checked.)"
