"""ADR-0009: column-level entity resolution, piggybacked on the ADR-0008 tick.

Two costs, deliberately kept apart, the same reasoning `sentinel.py` already
applies to drift: a free, deterministic embedding search runs on every column
a tick finds new or changed; the model runs only on what that search actually
flags as a cross-source candidate. `find_cluster_candidates` never calls a
model and never writes anything — the columns it's asked about are already
written (`profiling.remember_columns`) before it searches. A candidate that
crosses the threshold becomes a `propose_entity_mapping` job, not a persisted
row: unlike a `DriftEvent`, an unmatched tick has nothing worth keeping
around, so there is no table for this — only this tick's search result and
the job it enqueues.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from sqlalchemy import text

from lumen.agents.loop import AgentLoop
from lumen.llm.base import ToolSpec
from lumen_api.agents.registry import Tool, ToolRegistry
from lumen_api.agents.runner import StreamingSink
from lumen_api.llm import provider
from lumen_api.billing.quota import Decision, QuotaGate
from lumen_api.context.store import ContextStore, Kind, Scope
from lumen_api.datasets.store import DatasetHandle
from lumen_api.db.session import user_session
from lumen_api.profiling import remember_columns

# Calibrated against this project's actual small local embedding model
# (fastembed), not picked in the abstract: a bare "name (dtype)" string is
# short enough that the dtype token alone drags unrelated columns together —
# in one measurement, 'region' vs 'customer_id' scored 0.61 and 'id' vs
# 'amount' scored 0.82, both clearly different concepts sharing only a dtype.
# A truly shared name scores ~1.0. There is no threshold that cleanly
# separates "genuinely similar wording" from "coincidentally shares a dtype"
# with this little text to embed — so this sits deliberately high, trading
# recall (a real rename with very different wording may not be caught) for
# precision (ADR-0009 §2's own conservatism: a false-positive merge teaches
# the org a wrong fact until someone notices and rejects it, which costs more
# than a missed one that a human catches by hand).
CLUSTER_THRESHOLD = 0.92
# Context shown to the model alongside a seed that already cleared the bar
# above — looser than CLUSTER_THRESHOLD, but still well above the ~0.6-0.8
# range coincidental dtype-sharing produces, so the model is not handed noise
# to filter itself.
CONTEXT_THRESHOLD = 0.85
MAX_CONTEXT_CANDIDATES = 5


async def detect_and_enqueue_clusters(
    ctx: dict[str, Any],
    memory: ContextStore,
    handle: DatasetHandle,
    org_id: uuid.UUID,
    source_id: uuid.UUID,
    acting_user_id: str,
    columns: list[str],
) -> int:
    """Embed `columns` (new or changed this tick), search for cross-source
    matches, and enqueue one `propose_entity_mapping` job per column that
    found one. Returns how many jobs were enqueued — purely informational,
    `process_schedule` doesn't branch on it.
    """
    written = await remember_columns(memory, handle, columns)
    candidates = await find_cluster_candidates(memory, source_id, written)
    for candidate in candidates:
        await ctx["redis"].enqueue_job(
            "propose_entity_mapping",
            str(org_id),
            acting_user_id,
            str(source_id),
            candidate["column"],
            candidate["candidates"],
        )
    return len(candidates)


async def find_cluster_candidates(
    memory: ContextStore, source_id: uuid.UUID, columns: list[tuple[uuid.UUID, str, str]]
) -> list[dict[str, Any]]:
    """For each `(entry_id, column, embedded_content)` triple, search org-wide
    for `Kind.SCHEMA` matches from *other* sources. Returns one entry per
    column whose best cross-source match clears `CLUSTER_THRESHOLD`.
    """
    candidates: list[dict[str, Any]] = []
    for _entry_id, column, content in columns:
        matches = await memory.search(
            content,
            kinds=[Kind.SCHEMA],
            scope=Scope.ORG,
            limit=MAX_CONTEXT_CANDIDATES + 3,
            min_similarity=CONTEXT_THRESHOLD,
        )
        cross_source = [m for m in matches if m.source_id is not None and m.source_id != source_id]
        if not cross_source:
            continue
        best = max(cross_source, key=lambda m: m.similarity)
        if best.similarity < CLUSTER_THRESHOLD:
            continue
        candidates.append(
            {
                "column": column,
                "candidates": [
                    {
                        "source_id": str(m.source_id),
                        "column": m.metadata.get("column"),
                        "dtype": m.metadata.get("dtype"),
                        "similarity": round(m.similarity, 4),
                    }
                    for m in cross_source[:MAX_CONTEXT_CANDIDATES]
                ],
            }
        )
    return candidates


GLOSSARY_SYSTEM_PROMPT = """\
You are Lumen's Glossary agent. A deterministic embedding search found columns \
across different data sources in this workspace whose names and types look \
like the same business concept — not a hunch, a similarity score already \
above threshold. Your one job is to decide whether they truly are the same \
concept, and if so, name it.

This is unattended: nobody is here to catch a wrong merge before it ships as \
a proposal. Be conservative — a false merge teaches this workspace a wrong \
fact that nobody notices until several questions have already used it.

- Call `propose_entity_mapping` only when the columns shown to you are \
  plausibly the same real-world concept under different names (a customer \
  identifier, an order reference) — not just similarly typed or similarly \
  named by coincidence (two unrelated 'status' columns; an 'id' that means \
  something different per source).
- Choose members conservatively: include only the columns you are confident \
  belong, not every candidate shown to you. Two members is enough; do not pad.
- If nothing shown to you is confidently the same concept, call \
  `not_the_same_entity`. An alert with no proposal costs nothing; a wrong \
  merge costs the workspace's trust in every proposal after it.
- Call exactly one tool. Do not ask questions; nobody is here to answer them.
"""


def _cluster_briefing(
    seed_source_id: str, seed_source_name: str, seed_column: str, candidates: list[dict[str, Any]]
) -> str:
    # source_id is spelled out for every row, seed included: propose_entity_mapping's
    # `members` are keyed by source_id, and a model can only reference a value
    # it was actually shown — a human-readable source name alone would leave
    # it no way to name what it means to merge.
    lines = [
        f"Seed column: source_id={seed_source_id} source_name={seed_source_name} "
        f"column={seed_column}",
        "Candidates from other sources:",
    ]
    lines.extend(
        f"- source_id={c['source_id']} source_name={c['source_name']} column={c['column']} "
        f"dtype={c.get('dtype', 'unknown')} similarity={c['similarity']:.2f}"
        for c in candidates
    )
    return "\n".join(lines)


async def propose_entity_mapping(
    ctx: dict[str, Any],
    org_id: str,
    acting_user_id: str,
    source_id: str,
    column: str,
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    """The model call the clustering search deliberately does not make
    inline. Given one seed column and its cross-source candidates (already
    above `CLUSTER_THRESHOLD`), decide whether they are the same business
    concept and, if so, name it — always as an `awaiting_review`
    `entity_mapping` proposal. There is no auto-apply tier for this kind
    (ADR-0009 §2): getting a business term wrong is a worse failure than a
    bad cleaning step.
    """
    org_uuid, user_uuid = uuid.UUID(org_id), uuid.UUID(acting_user_id)

    gate = QuotaGate(org_uuid, user_uuid)
    statuses = [
        await gate.status(metric) for metric in ("agent_run", "llm_input_tokens", "llm_output_tokens")
    ]
    if Decision.DENY in (status.decision for status in statuses):
        return {"status": "skipped", "reason": "quota exceeded"}

    valid_members = {(source_id, column)} | {(c["source_id"], c["column"]) for c in candidates}

    async with user_session(user_uuid) as db:
        # Dedupe: never re-propose a cluster that already has an
        # entity_mapping proposal covering it, accepted or rejected — "a
        # rejected cluster is not re-proposed" (ADR-0009 §2). The proposal
        # itself is the record; no second memory mechanism for this.
        existing = (
            await db.execute(
                text(
                    "select spec from public.proposals "
                    "where org_id = :org and kind = 'entity_mapping'"
                ),
                {"org": org_uuid},
            )
        ).scalars().all()
        for spec in existing:
            covered = {(m["source_id"], m["column"]) for m in (spec or {}).get("members", [])}
            if covered & valid_members:
                return {"status": "skipped", "reason": "already proposed or decided"}

        source_uuids = [uuid.UUID(sid) for sid in {source_id, *(c["source_id"] for c in candidates)}]
        rows = (
            await db.execute(
                text("select id, name from public.data_sources where id = any(:ids)"),
                {"ids": source_uuids},
            )
        ).mappings().all()
    names = {str(row["id"]): row["name"] for row in rows}

    run_id = uuid.uuid4()
    async with user_session(user_uuid) as db:
        await db.execute(
            text(
                "insert into public.runs (id, org_id, thread_id, kind, status, backend, created_by) "
                "values (:id, :org, :id, 'glossary_propose', 'running', 'polars', :user)"
            ),
            {"id": run_id, "org": org_uuid, "user": user_uuid},
        )
        await gate.record(db, metric="agent_run", quantity=1, agent="glossary", run_id=run_id)

    outcome: dict[str, Any] = {"decision": None, "proposal_id": None}

    async def propose(
        canonical_name: str,
        canonical_type: str,
        members: list[dict[str, str]],
        reconciliation_rule: dict[str, Any],
        rationale: str,
    ) -> dict[str, Any]:
        chosen = {(m.get("source_id"), m.get("column")) for m in members}
        if not chosen.issubset(valid_members):
            return {
                "ok": False,
                "error": f"members must come from what you were shown: {sorted(valid_members)}",
            }
        if len(chosen) < 2:
            return {"ok": False, "error": "an entity mapping needs at least two columns to reconcile"}

        async with user_session(user_uuid) as db:
            proposal_id = (
                await db.execute(
                    text(
                        "insert into public.proposals "
                        "(org_id, run_id, thread_id, author_agent, kind, spec, rationale) "
                        "values (:org, :run, :run, 'glossary', 'entity_mapping', "
                        "        cast(:spec as jsonb), :rationale) returning id"
                    ),
                    {
                        "org": org_uuid,
                        "run": run_id,
                        "spec": json.dumps(
                            {
                                "canonical_name": canonical_name,
                                "canonical_type": canonical_type,
                                "members": members,
                                "reconciliation_rule": reconciliation_rule,
                            }
                        ),
                        "rationale": rationale,
                    },
                )
            ).scalar_one()
        outcome["decision"] = "proposed"
        outcome["proposal_id"] = str(proposal_id)
        return {"ok": True, "data": {"proposal_id": str(proposal_id), "members": members}}

    async def decline(reason: str) -> dict[str, Any]:
        outcome["decision"] = "not_the_same_entity"
        return {"ok": True, "data": {"reason": reason}}

    registry = ToolRegistry(
        [
            Tool(
                ToolSpec(
                    name="propose_entity_mapping",
                    description=(
                        "Propose that two or more columns across sources are the same "
                        "business concept. Members must be chosen from the columns you "
                        "were shown, not invented."
                    ),
                    input_schema={
                        "type": "object",
                        "properties": {
                            "canonical_name": {
                                "type": "string",
                                "description": "The business term, e.g. 'customer'.",
                            },
                            "canonical_type": {
                                "type": "string",
                                "description": "What kind of thing this identifies, e.g. 'identifier'.",
                            },
                            "members": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "source_id": {"type": "string"},
                                        "column": {"type": "string"},
                                    },
                                    "required": ["source_id", "column"],
                                },
                            },
                            "reconciliation_rule": {
                                "type": "object",
                                "description": (
                                    "How to normalise values so members actually match "
                                    "— e.g. {\"strip\": \"-\", \"case\": \"upper\"}."
                                ),
                            },
                            "rationale": {
                                "type": "string",
                                "description": "Why these columns are judged equivalent.",
                            },
                        },
                        "required": [
                            "canonical_name",
                            "canonical_type",
                            "members",
                            "reconciliation_rule",
                            "rationale",
                        ],
                    },
                ),
                propose,
            ),
            Tool(
                ToolSpec(
                    name="not_the_same_entity",
                    description="Decline to merge — nothing shown is confidently the same concept.",
                    input_schema={
                        "type": "object",
                        "properties": {"reason": {"type": "string"}},
                        "required": ["reason"],
                    },
                ),
                decline,
            ),
        ]
    )

    sink = StreamingSink(org_uuid, user_uuid, run_id)
    loop = AgentLoop(
        provider(tier="specialist"),
        registry,
        agent_name="glossary",
        sink=sink,
        # One bounded decision over one candidate cluster — the same reduced
        # bounds sentinel.py's diagnose_drift uses, for the same reason.
        max_iterations=4,
        deadline_seconds=60.0,
        max_total_tokens=15_000,
    )

    briefing = _cluster_briefing(
        source_id,
        names.get(source_id, source_id),
        column,
        [{**c, "source_name": names.get(c["source_id"], c["source_id"])} for c in candidates],
    )
    error: str | None = None
    stop_reason = "failed"
    try:
        result = await loop.run(GLOSSARY_SYSTEM_PROMPT, briefing)
        stop_reason = result.stop_reason
    except Exception as exc:  # noqa: BLE001 — a failed proposal is a result, not a crashed worker
        error = f"{type(exc).__name__}: {exc}"

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
                agent="glossary", run_id=run_id,
            )
        if sink.output_tokens:
            await gate.record(
                db, metric="llm_output_tokens", quantity=sink.output_tokens,
                agent="glossary", run_id=run_id,
            )

    return {
        "status": outcome["decision"] or "no_decision",
        "proposal_id": outcome["proposal_id"],
        "run_id": str(run_id),
        "stop_reason": stop_reason,
    }
