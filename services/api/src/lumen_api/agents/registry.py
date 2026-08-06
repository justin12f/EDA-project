"""Tools are thin typed wrappers over engine capabilities.

Two rules make this layer safe:

1. **Every handler returns `{"ok": bool, "data"|"error": ...}` and never raises.**
   A failure is a result the model can read and recover from. The loop turns an
   escaped exception into the same shape, but a tool that gets there has already
   lost the model its chance to explain itself.

2. **`propose_cleaning_pipeline` validates but does not execute.** It builds the
   plan through the engine's own `PipelineBuilder`, so an invented step name
   fails here — before a human is asked to approve it, and long before the worker
   would run it. This is what makes "the agent's output is the engine's input"
   more than a slogan.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from sqlalchemy import text

from lumen.agents.master_factory import AgentMasterFactory
from lumen.data_cleaning.data_cleaning_pipeline import PipelineBuilder
from lumen.data_cleaning.step_factory import AbstractDataCleaningStepFactory
from lumen.datasets.materialize import duplicate_counts, null_rates
from lumen.llm.base import ToolSpec
import numpy as np

from lumen.prediction import PredictorRegistry, Task
from lumen.prediction.extract import to_series
from lumen.prediction.timeseries import detect_season
from lumen_api.billing.quota import Decision, QuotaGate
from lumen_api.context.store import ContextEntry, ContextStore, Kind, Scope
from lumen_api.datasets.store import HandleStore
from lumen_api.db.session import user_session
from lumen_api.jsonable import jsonable

# Read-mostly account tools (ADR-0004's AdminAgent). Kept in the same flat
# registry as the data tools rather than a second orchestrator: the current
# system is one agent with a tool registry, not a supervisor of named
# specialist agents, and "AdminAgent" is a scope of tools, not a separate
# process to stand up.
PLAN_ORDER = ("free", "pro", "team", "ent")
QUOTA_METRICS = ("llm_input_tokens", "llm_output_tokens", "agent_run")

Handler = Callable[..., Awaitable[dict[str, Any]]]

# Columns worth checking for duplicates: an id-like or hash-like name. Checking
# every column on a wide frame costs more than it tells you.
DUPLICATE_HINTS = ("id", "hash", "email", "key", "code", "uuid")


@dataclass(frozen=True)
class Tool:
    spec: ToolSpec
    handler: Handler


class ToolRegistry:
    def __init__(self, tools: list[Tool]) -> None:
        self._tools = {tool.spec.name: tool for tool in tools}

    def specs(self) -> list[ToolSpec]:
        return [tool.spec for tool in self._tools.values()]

    def has(self, name: str) -> bool:
        return name in self._tools

    async def invoke(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        tool = self._tools.get(name)
        if tool is None:
            return {
                "ok": False,
                "error": f"Unknown tool '{name}'. Available: {', '.join(sorted(self._tools))}",
            }
        try:
            return await tool.handler(**arguments)
        except TypeError as exc:
            return {"ok": False, "error": f"Bad arguments for '{name}': {exc}"}
        except Exception as exc:  # noqa: BLE001 — a tool error is data, not a crash
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def build_tool_registry(
    org_id: uuid.UUID,
    user_id: uuid.UUID,
    backend: str = "polars",
    *,
    run_id: uuid.UUID | None = None,
    thread_id: uuid.UUID | None = None,
) -> ToolRegistry:
    master = AgentMasterFactory(backend)
    store = HandleStore(org_id, user_id, backend)
    memory = ContextStore(org_id, user_id)

    async def list_data_sources() -> dict[str, Any]:
        async with user_session(user_id) as db:
            rows = (
                await db.execute(
                    text(
                        "select id, name, kind, table_name, row_count, status "
                        "from public.data_sources order by created_at"
                    )
                )
            ).mappings().all()
        return {
            "ok": True,
            "data": {
                "sources": [
                    {
                        "id": str(row["id"]),
                        "name": row["name"],
                        "kind": row["kind"],
                        "table": row["table_name"],
                        "rows": row["row_count"],
                        "status": row["status"],
                    }
                    for row in rows
                ]
            },
        }

    async def read_source(source_id: str) -> dict[str, Any]:
        async with user_session(user_id) as db:
            row = (
                await db.execute(
                    text(
                        "select name, object_path from public.data_sources where id = :id"
                    ),
                    {"id": uuid.UUID(source_id)},
                )
            ).mappings().first()

        if row is None:
            return {"ok": False, "error": f"No data source with id {source_id}"}
        if not row["object_path"]:
            return {
                "ok": False,
                "error": f"Source '{row['name']}' has no uploaded file yet",
            }

        payload = await store._storage.download(row["object_path"])  # noqa: SLF001
        import os
        import tempfile

        directory = tempfile.mkdtemp(prefix="lumen-src-")
        suffix = os.path.splitext(row["object_path"])[1] or ".csv"
        local = os.path.join(directory, f"source{suffix}")
        with open(local, "wb") as file:
            file.write(payload)

        frame = master.readers().read(local)
        handle = await store.put(
            frame, label=row["name"], source_id=uuid.UUID(source_id)
        )
        return {
            "ok": True,
            "data": {
                "rid": handle.rid,
                "row_count": handle.row_count,
                "schema": handle.schema,
            },
        }

    async def profile_source(rid: str) -> dict[str, Any]:
        handle = await store.get(rid)
        frame = await store.resolve(rid)

        rates = null_rates(frame, handle.backend)
        candidates = [
            column
            for column in handle.schema
            if any(hint in column.lower() for hint in DUPLICATE_HINTS)
        ]
        duplicates = duplicate_counts(frame, handle.backend, candidates) if candidates else {}

        notable = {c: r for c, r in rates.items() if r > 0.005}
        dupes = {k: v for k, v in duplicates.items() if v > 0}

        # An org-scoped fact: measured once, useful to every member.
        summary = (
            f"{handle.row_count} rows, {len(handle.schema)} columns. "
            + (
                "Null rates above 0.5%: "
                + ", ".join(f"{c} {r * 100:.1f}%" for c, r in sorted(notable.items()))
                if notable
                else "No column exceeds a 0.5% null rate."
            )
            + (
                " Duplicates: " + ", ".join(f"{c} x{v}" for c, v in sorted(dupes.items()))
                if dupes
                else ""
            )
        )
        await memory.remember(
            ContextEntry(
                kind=Kind.PROFILE,
                title=f"Profile of {handle.label or rid}",
                content=summary,
                rid=rid,
                metadata={"null_rates": rates, "duplicates": dupes},
            )
        )

        return {
            "ok": True,
            "data": {
                "rid": rid,
                "row_count": handle.row_count,
                "columns": handle.schema,
                "null_rate_by_column": rates,
                "duplicate_counts": dupes,
            },
        }

    async def propose_cleaning_pipeline(
        rid: str, steps: list[dict[str, Any]], rationale: str
    ) -> dict[str, Any]:
        """Validate the plan against the engine's factories, then persist it.

        Validation happens before anything is written: an invented step name
        fails here, not as a row a human is asked to review. Once it passes, the
        proposal is a real `proposals` row — awaiting_review — because a plan
        that only ever lived inside the model's response is not something a
        person can accept.
        """
        frame = await store.resolve(rid)
        try:
            PipelineBuilder(frame).build(steps)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"Invalid pipeline: {exc}"}

        proposal_id: uuid.UUID | None = None
        if run_id is not None:
            async with user_session(user_id) as db:
                proposal_id = (
                    await db.execute(
                        text(
                            "insert into public.proposals "
                            "(org_id, run_id, thread_id, author_agent, kind, spec, rationale) "
                            "values (:org, :run, :thread, 'analyst', 'cleaning_pipeline', "
                            "        cast(:spec as jsonb), :rationale) "
                            "returning id"
                        ),
                        {
                            "org": org_id,
                            "run": run_id,
                            "thread": thread_id or run_id,
                            "spec": json.dumps({"rid": rid, "steps": steps}),
                            "rationale": rationale,
                        },
                    )
                ).scalar_one()

        return {
            "ok": True,
            "data": {
                "rid": rid,
                "steps": steps,
                "rationale": rationale,
                "proposal_id": str(proposal_id) if proposal_id else None,
            },
        }

    async def run_statistic(
        rid: str, domain: str, calculator: str, column: str | None = None
    ) -> dict[str, Any]:
        frame = await store.resolve(rid)
        value = master.statistics().run(domain, calculator, frame, column=column)
        return {"ok": True, "data": {"value": jsonable(value)}}

    async def list_predictors(family: str | None = None) -> dict[str, Any]:
        rows = PredictorRegistry.describe()
        if family:
            rows = [r for r in rows if r["family"] == family]
        return {"ok": True, "data": {"predictors": rows}}

    async def compare_predictors(
        rid: str,
        target: str,
        features: list[str] | None = None,
        task: str = "regression",
        candidates: list[str] | None = None,
    ) -> dict[str, Any]:
        frame = await store.resolve(rid)
        comparison = master.prediction().compare(
            frame,
            target=target,
            features=features,
            task=Task(task),
            candidates=candidates,
        )
        best = comparison.best
        return {
            "ok": True,
            "data": {
                "ranked": comparison.summary(),
                "best": best.predictor if best else None,
                # Stated explicitly so the model reports a score measured on data
                # it never saw, rather than a training-set number.
                "scored_on": "a held-out split, not the training data",
            },
        }

    async def forecast_series(
        rid: str,
        column: str,
        horizon: int,
        method: str | None = None,
        order_by: str | None = None,
        season_length: int | None = None,
    ) -> dict[str, Any]:
        frame = await store.resolve(rid)
        prediction = master.prediction()

        chosen = method
        ranked: list[dict[str, Any]] = []
        if chosen is None:
            # Backtest at the horizon actually being forecast. Evaluating at a
            # shorter one selects for a different problem: on a trending seasonal
            # series, scoring at 3 steps picks `naive` and scoring at 7 picks
            # Holt-Winters — and only the second is right for a 7-step forecast.
            #
            # Short series are handled by reducing folds, never the horizon.
            # Each fold consumes `horizon` points, so a series with n points
            # supports at most (n - 2) // horizon of them.
            length = int(np.isfinite(to_series(frame, prediction.backend, column,
                                               order_by=order_by)).sum())
            folds = max(1, min(3, (length - 2) // max(1, horizon)))

            comparison = prediction.compare_forecasters(
                frame,
                column,
                folds=folds,
                horizon=horizon,
                order_by=order_by,
            )
            ranked = comparison.summary()
            if comparison.best is None:
                return {"ok": False, "error": "no forecaster could fit this series"}
            chosen = comparison.best.predictor

        params: dict[str, Any] = {}
        detected = None
        if chosen in ("exponential_smoothing", "seasonal_naive"):
            if season_length is None:
                # Detected rather than defaulted. Running a seasonal method with
                # no season silently drops the seasonal component and lands
                # wrong by roughly its amplitude, which reads as a mediocre
                # forecast rather than as a missing argument.
                detected = detect_season(
                    to_series(frame, prediction.backend, column, order_by=order_by)
                )
                season_length = detected
            if season_length:
                params["season_length"] = season_length

        forecast, report = prediction.forecast(
            frame, chosen, column, horizon, order_by=order_by, **params
        )
        return {
            "ok": True,
            "data": {
                "method": chosen,
                "chosen_by": "backtest" if method is None else "the caller",
                "backtested_at_horizon": horizon if method is None else None,
                "season_length": season_length,
                "season_detected": detected is not None,
                "horizon": horizon,
                "values": [round(v, 4) for v in forecast.as_list()],
                "diagnostics": report.diagnostics,
                "ranked": ranked,
            },
        }

    async def recall_context(query: str, limit: int = 5) -> dict[str, Any]:
        matches = await memory.search(query, limit=limit)
        return {
            "ok": True,
            "data": {
                "matches": [
                    {
                        "kind": str(m.kind),
                        "scope": str(m.scope),
                        "mine": m.is_mine,
                        "title": m.title,
                        "content": m.content,
                        "similarity": round(m.similarity, 3),
                    }
                    for m in matches
                ]
            },
        }

    async def remember_decision(title: str, content: str, source_id: str | None = None) -> dict[str, Any]:
        entry_id = await memory.remember(
            ContextEntry(
                kind=Kind.DECISION,
                scope=Scope.USER,
                title=title,
                content=content,
                source_id=uuid.UUID(source_id) if source_id else None,
            )
        )
        return {"ok": True, "data": {"id": str(entry_id), "scope": "user"}}

    # ── account tools (ADR-0004) ──────────────────────────────────────────

    gate = QuotaGate(org_id, user_id)

    async def get_usage() -> dict[str, Any]:
        statuses = [await gate.status(m) for m in QUOTA_METRICS]
        return {"ok": True, "data": {"period": "current_month", "usage": [s.as_dict() for s in statuses]}}

    async def get_quota_status() -> dict[str, Any]:
        statuses = [await gate.status(m) for m in QUOTA_METRICS]
        return {
            "ok": True,
            "data": {
                "statuses": [s.as_dict() for s in statuses],
                "blocked": [s.metric for s in statuses if s.decision == Decision.DENY],
                "warned": [s.metric for s in statuses if s.decision == Decision.WARN],
            },
        }

    async def explain_cost(run_id_arg: str) -> dict[str, Any]:
        async with user_session(user_id) as db:
            rows = (
                await db.execute(
                    text(
                        "select metric, quantity, cost_micros, agent, occurred_at "
                        "from public.usage_records where run_id = :run and org_id = :org "
                        "order by occurred_at"
                    ),
                    {"run": uuid.UUID(run_id_arg), "org": org_id},
                )
            ).mappings().all()
        if not rows:
            return {"ok": False, "error": f"No usage recorded for run '{run_id_arg}'."}
        return {"ok": True, "data": {"run_id": run_id_arg, "records": jsonable([dict(r) for r in rows])}}

    async def list_members() -> dict[str, Any]:
        async with user_session(user_id) as db:
            rows = (
                await db.execute(
                    text(
                        "select m.user_id, p.email, p.display_name, m.role, m.created_at "
                        "from public.memberships m join public.profiles p on p.id = m.user_id "
                        "where m.org_id = :org order by m.created_at"
                    ),
                    {"org": org_id},
                )
            ).mappings().all()
        return {"ok": True, "data": {"members": jsonable([dict(r) for r in rows])}}

    async def recommend_plan() -> dict[str, Any]:
        async with user_session(user_id) as db:
            current = (
                await db.execute(
                    text("select plan_code from public.organizations where id = :org"), {"org": org_id}
                )
            ).scalar_one()
        statuses = [await gate.status(m) for m in QUOTA_METRICS]
        pressured = [s.metric for s in statuses if s.decision != Decision.ALLOW]
        if not pressured:
            return {
                "ok": True,
                "data": {
                    "current_plan": current,
                    "recommendation": current,
                    "reason": "usage is comfortably within the current plan this period",
                },
            }
        index = PLAN_ORDER.index(current) if current in PLAN_ORDER else 0
        next_plan = PLAN_ORDER[min(index + 1, len(PLAN_ORDER) - 1)]
        return {
            "ok": True,
            "data": {
                "current_plan": current,
                "recommendation": next_plan,
                "reason": (
                    f"{', '.join(m.replace('_', ' ') for m in pressured)} at or above 80% of the "
                    f"'{current}' plan's limit this period"
                ),
            },
        }

    async def suggest_cost_optimisation() -> dict[str, Any]:
        async with user_session(user_id) as db:
            rows = (
                await db.execute(
                    text(
                        "select metric, sum(quantity) as total, count(distinct run_id) as run_count "
                        "from public.usage_records "
                        "where org_id = :org and occurred_at >= date_trunc('month', now()) "
                        "group by metric order by total desc limit 3"
                    ),
                    {"org": org_id},
                )
            ).mappings().all()
        if not rows:
            return {"ok": True, "data": {"suggestions": [], "note": "No usage recorded yet this period."}}
        top = rows[0]
        suggestion = (
            f"{top['metric'].replace('_', ' ')} is the largest cost driver this period — "
            f"{int(top['total'])} across {top['run_count']} run(s)."
        )
        return {
            "ok": True,
            "data": {"top_metric": jsonable(dict(top)), "suggestions": [suggestion]},
        }

    async def propose_plan_change(plan_code: str, rationale: str) -> dict[str, Any]:
        async with user_session(user_id) as db:
            plan = (
                await db.execute(
                    text("select code, price_cents from public.plans where code = :code"),
                    {"code": plan_code},
                )
            ).mappings().first()
            if plan is None:
                return {
                    "ok": False,
                    "error": f"Unknown plan '{plan_code}'. Available: {', '.join(PLAN_ORDER)}.",
                }
            current = (
                await db.execute(
                    text("select plan_code from public.organizations where id = :org"), {"org": org_id}
                )
            ).scalar_one()
            if current == plan_code:
                return {"ok": False, "error": f"This workspace is already on the '{plan_code}' plan."}

        proposal_id: uuid.UUID | None = None
        if run_id is not None:
            async with user_session(user_id) as db:
                proposal_id = (
                    await db.execute(
                        text(
                            "insert into public.proposals "
                            "(org_id, run_id, thread_id, author_agent, kind, spec, rationale) "
                            "values (:org, :run, :thread, 'admin', 'plan_change', "
                            "        cast(:spec as jsonb), :rationale) returning id"
                        ),
                        {
                            "org": org_id,
                            "run": run_id,
                            "thread": thread_id or run_id,
                            "spec": json.dumps({"plan_code": plan_code, "price_cents": plan["price_cents"]}),
                            "rationale": rationale,
                        },
                    )
                ).scalar_one()
        return {
            "ok": True,
            "data": {
                "plan_code": plan_code,
                "proposal_id": str(proposal_id) if proposal_id else None,
            },
        }

    async def propose_member_role_change(target_user_id: str, role: str) -> dict[str, Any]:
        valid_roles = {"owner", "admin", "member", "viewer"}
        if role not in valid_roles:
            return {"ok": False, "error": f"'{role}' is not a role. Use one of: {', '.join(sorted(valid_roles))}."}

        async with user_session(user_id) as db:
            member = (
                await db.execute(
                    text(
                        "select p.display_name, m.role from public.memberships m "
                        "join public.profiles p on p.id = m.user_id "
                        "where m.org_id = :org and m.user_id = :target"
                    ),
                    {"org": org_id, "target": uuid.UUID(target_user_id)},
                )
            ).mappings().first()
        if member is None:
            return {"ok": False, "error": "No member with that user id in this workspace."}
        if str(member["role"]) == role:
            return {"ok": False, "error": f"{member['display_name']} already has the '{role}' role."}

        proposal_id: uuid.UUID | None = None
        if run_id is not None:
            async with user_session(user_id) as db:
                proposal_id = (
                    await db.execute(
                        text(
                            "insert into public.proposals "
                            "(org_id, run_id, thread_id, author_agent, kind, spec, rationale) "
                            "values (:org, :run, :thread, 'admin', 'member_role_change', "
                            "        cast(:spec as jsonb), :rationale) returning id"
                        ),
                        {
                            "org": org_id,
                            "run": run_id,
                            "thread": thread_id or run_id,
                            "spec": json.dumps(
                                {"user_id": target_user_id, "role": role, "previous_role": str(member["role"])}
                            ),
                            "rationale": f"Change {member['display_name']}'s role from "
                            f"{member['role']} to {role}.",
                        },
                    )
                ).scalar_one()
        return {
            "ok": True,
            "data": {
                "user_id": target_user_id,
                "role": role,
                "proposal_id": str(proposal_id) if proposal_id else None,
            },
        }

    tools = [
        Tool(
            ToolSpec(
                name="list_data_sources",
                description=(
                    "List every data source in this workspace with its table, row count "
                    "and status."
                ),
                input_schema={"type": "object", "properties": {}, "required": []},
            ),
            list_data_sources,
        ),
        Tool(
            ToolSpec(
                name="read_source",
                description=(
                    "Load a data source into a working dataset. Returns a handle id (rid) "
                    "and the schema. Call this before profiling."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "source_id": {"type": "string", "description": "Data source uuid"}
                    },
                    "required": ["source_id"],
                },
            ),
            read_source,
        ),
        Tool(
            ToolSpec(
                name="profile_source",
                description=(
                    "Return row count, column types, per-column null rate and duplicate "
                    "counts for key-like columns of a loaded dataset."
                ),
                input_schema={
                    "type": "object",
                    "properties": {"rid": {"type": "string", "description": "Dataset handle id"}},
                    "required": ["rid"],
                },
            ),
            profile_source,
        ),
        Tool(
            ToolSpec(
                name="propose_cleaning_pipeline",
                description=_pipeline_tool_description(backend),
                input_schema={
                    "type": "object",
                    "properties": {
                        "rid": {"type": "string"},
                        "steps": {"type": "array", "items": {"type": "object"}},
                        "rationale": {
                            "type": "string",
                            "description": "Why, citing the numbers you observed.",
                        },
                    },
                    "required": ["rid", "steps", "rationale"],
                },
            ),
            propose_cleaning_pipeline,
        ),
        Tool(
            ToolSpec(
                name="run_statistic",
                description=(
                    "Run one registered statistic, e.g. domain='descriptive', "
                    "calculator='central_tendency_calculator'."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "rid": {"type": "string"},
                        "domain": {"type": "string"},
                        "calculator": {"type": "string"},
                        "column": {"type": "string"},
                    },
                    "required": ["rid", "domain", "calculator"],
                },
            ),
            run_statistic,
        ),
        Tool(
            ToolSpec(
                name="recall_context",
                description=(
                    "Search what is already known about this workspace's data, and what "
                    "this particular person decided before. Call it BEFORE profiling: a "
                    "source someone already audited does not need auditing again, and a "
                    "proposal this person rejected should not be proposed again unchanged. "
                    "Results marked mine=true are this user's own history; the rest are "
                    "shared facts about the data."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "What you want to remember, in natural language.",
                        },
                        "limit": {"type": "integer"},
                    },
                    "required": ["query"],
                },
            ),
            recall_context,
        ),
        Tool(
            ToolSpec(
                name="remember_decision",
                description=(
                    "Record something this person decided or prefers, so future runs "
                    "start from it. Private to them. Use it when they reject a proposal, "
                    "correct you, or state a preference — not for facts about the data, "
                    "which are recorded automatically and shared with their team."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "content": {
                            "type": "string",
                            "description": "What they decided and why, in their terms.",
                        },
                        "source_id": {"type": "string"},
                    },
                    "required": ["title", "content"],
                },
            ),
            remember_decision,
        ),
        Tool(
            ToolSpec(
                name="list_predictors",
                description=_predictor_catalogue_description(),
                input_schema={
                    "type": "object",
                    "properties": {
                        "family": {
                            "type": "string",
                            "enum": ["numerical", "timeseries", "ml"],
                        }
                    },
                    "required": [],
                },
            ),
            list_predictors,
        ),
        Tool(
            ToolSpec(
                name="compare_predictors",
                description=(
                    "Fit several predictors on a held-out split and rank them. Prefer this "
                    "to picking one method and reporting its score: a number without a "
                    "holdout is not evidence, and a model's training error says nothing "
                    "about whether it will work. Returns every candidate with its metrics, "
                    "best first."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "rid": {"type": "string"},
                        "target": {"type": "string", "description": "Column to predict."},
                        "features": {"type": "array", "items": {"type": "string"}},
                        "task": {
                            "type": "string",
                            "enum": ["regression", "classification"],
                        },
                        "candidates": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["rid", "target"],
                },
            ),
            compare_predictors,
        ),
        Tool(
            ToolSpec(
                name="forecast_series",
                description=(
                    "Forecast a column forward. Omit `method` and it is chosen by "
                    "rolling-origin backtest across every forecaster including the naive "
                    "baseline — the honest way to choose, because a forecast that cannot "
                    "beat 'tomorrow equals today' has not earned its complexity. Pass "
                    "`order_by` with the date column: every method treats row order as "
                    "time order, and a parquet scan does not guarantee it."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "rid": {"type": "string"},
                        "column": {"type": "string", "description": "The series to forecast."},
                        "horizon": {"type": "integer", "description": "Steps ahead."},
                        "method": {"type": "string"},
                        "order_by": {"type": "string", "description": "Date or index column."},
                        "season_length": {
                            "type": "integer",
                            "description": "e.g. 7 for daily data with a weekly cycle.",
                        },
                    },
                    "required": ["rid", "column", "horizon"],
                },
            ),
            forecast_series,
        ),
        Tool(
            ToolSpec(
                name="get_usage",
                description=(
                    "This workspace's usage against its plan limits for the current calendar "
                    "month: LLM input/output tokens and agent runs."
                ),
                input_schema={"type": "object", "properties": {}, "required": []},
            ),
            get_usage,
        ),
        Tool(
            ToolSpec(
                name="get_quota_status",
                description=(
                    "Whether any metric is at or near its limit right now — call this before "
                    "telling someone a run failed or was refused, to say why."
                ),
                input_schema={"type": "object", "properties": {}, "required": []},
            ),
            get_quota_status,
        ),
        Tool(
            ToolSpec(
                name="explain_cost",
                description="Itemized usage records for one run, to answer 'why did this cost what it cost'.",
                input_schema={
                    "type": "object",
                    "properties": {"run_id_arg": {"type": "string", "description": "The run's id."}},
                    "required": ["run_id_arg"],
                },
            ),
            explain_cost,
        ),
        Tool(
            ToolSpec(
                name="list_members",
                description="Everyone in this workspace and their role.",
                input_schema={"type": "object", "properties": {}, "required": []},
            ),
            list_members,
        ),
        Tool(
            ToolSpec(
                name="recommend_plan",
                description=(
                    "Whether this workspace's current usage suggests upgrading, based on how "
                    "close it is to its plan's limits this period."
                ),
                input_schema={"type": "object", "properties": {}, "required": []},
            ),
            recommend_plan,
        ),
        Tool(
            ToolSpec(
                name="suggest_cost_optimisation",
                description="The largest cost driver this period, as a starting point for reducing it.",
                input_schema={"type": "object", "properties": {}, "required": []},
            ),
            suggest_cost_optimisation,
        ),
        Tool(
            ToolSpec(
                name="propose_plan_change",
                description=(
                    "Propose changing this workspace's plan. A person with the owner role must "
                    "accept it — this never changes billing on its own."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "plan_code": {"type": "string", "enum": list(PLAN_ORDER)},
                        "rationale": {"type": "string"},
                    },
                    "required": ["plan_code", "rationale"],
                },
            ),
            propose_plan_change,
        ),
        Tool(
            ToolSpec(
                name="propose_member_role_change",
                description=(
                    "Propose changing a member's role. A person with the owner role must accept "
                    "it — this never changes a permission on its own."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "target_user_id": {"type": "string"},
                        "role": {"type": "string", "enum": ["owner", "admin", "member", "viewer"]},
                    },
                    "required": ["target_user_id", "role"],
                },
            ),
            propose_member_role_change,
        ),
    ]
    return ToolRegistry(tools)


def registered_steps(backend: str) -> list[str]:
    """Step names this backend actually implements.

    Read from the factory rather than written down, so the tool description can
    never drift from what the engine accepts — the drift that made an agent
    propose `drop_nulls`, a name nothing has ever registered.
    """
    return sorted(
        key
        for key in AbstractDataCleaningStepFactory.registered_keys()
        if AbstractDataCleaningStepFactory.is_registered(key, backend)
    )


def _pipeline_tool_description(backend: str) -> str:
    steps = registered_steps(backend)
    return (
        "Validate an ordered cleaning pipeline. Each step is an object with exactly "
        "one key — the step name — mapped to its keyword arguments, e.g. "
        '[{"remove_duplicates_rows": {}}, '
        '{"impute_categorical": {"columns": ["country_code"], "strategy": "mode"}}]. '
        "Executes nothing; it only checks the plan builds.\n\n"
        "The ONLY valid step names are: " + ", ".join(steps) + ". "
        "Do not invent others — an unknown name is rejected."
    )


def _predictor_catalogue_description() -> str:
    """Generated from the registry, for the reason the cleaning vocabulary is:
    a model that has to guess a method name eventually guesses one that has never
    existed, and then a human is asked to approve it."""
    by_family: dict[str, list[str]] = {}
    for row in PredictorRegistry.describe():
        by_family.setdefault(row["family"], []).append(row["name"])
    families = "; ".join(
        f"{family}: {', '.join(sorted(names))}"
        for family, names in sorted(by_family.items())
    )
    return (
        "List every available prediction method with its parameters and what it is "
        "for. Call this before compare_predictors or forecast_series so you use a "
        "method that exists.\n\nAvailable — " + families
    )


