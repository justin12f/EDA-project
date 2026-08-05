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
from lumen_api.context.store import ContextEntry, ContextStore, Kind, Scope
from lumen_api.datasets.store import HandleStore
from lumen_api.db.session import user_session

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
    org_id: uuid.UUID, user_id: uuid.UUID, backend: str = "polars"
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
        """Validate the plan against the engine's factories. Nothing runs here."""
        frame = await store.resolve(rid)
        try:
            PipelineBuilder(frame).build(steps)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"Invalid pipeline: {exc}"}
        return {"ok": True, "data": {"rid": rid, "steps": steps, "rationale": rationale}}

    async def run_statistic(
        rid: str, domain: str, calculator: str, column: str | None = None
    ) -> dict[str, Any]:
        frame = await store.resolve(rid)
        value = master.statistics().run(domain, calculator, frame, column=column)
        return {"ok": True, "data": {"value": _jsonable(value)}}

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


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)
