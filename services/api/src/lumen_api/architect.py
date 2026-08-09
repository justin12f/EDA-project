"""The Data Architect — design, propose, apply.

Deterministic by default. Types, keys and relationships come from
statistics the engine already computes; the model's only job is to make the
names readable and write the rationale prose (Task 16), and it is allowed to
fail without stopping anything. That split is the same one ADR-0008 states
for detection and ADR-0013 for calibration, and it is what keeps the
keyless MockProvider path working end to end.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import replace
from typing import Any

import polars as pl
from sqlalchemy import text

from lumen.architect.adapters.postgres import PostgresAdapter
from lumen.architect.ddl import render_ddl, sanitize_identifier
from lumen.architect.spec import (
    ColumnSpec,
    Evidence,
    ForeignKeySpec,
    SchemaSpec,
    SqlType,
    TableSpec,
)
from lumen.architect.infer import detect_foreign_keys, infer_sql_type, select_primary_key
from lumen.datasets.materialize import frame_schema
from lumen.llm.base import ChatMessage
from lumen_api.db.session import user_session
from lumen_api.lineage import record_schema_table_dependency
from lumen_api.llm import provider
from lumen_api.settings import get_settings
from lumen_api.tenant_db import (
    ensure_tenant_schema,
    tenant_raw_schema_name,
    tenant_schema_name,
    tenant_session,
)

BACKEND = "polars"


async def _semantic_pairs(org_id: uuid.UUID, user_id: uuid.UUID) -> list[tuple[str, str, str, str]]:
    """ADR-0009's canonical entities, shaped as FK candidates.

    A canonical entity says "these columns across sources are the same
    business concept" — which is exactly the hint that one of them
    references the other. The engine never learns what a canonical entity
    is; this function is the whole translation.
    """
    async with user_session(user_id) as db:
        rows = (
            await db.execute(
                text(
                    "select s.table_name as table_name, m.column_name as column_name "
                    "from public.canonical_entity_members m "
                    "join public.canonical_entities e on e.id = m.entity_id "
                    "join public.data_sources s on s.id = m.source_id "
                    "where e.org_id = :org and e.status = 'approved' "
                    "  and s.table_name is not null"
                ),
                {"org": org_id},
            )
        ).mappings().all()

    members = [(r["table_name"], r["column_name"]) for r in rows]
    pairs: list[tuple[str, str, str, str]] = []
    for child_table, child_column in members:
        for parent_table, parent_column in members:
            if child_table != parent_table:
                pairs.append((child_table, child_column, parent_table, parent_column))
    return pairs


async def _staged_tables(org_id: uuid.UUID) -> dict[str, pl.DataFrame]:
    """Every table currently in this org's staging schema.

    Read wholesale because FK detection is inherently cross-table — a
    relationship cannot be found by looking at one source in isolation.
    """
    dsn = get_settings().tenant_database_url.get_secret_value()
    raw = tenant_raw_schema_name(org_id)

    names = pl.read_database_uri(
        query=(
            "select table_name from information_schema.tables "
            f"where table_schema = '{raw}'"
        ),
        uri=dsn,
    )
    return {
        name: pl.read_database_uri(query=f'select * from "{raw}"."{name}"', uri=dsn)
        for name in names["table_name"].to_list()
    }


async def design_schema(
    org_id: uuid.UUID, user_id: uuid.UUID, source_id: uuid.UUID
) -> SchemaSpec:
    """Design the modelled schema for this org, including `source_id`.

    Returns the whole org's spec rather than one table's, because a foreign
    key is a statement about two tables and cannot be designed from one.
    """
    async with user_session(user_id) as db:
        sources = (
            await db.execute(
                text(
                    "select id, table_name from public.data_sources "
                    "where org_id = :org and table_name is not null"
                ),
                {"org": org_id},
            )
        ).mappings().all()
    source_of = {r["table_name"]: r["id"] for r in sources}

    frames = await _staged_tables(org_id)

    tables: list[TableSpec] = []
    taken_tables: set[str] = set()
    for raw_name, frame in sorted(frames.items()):
        table_name = sanitize_identifier(raw_name, taken=taken_tables)
        taken_tables.add(table_name)

        taken_columns: set[str] = set()
        columns: list[ColumnSpec] = []
        for source_column, dtype in frame_schema(frame, BACKEND).items():
            name = sanitize_identifier(source_column, taken=taken_columns)
            taken_columns.add(name)
            sql_type, type_arg = infer_sql_type(dtype)
            columns.append(
                ColumnSpec(
                    name=name,
                    source_column=source_column,
                    sql_type=sql_type,
                    type_arg=type_arg,
                )
            )

        # PK selection reads the frame under its original column names; the
        # sanitised name is what goes into the spec.
        renamed = {c.source_column: c.name for c in columns}
        key, rationale = select_primary_key(frame, BACKEND, list(renamed))
        tables.append(
            TableSpec(
                name=table_name,
                source_id=source_of.get(raw_name, source_id),
                columns=tuple(columns),
                primary_key=tuple(renamed[k] for k in key) if key else None,
                pk_rationale=rationale,
                source_table=raw_name,
            )
        )

    table_tuple = tuple(tables)
    renamed_frames = {
        table.name: frames[table.source_table].rename(
            {c.source_column: c.name for c in table.columns}
        )
        for table in table_tuple
    }

    keys = detect_foreign_keys(
        renamed_frames,
        BACKEND,
        table_tuple,
        semantic_pairs=await _semantic_pairs(org_id, user_id),
    )

    spec = SchemaSpec(tables=table_tuple, foreign_keys=tuple(keys))
    spec.validate()
    return spec


_ENRICH_PROMPT = """\
You are given a database schema that was designed deterministically from a \
customer's data. Improve only the human-facing prose.

Return JSON of the form:
  {"tables": {"<table>": {"pk_rationale": "<one clear sentence>"}}}

Rules:
- Do not rename anything. Do not add or remove tables or columns.
- Do not mention types, thresholds, or percentages.
- Write for a business user reading a schema for the first time.

Schema:
"""


async def enrich_spec(spec: SchemaSpec) -> SchemaSpec:
    """Better prose from the model, or the spec exactly as it came in.

    Deliberately impossible to make load-bearing. Every failure mode —
    provider down, quota denied, MockProvider running, malformed JSON,
    a model trying to rename a table — returns the deterministic spec.
    That is what lets a deployment with no API key still produce a real
    database, and it is why the model is not on the critical path.
    """
    summary = {
        table.name: {
            "columns": [c.name for c in table.columns],
            "primary_key": list(table.primary_key or ()),
        }
        for table in spec.tables
    }

    try:
        response = await provider(tier="fast").complete(
            [ChatMessage(role="user", content=_ENRICH_PROMPT + json.dumps(summary, indent=2))],
            [],
        )
        payload = json.loads(response.text)
        improvements = payload["tables"]
    except Exception:  # noqa: BLE001 — every failure is the same non-event
        return spec

    tables = tuple(
        replace(table, pk_rationale=str(improvements[table.name]["pk_rationale"]))
        if isinstance(improvements.get(table.name), dict)
        and improvements[table.name].get("pk_rationale")
        else table
        for table in spec.tables
    )
    # Structure is never taken from the model — only the prose field is
    # substituted, on tables that already existed.
    return replace(spec, tables=tables)


async def propose_schema(
    org_id: uuid.UUID,
    user_id: uuid.UUID,
    source_id: uuid.UUID,
    spec: SchemaSpec,
    kind: str = "schema_design",
) -> uuid.UUID:
    """Write the spec as a Proposal for a human to accept.

    Non-negotiable #2: every mutating agent action produces one of these.
    Creating a customer's database is unambiguously mutating.
    """
    spec.validate()

    async with user_session(user_id) as db:
        run_id = (
            await db.execute(
                text(
                    "insert into public.runs "
                    "(org_id, source_id, thread_id, kind, status, backend, created_by) "
                    "values (:org, :source, gen_random_uuid(), 'architect', 'succeeded', "
                    "        'polars', :user) returning id"
                ),
                {"org": org_id, "source": source_id, "user": user_id},
            )
        ).scalar_one()

        proposal_id = (
            await db.execute(
                text(
                    "insert into public.proposals "
                    "(org_id, run_id, thread_id, author_agent, kind, spec, rationale) "
                    "values (:org, :run, :run, 'architect', :kind, cast(:spec as jsonb), "
                    "        :rationale) returning id"
                ),
                {
                    "org": org_id,
                    "run": run_id,
                    "kind": kind,
                    "spec": json.dumps(_spec_to_json(spec)),
                    "rationale": _describe(spec),
                },
            )
        ).scalar_one()
    return proposal_id


def _spec_to_json(spec: SchemaSpec) -> dict[str, Any]:
    return {
        "layout": spec.layout,
        "tables": [
            {
                "name": t.name,
                "source_id": str(t.source_id),
                "source_table": t.source_table,
                "primary_key": list(t.primary_key or ()),
                "pk_rationale": t.pk_rationale,
                "columns": [
                    {
                        "name": c.name,
                        "source_column": c.source_column,
                        "sql_type": c.sql_type.value,
                        "type_arg": c.type_arg,
                        "nullable": c.nullable,
                        "deprecated": c.deprecated,
                    }
                    for c in t.columns
                ],
            }
            for t in spec.tables
        ],
        "foreign_keys": [
            {
                "from_table": k.from_table,
                "from_column": k.from_column,
                "to_table": k.to_table,
                "to_column": k.to_column,
                "containment": k.containment,
                "enforced": k.enforced,
                "evidence": [e.value for e in k.evidence],
                "rationale": k.rationale,
            }
            for k in spec.foreign_keys
        ],
    }


def _describe(spec: SchemaSpec) -> str:
    enforced = sum(1 for k in spec.foreign_keys if k.enforced)
    observed = len(spec.foreign_keys) - enforced
    parts = [f"{len(spec.tables)} table{'s' if len(spec.tables) != 1 else ''}"]
    if enforced:
        parts.append(f"{enforced} enforced relationship{'s' if enforced != 1 else ''}")
    if observed:
        parts.append(f"{observed} observed relationship{'s' if observed != 1 else ''}")
    return "A database with " + ", ".join(parts) + "."


def _spec_from_json(payload: dict[str, Any]) -> SchemaSpec:
    tables = tuple(
        TableSpec(
            name=t["name"],
            source_id=uuid.UUID(t["source_id"]),
            source_table=t.get("source_table"),
            primary_key=tuple(t["primary_key"]) or None,
            pk_rationale=t.get("pk_rationale", ""),
            columns=tuple(
                ColumnSpec(
                    name=c["name"],
                    source_column=c["source_column"],
                    sql_type=SqlType(c["sql_type"]),
                    type_arg=c.get("type_arg"),
                    nullable=c.get("nullable", True),
                    deprecated=c.get("deprecated", False),
                )
                for c in t["columns"]
            ),
        )
        for t in payload["tables"]
    )
    keys = tuple(
        ForeignKeySpec(
            from_table=k["from_table"],
            from_column=k["from_column"],
            to_table=k["to_table"],
            to_column=k["to_column"],
            containment=k["containment"],
            enforced=k["enforced"],
            evidence=tuple(Evidence(e) for e in k["evidence"]),
            rationale=k.get("rationale", ""),
        )
        for k in payload.get("foreign_keys", [])
    )
    return SchemaSpec(tables=tables, foreign_keys=keys, layout=payload.get("layout", "merged"))


async def apply_schema(
    org_id: uuid.UUID, user_id: uuid.UUID, spec: SchemaSpec
) -> dict[str, Any]:
    """Create the schema, load it from staging, then record it.

    Two sessions, in this order on purpose (spec §3.2). No transaction spans
    the instances, so one of them must go first — and writing the
    control-plane record LAST means a crash leaves an applied schema with a
    proposal still marked accepted, never a record of a schema that does not
    exist. Reconciliation is a re-run of discovery, which is authoritative.
    """
    await ensure_tenant_schema(org_id)
    schema = tenant_schema_name(org_id)
    raw = tenant_raw_schema_name(org_id)

    loaded: dict[str, int] = {}
    async with tenant_session(org_id) as db:
        for statement in render_ddl(spec, schema):
            await db.execute(text(statement))

        await db.execute(text("SET CONSTRAINTS ALL DEFERRED"))
        for table in spec.tables:
            columns = ", ".join(f'"{c.name}"' for c in table.columns)
            source = ", ".join(f'"{c.source_column}"' for c in table.columns)
            await db.execute(text(f'DELETE FROM "{schema}"."{table.name}"'))
            await db.execute(
                text(
                    f'INSERT INTO "{schema}"."{table.name}" ({columns}) '
                    f'SELECT {source} FROM "{raw}"."{table.source_table}"'
                )
            )
            loaded[table.name] = (
                await db.execute(text(f'SELECT count(*) FROM "{schema}"."{table.name}"'))
            ).scalar_one()

    async with user_session(user_id) as db:
        for table in spec.tables:
            await db.execute(
                text(
                    "update public.data_sources set table_name = :table, row_count = :rows "
                    "where id = :id"
                ),
                {"table": table.name, "rows": loaded[table.name], "id": table.source_id},
            )
            await record_schema_table_dependency(
                db, org_id, table.source_id, [c.source_column for c in table.columns],
            )

    return {"tables": list(loaded), "rows": loaded, "schema": schema}


async def current_spec(org_id: uuid.UUID, user_id: uuid.UUID) -> SchemaSpec:
    """The schema currently applied on the tenant instance, as a SchemaSpec.

    The same assembly `PostgresAdapter.discover()` performs against a
    customer's own database, pointed at our own tenant schema instead — to
    the adapter, a customer database and ours are the same shape. Every
    discovered column, key and relationship is DECLARED evidence, because
    it was read from the database's own information_schema, not inferred.
    """
    dsn = get_settings().tenant_database_url.get_secret_value()
    structure = await PostgresAdapter(dsn, tenant_schema_name(org_id)).discover()

    async with user_session(user_id) as db:
        sources = (
            await db.execute(
                text(
                    "select id, table_name from public.data_sources "
                    "where org_id = :org and table_name is not null"
                ),
                {"org": org_id},
            )
        ).mappings().all()
    source_of = {r["table_name"]: r["id"] for r in sources}

    tables = tuple(
        TableSpec(
            name=table.name,
            source_id=source_of.get(table.name, uuid.uuid4()),
            source_table=table.name,
            primary_key=table.primary_key,
            pk_rationale="",
            columns=tuple(
                ColumnSpec(
                    name=column.name,
                    source_column=column.name,
                    sql_type=column.sql_type,
                    type_arg=column.type_arg,
                    nullable=column.nullable,
                )
                for column in table.columns
            ),
        )
        for table in structure.tables
    )

    keys = tuple(
        ForeignKeySpec(
            from_table=table.name,
            from_column=from_column,
            to_table=to_table,
            to_column=to_column,
            containment=1.0,
            enforced=True,
            evidence=(Evidence.DECLARED,),
            rationale="Declared by the currently applied schema.",
        )
        for table in structure.tables
        for from_column, to_table, to_column in table.foreign_keys
    )

    return SchemaSpec(tables=tables, foreign_keys=keys)
