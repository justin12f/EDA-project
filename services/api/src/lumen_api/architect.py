"""The Data Architect — design, propose, apply.

Deterministic by default. Types, keys and relationships come from
statistics the engine already computes; the model's only job is to make the
names readable and write the rationale prose (Task 16), and it is allowed to
fail without stopping anything. That split is the same one ADR-0008 states
for detection and ADR-0013 for calibration, and it is what keeps the
keyless MockProvider path working end to end.
"""

from __future__ import annotations

import uuid
from typing import Any

import polars as pl
from sqlalchemy import text

from lumen.architect.ddl import sanitize_identifier
from lumen.architect.infer import detect_foreign_keys, infer_sql_type, select_primary_key
from lumen.architect.spec import ColumnSpec, SchemaSpec, TableSpec
from lumen.datasets.materialize import frame_schema
from lumen_api.db.session import user_session
from lumen_api.settings import get_settings
from lumen_api.tenant_db import tenant_raw_schema_name, tenant_schema_name

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
