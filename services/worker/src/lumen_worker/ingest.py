"""Ingestion jobs.

Two of them, and the split matters: staging lands data so a customer can
see it immediately (D4), and design proposes a schema. Keeping them apart
means a quota-denied or provider-down design never blocks a customer from
looking at the file they just uploaded.
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from typing import Any

from sqlalchemy import text

from lumen.architect.adapters.file import FileAdapter
from lumen.architect.migrate import classify_migration, render_migration
from lumen.readers.exceptions import ReaderError
from lumen_api.architect import current_spec, design_schema, enrich_spec, propose_schema
from lumen_api.datasets.store import SupabaseStorage
from lumen_api.db.session import user_session
from lumen_api.settings import get_settings
from lumen_api.tenant_db import (
    ensure_tenant_schema,
    raw_table_name,
    tenant_raw_schema_name,
    tenant_schema_name,
    tenant_session,
)
from lumen_api.trust import is_auto_apply_eligible, structural_shape

SUPPORTED = (".csv", ".parquet", ".json", ".xlsx", ".xls")


async def _mark_error(user_id: uuid.UUID, source_id: uuid.UUID) -> None:
    async with user_session(user_id) as db:
        await db.execute(
            text("update public.data_sources set status = 'error' where id = :id"),
            {"id": source_id},
        )


async def ingest_to_staging(
    ctx: dict[str, Any], source_id: str, org_id: str, acting_user_id: str
) -> dict[str, Any]:
    """Load a source into `tenant_<hex>_raw`, then enqueue its design.

    Staging is a permanent landing zone, not a temporary buffer: the
    raw-data browser reads it while a schema is awaiting review, and a
    failed promotion must be retryable without re-downloading the origin.
    """
    source_uuid, org_uuid, user_uuid = (
        uuid.UUID(source_id), uuid.UUID(org_id), uuid.UUID(acting_user_id)
    )

    async with user_session(user_uuid) as db:
        source = (
            await db.execute(
                text("select name, object_path from public.data_sources where id = :id"),
                {"id": source_uuid},
            )
        ).mappings().first()

    if source is None or not source["object_path"]:
        return {"status": "skipped", "reason": "source has no uploaded file yet"}

    suffix = os.path.splitext(source["object_path"])[1].lower()
    if suffix not in SUPPORTED:
        await _mark_error(user_uuid, source_uuid)
        return {
            "status": "error",
            "reason": f"{suffix or 'this file'} is not a supported format; "
                      f"supported: {', '.join(SUPPORTED)}",
        }

    await ensure_tenant_schema(org_uuid)

    payload = await SupabaseStorage().download(source["object_path"])
    directory = tempfile.mkdtemp(prefix=f"lumen-ingest-{org_uuid.hex}-")
    local = os.path.join(directory, f"source{suffix}")
    with open(local, "wb") as file:
        file.write(payload)

    adapter = FileAdapter(local)
    try:
        frame = await adapter.read(adapter.table_name)
    except ReaderError:
        await _mark_error(user_uuid, source_uuid)
        return {"status": "error", "reason": "the file could not be read"}

    materialised = frame.collect() if hasattr(frame, "collect") else frame
    table = os.path.splitext(source["name"])[0]
    # Namespaced by source id, not by alias alone: two sources named
    # "users.csv" would otherwise write to the exact same physical raw
    # table and silently overwrite each other before design ever ran.
    raw_table = raw_table_name(source_uuid, table)

    materialised.write_database(
        table_name=f"{tenant_raw_schema_name(org_uuid)}.{raw_table}",
        connection=get_settings().tenant_database_url.get_secret_value(),
        if_table_exists="replace",
    )

    await ctx["redis"].enqueue_job("design_schema_job", source_id, org_id, acting_user_id)
    return {"status": "staged", "table": table, "rows": materialised.height}


async def design_schema_job(
    ctx: dict[str, Any], source_id: str, org_id: str, acting_user_id: str
) -> dict[str, Any]:
    """Design a schema over everything in staging and propose it.

    Creates nothing in the modelled schema — that happens only when a human
    accepts the proposal (D4).
    """
    source_uuid, org_uuid, user_uuid = (
        uuid.UUID(source_id), uuid.UUID(org_id), uuid.UUID(acting_user_id)
    )

    spec = await design_schema(org_uuid, user_uuid, source_uuid)
    spec = await enrich_spec(spec)
    proposal_id = await propose_schema(org_uuid, user_uuid, source_uuid, spec)

    return {"status": "proposed", "proposal_id": str(proposal_id), "tables": len(spec.tables)}


async def refresh_source(
    ctx: dict[str, Any], source_id: str, org_id: str, acting_user_id: str
) -> dict[str, Any]:
    """Reload a source whose file changed, replacing the table's contents.

    Full replacement, not upsert: a file is a snapshot of the whole source,
    so a row the customer deleted at origin must disappear here. An upsert
    would keep it alive forever with nobody able to say why.

    Everything runs in one transaction with constraints deferred, so a
    parent can be cleared while a child still references it. A violation at
    COMMIT means the new snapshot genuinely breaks a relationship the data
    used to satisfy — that is a finding, not a crash (D5), so the
    transaction rolls back and a DriftEvent records it.
    """
    source_uuid, org_uuid, user_uuid = (
        uuid.UUID(source_id), uuid.UUID(org_id), uuid.UUID(acting_user_id)
    )

    staged = await ingest_to_staging(ctx, source_id, org_id, acting_user_id)
    if staged["status"] != "staged":
        return staged

    async with user_session(user_uuid) as db:
        row = (
            await db.execute(
                text("select name, table_name from public.data_sources where id = :id"),
                {"id": source_uuid},
            )
        ).mappings().first()
    table = row["table_name"] if row else None
    if not table:
        # Never promoted — the data is in staging and that is correct.
        return {"status": "staged_only", "reason": "this source has no approved schema yet"}

    # The raw table this source landed in is named by its own id, not by
    # `table` — `table` is data_sources.table_name, the *modelled* name a
    # collision may have qualified (e.g. "crm__users"); the raw name never
    # changes once staged.
    alias = os.path.splitext(row["name"])[0]
    raw_table = raw_table_name(source_uuid, alias)

    schema = tenant_schema_name(org_uuid)
    raw = tenant_raw_schema_name(org_uuid)

    try:
        async with tenant_session(org_uuid) as db:
            await db.execute(text("SET CONSTRAINTS ALL DEFERRED"))
            await db.execute(text(f'DELETE FROM "{schema}"."{table}"'))
            await db.execute(
                text(f'INSERT INTO "{schema}"."{table}" SELECT * FROM "{raw}"."{raw_table}"')
            )
    except Exception as exc:  # noqa: BLE001 — see the docstring
        async with user_session(user_uuid) as db:
            await db.execute(
                text(
                    "insert into public.drift_events "
                    "(org_id, source_id, kind, severity, details) "
                    "values (:org, :source, 'schema_constraint', 0.8, cast(:details as jsonb))"
                ),
                {
                    "org": org_uuid,
                    "source": source_uuid,
                    "details": json.dumps({"error": str(exc), "table": table}),
                },
            )
        return {"status": "constraint_violation", "table": table}

    return {"status": "refreshed", "table": table}


async def evolve_schema(
    ctx: dict[str, Any], source_id: str, org_id: str, acting_user_id: str
) -> dict[str, Any]:
    """Compare a re-staged source against its applied schema and migrate.

    Two gates, and only one of them is negotiable. A reversible plan may
    auto-apply if this org has earned trust in that shape (ADR-0011); a
    plan with any irreversible step is always a proposal, at any trust
    level, because ADR-0017 §3 makes irreversibility a ceiling no learned
    signal may raise.
    """
    source_uuid, org_uuid, user_uuid = (
        uuid.UUID(source_id), uuid.UUID(org_id), uuid.UUID(acting_user_id)
    )

    staged = await ingest_to_staging(ctx, source_id, org_id, acting_user_id)
    if staged["status"] != "staged":
        return staged

    current = await current_spec(org_uuid, user_uuid)
    proposed = await design_schema(org_uuid, user_uuid, source_uuid)
    plan = classify_migration(current, proposed)

    if not plan.steps:
        return await refresh_source(ctx, source_id, org_id, acting_user_id)

    shape = structural_shape("schema_migration", {"steps": [vars(s) for s in plan.steps]})

    trusted = False
    if plan.reversible:
        async with user_session(user_uuid) as db:
            trusted = await is_auto_apply_eligible(db, org_uuid, shape)

    if not (plan.reversible and trusted):
        spec = await enrich_spec(proposed)
        proposal_id = await propose_schema(
            org_uuid, user_uuid, source_uuid, spec, kind="schema_migration"
        )
        return {"status": "proposed", "proposal_id": str(proposal_id), "shape": shape}

    async with tenant_session(org_uuid) as db:
        for statement in render_migration(plan, tenant_schema_name(org_uuid)):
            await db.execute(text(statement))

    await refresh_source(ctx, source_id, org_id, acting_user_id)
    return {"status": "applied", "shape": shape, "steps": len(plan.steps)}
