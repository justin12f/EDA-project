"""Ingestion jobs.

Two of them, and the split matters: staging lands data so a customer can
see it immediately (D4), and design proposes a schema. Keeping them apart
means a quota-denied or provider-down design never blocks a customer from
looking at the file they just uploaded.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from typing import Any

from sqlalchemy import text

from lumen.architect.adapters.file import FileAdapter
from lumen.readers.exceptions import ReaderError
from lumen_api.architect import design_schema, enrich_spec, propose_schema
from lumen_api.datasets.store import SupabaseStorage
from lumen_api.db.session import user_session
from lumen_api.settings import get_settings
from lumen_api.tenant_db import ensure_tenant_schema, tenant_raw_schema_name

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

    materialised.write_database(
        table_name=f"{tenant_raw_schema_name(org_uuid)}.{table}",
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
