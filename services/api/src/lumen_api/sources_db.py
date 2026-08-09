"""Connecting a customer's own database.

Structure first, data on demand (D10). Discovery reads the customer's
information_schema and stores what it found; not a byte is copied until
someone selects a table. That is what makes the diagram appear in seconds
instead of after a 500-table ERP finishes replicating.
"""

from __future__ import annotations

import json
import uuid
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import text

from lumen.architect.adapters.mysql import MySQLAdapter
from lumen.architect.adapters.postgres import PostgresAdapter
from lumen_api.auth.dependencies import Identity, current_identity, require_role
from lumen_api.credentials import decrypt_dsn, encrypt_dsn
from lumen_api.db.session import user_session
from lumen_api.errors import BadRequest, NotFound
from lumen_api.queue import enqueue_job
from lumen_api.tenant_db import ensure_tenant_schema

router = APIRouter(prefix="/v1/sources", tags=["sources"])


class DatabaseSourceCreate(BaseModel):
    name: str
    kind: Literal["postgres", "mysql"]
    dsn: str
    schema: str = "public"


class TableSelection(BaseModel):
    tables: list[str]


def _adapter(kind: str, dsn: str, schema: str):
    return PostgresAdapter(dsn, schema) if kind == "postgres" else MySQLAdapter(dsn, schema)


@router.post("/database")
async def connect_database(
    body: DatabaseSourceCreate,
    identity: Annotated[Identity, Depends(require_role("member"))],
) -> dict[str, Any]:
    """Connect a database and mirror its structure. No data is copied."""
    await ensure_tenant_schema(identity.org_id)

    try:
        structure = await _adapter(body.kind, body.dsn, body.schema).discover()
    except Exception as exc:  # noqa: BLE001 — the customer needs the reason
        raise BadRequest(f"Could not read that database: {exc}") from exc

    discovered = {
        "schema": body.schema,
        "declared": structure.declared,
        "tables": [
            {
                "name": t.name,
                "primary_key": list(t.primary_key or ()),
                "foreign_keys": [list(fk) for fk in t.foreign_keys],
                "columns": [
                    {"name": c.name, "sql_type": c.sql_type.value, "nullable": c.nullable}
                    for c in t.columns
                ],
            }
            for t in structure.tables
        ],
    }

    source_id = uuid.uuid4()
    async with user_session(identity.user_id) as db:
        await db.execute(
            text(
                "insert into public.data_sources "
                "(id, org_id, name, kind, status, dsn_encrypted, discovered_structure) "
                "values (:id, :org, :name, cast(:kind as public.source_kind), 'idle', "
                "        :dsn, cast(:structure as jsonb))"
            ),
            {
                "id": source_id,
                "org": identity.org_id,
                "name": body.name,
                "kind": body.kind,
                "dsn": encrypt_dsn(body.dsn),
                "structure": json.dumps(discovered),
            },
        )

    # The DSN is never echoed back, here or anywhere else.
    return {"id": str(source_id), "name": body.name, "tables": len(structure.tables)}


@router.get("/{source_id}/tables")
async def list_source_tables(
    source_id: uuid.UUID,
    identity: Annotated[Identity, Depends(current_identity)],
) -> dict[str, Any]:
    """What this database contains, and which of it has been imported."""
    async with user_session(identity.user_id) as db:
        row = (
            await db.execute(
                text(
                    "select discovered_structure, imported_tables "
                    "from public.data_sources where id = :id"
                ),
                {"id": source_id},
            )
        ).mappings().first()
    if row is None or not row["discovered_structure"]:
        raise NotFound("That source has no discovered structure")

    structure = dict(row["discovered_structure"])
    imported = set(row["imported_tables"] or [])
    return {
        "declared": structure.get("declared", False),
        "tables": [
            {**table, "imported": table["name"] in imported}
            for table in structure["tables"]
        ],
    }


@router.post("/{source_id}/tables")
async def select_tables(
    source_id: uuid.UUID,
    body: TableSelection,
    identity: Annotated[Identity, Depends(require_role("member"))],
) -> dict[str, Any]:
    """Choose which tables to import. Enqueues the copy; returns at once."""
    async with user_session(identity.user_id) as db:
        row = (
            await db.execute(
                text("select discovered_structure from public.data_sources where id = :id"),
                {"id": source_id},
            )
        ).mappings().first()
        if row is None or not row["discovered_structure"]:
            raise NotFound("That source has no discovered structure")

        known = {t["name"] for t in dict(row["discovered_structure"])["tables"]}
        unknown = set(body.tables) - known
        if unknown:
            raise BadRequest(f"Not present on that source: {', '.join(sorted(unknown))}")

        await db.execute(
            text(
                "update public.data_sources set imported_tables = cast(:tables as jsonb) "
                "where id = :id"
            ),
            {"tables": json.dumps(body.tables), "id": source_id},
        )

    await enqueue_job(
        "import_tables", str(source_id), str(identity.org_id), str(identity.user_id)
    )
    return {"id": str(source_id), "selected": body.tables}
