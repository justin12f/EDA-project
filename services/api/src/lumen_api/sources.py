"""Data sources: upload a file, list what a workspace has connected.

A source is the org-level record — "there is a file called signups.csv". Reading
it into a working dataset (a `dataset_handle`) is a separate step the agent takes
through the `read_source` tool, because loading and materializing a frame is
work worth letting the agent decide when to do, not something upload should do
for it.
"""

from __future__ import annotations

import os
import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, UploadFile
from sqlalchemy import text

from lumen_api.auth.dependencies import Identity, current_identity
from lumen_api.datasets.store import SupabaseStorage
from lumen_api.db.session import user_session
from lumen_api.errors import BadRequest, NotFound, UnsupportedMedia

router = APIRouter(prefix="/v1/sources", tags=["sources"])

# The columns worth reading in from `source_kind`. xlsx/xls have no reader-agnostic
# home in that enum, so uploads are scoped to what the engine and the schema both
# already agree on.
_KIND_BY_EXTENSION = {".csv": "csv", ".json": "json", ".parquet": "parquet"}

# Bodies this large would rather fail the upload than the request timeout — a
# multi-gigabyte file needs a different ingestion path (streamed to Storage in
# chunks, or a warehouse connection), not a bigger number here.
MAX_UPLOAD_BYTES = 200 * 1024 * 1024


@router.get("")
async def list_sources(
    identity: Annotated[Identity, Depends(current_identity)],
) -> dict[str, Any]:
    async with user_session(identity.user_id) as db:
        rows = (
            await db.execute(
                text(
                    "select id, name, kind, status, row_count, byte_size, created_at "
                    "from public.data_sources order by created_at desc"
                )
            )
        ).mappings().all()
    return {"sources": [dict(row) for row in rows]}


@router.get("/{source_id}")
async def get_source(
    source_id: uuid.UUID,
    identity: Annotated[Identity, Depends(current_identity)],
) -> dict[str, Any]:
    async with user_session(identity.user_id) as db:
        row = (
            await db.execute(
                text(
                    "select id, name, kind, status, row_count, byte_size, created_at "
                    "from public.data_sources where id = :id"
                ),
                {"id": source_id},
            )
        ).mappings().first()
    if row is None:
        raise NotFound(f"No data source with id {source_id}")
    return dict(row)


@router.post("")
async def upload_source(
    identity: Annotated[Identity, Depends(current_identity)],
    file: UploadFile,
) -> dict[str, Any]:
    extension = os.path.splitext(file.filename or "")[1].lower()
    kind = _KIND_BY_EXTENSION.get(extension)
    if kind is None:
        raise UnsupportedMedia(
            f"'{extension or '(no extension)'}' is not supported. "
            f"Upload one of: {', '.join(sorted(_KIND_BY_EXTENSION))}."
        )

    payload = await file.read()
    if not payload:
        raise BadRequest("The uploaded file is empty")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise BadRequest(
            f"File is {len(payload) / 1_048_576:.1f}MB; the limit is "
            f"{MAX_UPLOAD_BYTES / 1_048_576:.0f}MB"
        )

    source_id = uuid.uuid4()
    path = f"org/{identity.org_id}/uploads/{source_id}{extension}"
    await SupabaseStorage().upload(path, payload, file.content_type or "application/octet-stream")

    async with user_session(identity.user_id) as db:
        await db.execute(
            text(
                "insert into public.data_sources "
                "(id, org_id, name, kind, status, object_path, byte_size, created_by) "
                "values (:id, :org, :name, cast(:kind as public.source_kind), 'idle', "
                "        :path, :size, :user)"
            ),
            {
                "id": source_id,
                "org": identity.org_id,
                "name": file.filename,
                "kind": kind,
                "path": path,
                "size": len(payload),
                "user": identity.user_id,
            },
        )

    return {
        "id": str(source_id),
        "name": file.filename,
        "kind": kind,
        "status": "idle",
        "byte_size": len(payload),
    }
