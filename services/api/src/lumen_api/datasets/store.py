"""Org-scoped dataset handles: Parquet in Supabase Storage, metadata in Postgres.

The handle is what lets the API process that reads a source and the worker
process that cleans it talk about the same data. Two properties matter:

* **It is tenanted.** `rid` alone grants nothing — every read goes through an
  RLS-scoped session, so a handle from another organization simply is not found.
* **It is durable.** It survives a restart, which the in-process dict it replaces
  did not.

Storage paths are prefixed `org/<org_id>/`, so deleting a customer is a prefix
delete plus a cascade, not an audit.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
from sqlalchemy import text

from lumen.datasets.materialize import read_parquet, write_parquet
from lumen_api.db.session import user_session
from lumen_api.errors import NotFound
from lumen_api.settings import get_settings

HANDLE_TTL = timedelta(days=7)


@dataclass(frozen=True)
class DatasetHandle:
    rid: str
    org_id: uuid.UUID
    object_path: str
    backend: str
    schema: dict[str, str]
    row_count: int
    byte_size: int
    label: str = ""
    source_id: uuid.UUID | None = None


class SupabaseStorage:
    """Minimal Storage client — upload, download, ensure-bucket.

    Deliberately not the supabase-py client: this needs three calls, and a thin
    httpx wrapper is easier to reason about than a dependency whose async story
    has changed twice.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._base = f"{settings.supabase_url.rstrip('/')}/storage/v1"
        self._bucket = settings.storage_bucket
        key = settings.supabase_service_role_key.get_secret_value()
        self._headers = {"Authorization": f"Bearer {key}", "apikey": key}

    async def ensure_bucket(self) -> None:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(
                f"{self._base}/bucket/{self._bucket}", headers=self._headers
            )
            if response.status_code == 200:
                return
            await client.post(
                f"{self._base}/bucket",
                headers=self._headers,
                json={"name": self._bucket, "id": self._bucket, "public": False},
            )

    async def upload(self, path: str, data: bytes, content_type: str) -> None:
        await self.ensure_bucket()
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                f"{self._base}/object/{self._bucket}/{path}",
                headers={
                    **self._headers,
                    "Content-Type": content_type,
                    # Overwrite rather than 409 — a retried job must be idempotent.
                    "x-upsert": "true",
                },
                content=data,
            )
            response.raise_for_status()

    async def download(self, path: str) -> bytes:
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.get(
                f"{self._base}/object/{self._bucket}/{path}", headers=self._headers
            )
            response.raise_for_status()
            return response.content

    async def remove(self, path: str) -> None:
        async with httpx.AsyncClient(timeout=60) as client:
            await client.delete(
                f"{self._base}/object/{self._bucket}/{path}", headers=self._headers
            )


class HandleStore:
    """Reads and writes dataset handles for one user, inside their RLS context."""

    def __init__(self, org_id: uuid.UUID, user_id: uuid.UUID, backend: str = "polars") -> None:
        self._org_id = org_id
        self._user_id = user_id
        self._backend = backend
        self._storage = SupabaseStorage()

    async def put(
        self,
        frame: Any,
        *,
        backend: str | None = None,
        label: str = "",
        source_id: uuid.UUID | None = None,
    ) -> DatasetHandle:
        backend = backend or self._backend
        rid = uuid.uuid4().hex[:16]
        path = f"org/{self._org_id}/datasets/{rid}.parquet"

        with tempfile.TemporaryDirectory() as tmp:
            local = os.path.join(tmp, f"{rid}.parquet")
            meta = write_parquet(frame, backend, local)
            with open(local, "rb") as handle:
                await self._storage.upload(path, handle.read(), "application/octet-stream")

        async with user_session(self._user_id) as db:
            await db.execute(
                text(
                    "insert into public.dataset_handles "
                    "(rid, org_id, source_id, object_path, backend, schema, row_count, "
                    " byte_size, label, expires_at) "
                    "values (:rid, :org, :source, :path, :backend, cast(:schema as jsonb), "
                    " :rows, :bytes, :label, :expires)"
                ),
                {
                    "rid": rid,
                    "org": self._org_id,
                    "source": source_id,
                    "path": path,
                    "backend": backend,
                    "schema": _json(meta.schema),
                    "rows": meta.row_count,
                    "bytes": meta.byte_size,
                    "label": label,
                    "expires": datetime.now(timezone.utc) + HANDLE_TTL,
                },
            )

        return DatasetHandle(
            rid=rid,
            org_id=self._org_id,
            object_path=path,
            backend=backend,
            schema=meta.schema,
            row_count=meta.row_count,
            byte_size=meta.byte_size,
            label=label,
            source_id=source_id,
        )

    async def get(self, rid: str) -> DatasetHandle:
        async with user_session(self._user_id) as db:
            row = (
                await db.execute(
                    text(
                        "select rid, org_id, source_id, object_path, backend, schema, "
                        "       row_count, byte_size, label "
                        "from public.dataset_handles where rid = :rid"
                    ),
                    {"rid": rid},
                )
            ).mappings().first()

        if row is None:
            # Also the answer when the handle belongs to another org: RLS filtered
            # it out, and "not found" is the right thing to say about data the
            # caller has no right to know exists.
            raise NotFound(f"Dataset handle '{rid}' not found")

        return DatasetHandle(
            rid=row["rid"],
            org_id=row["org_id"],
            object_path=row["object_path"],
            backend=row["backend"],
            schema=dict(row["schema"] or {}),
            row_count=row["row_count"],
            byte_size=row["byte_size"],
            label=row["label"],
            source_id=row["source_id"],
        )

    async def latest_for_source(self, source_id: uuid.UUID) -> DatasetHandle | None:
        """The most recent handle materialised for this source, or None if it
        has never been read. `process_schedule` calls `put()` every tick, so
        this is what "the source's current data" means to anything — the
        Sentinel included — that runs after a tick rather than inside one.
        """
        async with user_session(self._user_id) as db:
            row = (
                await db.execute(
                    text(
                        "select rid, org_id, source_id, object_path, backend, schema, "
                        "       row_count, byte_size, label "
                        "from public.dataset_handles where source_id = :source "
                        "order by created_at desc limit 1"
                    ),
                    {"source": source_id},
                )
            ).mappings().first()
        if row is None:
            return None
        return DatasetHandle(
            rid=row["rid"],
            org_id=row["org_id"],
            object_path=row["object_path"],
            backend=row["backend"],
            schema=dict(row["schema"] or {}),
            row_count=row["row_count"],
            byte_size=row["byte_size"],
            label=row["label"],
            source_id=row["source_id"],
        )

    async def resolve(self, rid: str) -> Any:
        """Return a backend-native frame for the handle."""
        handle = await self.get(rid)
        payload = await self._storage.download(handle.object_path)

        directory = tempfile.mkdtemp(prefix="lumen-")
        local = os.path.join(directory, f"{rid}.parquet")
        with open(local, "wb") as file:
            file.write(payload)
        return read_parquet(local, handle.backend)


def _json(value: dict[str, str]) -> str:
    import json

    return json.dumps(value)
