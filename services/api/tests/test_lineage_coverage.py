"""Every artifact-producing write path declares its ArtifactDependency
(ADR-0010's own Action Item #7 — the same coverage-test shape ADR-0002 uses
for RLS: enumerate the call sites, fail loudly if one stops declaring).

Two write paths exist today (`lineage.py`'s own module docstring on why only
two): a source's currently-applied pipeline, and an approved canonical
entity's members. Each test here exercises the *real* accept code path
(`decide_proposal`), not a hand-inserted row shaped like what that path
would produce — the point is to catch the day someone edits
`_apply_cleaning_pipeline` or `_apply_entity_mapping` and forgets this part,
which a hand-inserted row could never catch.
"""

from __future__ import annotations

import json
import uuid

import httpx
import pytest
import pytest_asyncio
from sqlalchemy import text

from lumen_api.auth.dependencies import Identity
from lumen_api.datasets.store import SupabaseStorage
from lumen_api.db.session import user_session
from lumen_api.proposals import DecisionRequest, decide_proposal
from lumen_api.settings import get_settings

pytestmark = pytest.mark.integration

CSV = b"id,customer_id\n1,c-1\n2,c-2\n"


def _admin_headers() -> dict[str, str]:
    key = get_settings().supabase_service_role_key.get_secret_value()
    return {"apikey": key, "Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _create_user() -> uuid.UUID:
    settings = get_settings()
    response = httpx.post(
        f"{settings.supabase_url}/auth/v1/admin/users",
        headers=_admin_headers(),
        json={
            "email": f"lumen-lineage-{uuid.uuid4().hex[:12]}@example.com",
            "password": uuid.uuid4().hex + "Aa1!",
            "email_confirm": True,
            "user_metadata": {"display_name": "Lineage Tester", "org_name": "Lineage Org"},
        },
        timeout=30,
    )
    response.raise_for_status()
    return uuid.UUID(response.json()["id"])


def _delete_user(user_id: uuid.UUID) -> None:
    settings = get_settings()
    httpx.delete(
        f"{settings.supabase_url}/auth/v1/admin/users/{user_id}", headers=_admin_headers(), timeout=30
    )


async def _identity_of(user_id: uuid.UUID) -> Identity:
    async with user_session(user_id) as db:
        row = (await db.execute(text("select * from public.current_identity()"))).mappings().first()
    return Identity(
        user_id=row["user_id"],
        email=row["email"],
        display_name=row["display_name"],
        avatar_url=row["avatar_url"],
        org_id=row["org_id"],
        org_name=row["org_name"],
        org_slug=row["org_slug"],
        plan_code=row["plan_code"],
        role=str(row["role"]),
    )


@pytest_asyncio.fixture
async def identity():
    user_id = _create_user()
    try:
        yield await _identity_of(user_id)
    finally:
        _delete_user(user_id)


@pytest.fixture(autouse=True)
def _force_mock_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_MODE", "mock")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


async def _seed_source(identity: Identity) -> uuid.UUID:
    path = f"org/{identity.org_id}/uploads/{uuid.uuid4().hex}.csv"
    await SupabaseStorage().upload(path, CSV, "text/csv")
    source_id = uuid.uuid4()
    async with user_session(identity.user_id) as db:
        await db.execute(
            text(
                "insert into public.data_sources (id, org_id, name, kind, status, object_path, table_name) "
                "values (:id, :org, 'lineage.csv', 'csv', 'idle', :path, 'lineage')"
            ),
            {"id": source_id, "org": identity.org_id, "path": path},
        )
    return source_id


async def _dependency_rows(identity: Identity, source_id: uuid.UUID) -> list[dict]:
    async with user_session(identity.user_id) as db:
        rows = (
            await db.execute(
                text(
                    "select artifact_kind, artifact_id, columns from public.artifact_dependencies "
                    "where source_id = :source"
                ),
                {"source": source_id},
            )
        ).mappings().all()
    return [dict(r) for r in rows]


async def test_accepting_a_cleaning_pipeline_declares_a_pipeline_dependency(identity):
    source_id = await _seed_source(identity)
    rid = await _rid_for(identity, source_id)
    spec = {
        "rid": rid,
        "steps": [{"impute_categorical": {"columns": ["customer_id"], "strategy": "fixed"}}],
    }

    async with user_session(identity.user_id) as db:
        run_id = (
            await db.execute(
                text(
                    "insert into public.runs (org_id, source_id, thread_id, kind, status, backend) "
                    "values (:org, :source, :thread, 'chat', 'succeeded', 'polars') returning id"
                ),
                {"org": identity.org_id, "source": source_id, "thread": uuid.uuid4()},
            )
        ).scalar_one()
        proposal_id = (
            await db.execute(
                text(
                    "insert into public.proposals "
                    "(org_id, run_id, thread_id, author_agent, kind, spec, rationale) "
                    "select org_id, :run, thread_id, 'analyst', 'cleaning_pipeline', "
                    "       cast(:spec as jsonb), 'test' "
                    "from public.runs where id = :run returning id"
                ),
                {"run": run_id, "spec": json.dumps(spec)},
            )
        ).scalar_one()

    assert await _dependency_rows(identity, source_id) == []  # nothing before accept

    result = await decide_proposal(proposal_id, DecisionRequest(decision="accept"), identity)
    assert result["status"] == "applied"

    rows = await _dependency_rows(identity, source_id)
    assert len(rows) == 1
    assert rows[0]["artifact_kind"] == "pipeline"
    assert rows[0]["columns"] == ["customer_id"]


async def test_accepting_an_entity_mapping_declares_a_canonical_entity_dependency(identity):
    source_a = await _seed_source(identity)
    source_b = await _seed_source(identity)

    async with user_session(identity.user_id) as db:
        run_id = (
            await db.execute(
                text(
                    "insert into public.runs (org_id, thread_id, kind, status, backend) "
                    "values (:org, :id, 'glossary_propose', 'succeeded', 'polars') returning id"
                ),
                {"org": identity.org_id, "id": uuid.uuid4()},
            )
        ).scalar_one()
        spec = {
            "canonical_name": "customer",
            "canonical_type": "identifier",
            "members": [
                {"source_id": str(source_a), "column": "customer_id"},
                {"source_id": str(source_b), "column": "customer_id"},
            ],
            "reconciliation_rule": {},
        }
        proposal_id = (
            await db.execute(
                text(
                    "insert into public.proposals "
                    "(org_id, run_id, thread_id, author_agent, kind, spec, rationale) "
                    "values (:org, :run, :run, 'glossary', 'entity_mapping', cast(:spec as jsonb), 'test') "
                    "returning id"
                ),
                {"org": identity.org_id, "run": run_id, "spec": json.dumps(spec)},
            )
        ).scalar_one()

    assert await _dependency_rows(identity, source_a) == []

    result = await decide_proposal(proposal_id, DecisionRequest(decision="accept"), identity)
    assert result["status"] == "applied"

    rows_a = await _dependency_rows(identity, source_a)
    rows_b = await _dependency_rows(identity, source_b)
    assert len(rows_a) == 1 and rows_a[0]["artifact_kind"] == "canonical_entity"
    assert len(rows_b) == 1 and rows_b[0]["artifact_kind"] == "canonical_entity"
    assert rows_a[0]["artifact_id"] == rows_b[0]["artifact_id"]  # same entity, two members


async def _rid_for(identity: Identity, source_id: uuid.UUID) -> str:
    """Materialize `source_id` into a dataset handle the way `read_source`
    would, and return its rid — the accept path needs a real handle to
    resolve, not just a source row."""
    import os
    import tempfile

    from lumen.agents.master_factory import AgentMasterFactory
    from lumen_api.datasets.store import HandleStore

    async with user_session(identity.user_id) as db:
        path = (
            await db.execute(
                text("select object_path from public.data_sources where id = :id"), {"id": source_id}
            )
        ).scalar_one()
    payload = await SupabaseStorage().download(path)
    directory = tempfile.mkdtemp(prefix="lumen-lineage-test-")
    local = os.path.join(directory, "source.csv")
    with open(local, "wb") as file:
        file.write(payload)
    frame = AgentMasterFactory("polars").readers().read(local)
    handle = await HandleStore(identity.org_id, identity.user_id).put(
        frame, label="lineage", source_id=source_id
    )
    return handle.rid
