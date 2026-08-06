"""ArtifactDependency + ImpactReport (ADR-0010), against the live project.

The match-rate arithmetic is worked out by hand in the test itself (not just
"some finding appears") — a wrong sign or a wrong normalization order would
still produce *a* finding, just the wrong one, and only checking the exact
numbers catches that.
"""

from __future__ import annotations

import uuid

import httpx
import pytest
import pytest_asyncio
from sqlalchemy import text

from lumen_api.auth.dependencies import Identity
from lumen_api.datasets.store import HandleStore, SupabaseStorage
from lumen_api.db.session import service_session, user_session
from lumen_api.impact import compute_impact_report
from lumen_api.settings import get_settings

pytestmark = pytest.mark.integration

# customer_id's third value is null on purpose: impute_categorical(fixed,
# "ZZZ") turns it into a value that will not match anything on the other
# side, which is the whole point — a predictable, one-directional shift.
SOURCE_A_CSV = b"id,customer_id,region\n1,c-1,west\n2,c-2,east\n3,,north\n"
SOURCE_B_CSV = b"id,cust_ref\n1,C1\n2,C2\n3,C4\n"
STEPS = [{"impute_categorical": {"columns": ["customer_id"], "strategy": "fixed", "fill_value": "ZZZ"}}]
RULE = {"strip": "-", "case": "upper"}


def _admin_headers() -> dict[str, str]:
    key = get_settings().supabase_service_role_key.get_secret_value()
    return {"apikey": key, "Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _create_user() -> uuid.UUID:
    settings = get_settings()
    response = httpx.post(
        f"{settings.supabase_url}/auth/v1/admin/users",
        headers=_admin_headers(),
        json={
            "email": f"lumen-impact-{uuid.uuid4().hex[:12]}@example.com",
            "password": uuid.uuid4().hex + "Aa1!",
            "email_confirm": True,
            "user_metadata": {"display_name": "Impact Tester", "org_name": "Impact Org"},
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
    # Impact summarization takes the deterministic fallback path under mock
    # mode (impact.py's own choice — nothing to paraphrase into a call worth
    # making) — this suite asserts on the fallback text precisely because of
    # that, not despite it.
    monkeypatch.setenv("LLM_MODE", "mock")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


async def _seed_source(identity: Identity, name: str, content: bytes) -> tuple[uuid.UUID, str]:
    path = f"org/{identity.org_id}/uploads/{uuid.uuid4().hex}.csv"
    await SupabaseStorage().upload(path, content, "text/csv")
    source_id = uuid.uuid4()
    async with user_session(identity.user_id) as db:
        await db.execute(
            text(
                "insert into public.data_sources (id, org_id, name, kind, status, object_path, table_name) "
                "values (:id, :org, :name, 'csv', 'idle', :path, :name)"
            ),
            {"id": source_id, "org": identity.org_id, "name": name, "path": path},
        )
    return source_id, path


async def _materialize(identity: Identity, source_id: uuid.UUID, path: str) -> str:
    """Read the uploaded CSV into a dataset handle tied to `source_id` — the
    same shape `read_source` produces, needed before anything can shadow-run
    against it."""
    import os
    import tempfile

    from lumen.agents.master_factory import AgentMasterFactory

    payload = await SupabaseStorage().download(path)
    directory = tempfile.mkdtemp(prefix="lumen-impact-test-")
    local = os.path.join(directory, "source.csv")
    with open(local, "wb") as file:
        file.write(payload)
    frame = AgentMasterFactory("polars").readers().read(local)
    store = HandleStore(identity.org_id, identity.user_id)
    handle = await store.put(frame, label=str(source_id), source_id=source_id)
    return handle.rid


async def _approve_entity(
    identity: Identity, source_a: uuid.UUID, column_a: str, source_b: uuid.UUID, column_b: str
) -> uuid.UUID:
    """What `_apply_entity_mapping` writes on accept — built directly here
    since this file is testing what reads `canonical_entity_members` and
    `artifact_dependencies` afterward, not the accept flow itself (that is
    `test_glossary_clustering.py`'s job)."""
    async with user_session(identity.user_id) as db:
        entity_id = (
            await db.execute(
                text(
                    "insert into public.canonical_entities "
                    "(org_id, name, entity_type, reconciliation_rule, status, created_by) "
                    "values (:org, 'customer', 'identifier', cast(:rule as jsonb), 'approved', :user) "
                    "returning id"
                ),
                {"org": identity.org_id, "rule": '{"strip": "-", "case": "upper"}', "user": identity.user_id},
            )
        ).scalar_one()
        for source_id, column in ((source_a, column_a), (source_b, column_b)):
            await db.execute(
                text(
                    "insert into public.canonical_entity_members (entity_id, source_id, column_name) "
                    "values (:entity, :source, :column)"
                ),
                {"entity": entity_id, "source": source_id, "column": column},
            )
            await db.execute(
                text(
                    "insert into public.artifact_dependencies "
                    "(org_id, artifact_kind, artifact_id, source_id, columns) "
                    "values (:org, 'canonical_entity', :entity, :source, :columns)"
                ),
                {"org": identity.org_id, "entity": entity_id, "source": source_id, "columns": [column]},
            )
    return entity_id


async def test_a_pipeline_change_shifts_a_canonical_entitys_match_rate(identity):
    source_a, path_a = await _seed_source(identity, "orders.csv", SOURCE_A_CSV)
    source_b, path_b = await _seed_source(identity, "crm.csv", SOURCE_B_CSV)
    rid_a = await _materialize(identity, source_a, path_a)
    await _materialize(identity, source_b, path_b)

    entity_id = await _approve_entity(identity, source_a, "customer_id", source_b, "cust_ref")

    proposal_id = uuid.uuid4()
    async with user_session(identity.user_id) as db:
        run_id = (
            await db.execute(
                text(
                    "insert into public.runs (org_id, thread_id, kind, status, backend) "
                    "values (:org, :id, 'chat', 'succeeded', 'polars') returning id"
                ),
                {"org": identity.org_id, "id": uuid.uuid4()},
            )
        ).scalar_one()
        await db.execute(
            text(
                "insert into public.proposals (id, org_id, run_id, thread_id, author_agent, kind, spec, rationale) "
                "values (:id, :org, :run, :run, 'analyst', 'cleaning_pipeline', '{}'::jsonb, 'test')"
            ),
            {"id": proposal_id, "org": identity.org_id, "run": run_id},
        )

    report = await compute_impact_report(
        identity.org_id, identity.user_id,
        proposal_id=proposal_id, source_id=source_a, rid=rid_a, steps=STEPS,
    )

    assert report["dependents_checked"] == 1
    assert report["dependents_total"] == 1
    assert len(report["findings"]) == 1
    finding = report["findings"][0]
    assert finding["artifact_kind"] == "canonical_entity"
    assert finding["artifact_id"] == str(entity_id)
    assert finding["metric"] == "match_rate"
    # Before: {c-1, c-2} -> {C1, C2}, both present on the other side -> 1.0.
    # After: {c-1, c-2, ZZZ} -> {C1, C2, ZZZ}, ZZZ matches nothing -> 2/3.
    assert finding["before"] == pytest.approx(1.0)
    assert finding["after"] == pytest.approx(2 / 3, abs=1e-4)
    assert finding["delta_pct"] == pytest.approx(2 / 3 - 1.0, abs=1e-4)

    async with service_session() as db:
        row = (
            await db.execute(
                text("select dependents_checked, findings, summary from public.impact_reports where proposal_id = :id"),
                {"id": proposal_id},
            )
        ).mappings().first()
    assert row is not None
    assert row["dependents_checked"] == 1
    assert len(row["findings"]) == 1
    assert "match_rate" in row["summary"]


async def test_no_dependents_produces_an_honest_empty_report(identity):
    source_a, path_a = await _seed_source(identity, "solo.csv", SOURCE_A_CSV)
    rid_a = await _materialize(identity, source_a, path_a)

    proposal_id = uuid.uuid4()
    async with user_session(identity.user_id) as db:
        run_id = (
            await db.execute(
                text(
                    "insert into public.runs (org_id, thread_id, kind, status, backend) "
                    "values (:org, :id, 'chat', 'succeeded', 'polars') returning id"
                ),
                {"org": identity.org_id, "id": uuid.uuid4()},
            )
        ).scalar_one()
        await db.execute(
            text(
                "insert into public.proposals (id, org_id, run_id, thread_id, author_agent, kind, spec, rationale) "
                "values (:id, :org, :run, :run, 'analyst', 'cleaning_pipeline', '{}'::jsonb, 'test')"
            ),
            {"id": proposal_id, "org": identity.org_id, "run": run_id},
        )

    report = await compute_impact_report(
        identity.org_id, identity.user_id,
        proposal_id=proposal_id, source_id=source_a, rid=rid_a, steps=STEPS,
    )
    assert report["dependents_checked"] == 0
    assert report["findings"] == []
    assert "No other artifacts" in report["summary"]
