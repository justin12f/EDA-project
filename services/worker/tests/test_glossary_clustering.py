"""process_schedule -> glossary clustering -> GlossaryAgent -> proposal ->
accept -> CanonicalEntity, end to end against the live project.

Two sources share a column *name* (not just a similar one) on purpose: the
column-level embedding is real (this project's configured embedding
provider, not mocked), and this file's job is to verify the pipeline —
search, threshold, job hand-off, agent decision, proposal, accept,
certification — not to grade the embedding model's judgment of "cust_no" vs
"customer_id". Identical text embeds at ~1.0 similarity regardless of model
quality, which is what makes this deterministic.

Runs on MockProvider (LLM_MODE=mock) — see `lumen.llm.mock_provider`'s
`_propose_entity_mapping` for what it does with the Glossary agent's
`propose_entity_mapping` / `not_the_same_entity` registry.
"""

from __future__ import annotations

import uuid

import httpx
import pytest
import pytest_asyncio
from sqlalchemy import text

from lumen_api.auth.dependencies import Identity
from lumen_api.certification import certification_for
from lumen_api.datasets.store import SupabaseStorage
from lumen_api.db.session import user_session
from lumen_api.proposals import DecisionRequest, decide_proposal
from lumen_api.settings import get_settings
from lumen_worker.glossary import propose_entity_mapping
from lumen_worker.sentinel import process_schedule

pytestmark = pytest.mark.integration

# order_id / crm_row_id are deliberately distinct text, not just distinct
# concepts — CLUSTER_THRESHOLD is calibrated high enough that two *different*
# id-like columns should not coincidentally cross it, only customer_id's
# identical text on both sides should.
SOURCE_A_CSV = b"order_id,customer_id,region\n1,C-1,west\n2,C-2,east\n"
SOURCE_B_CSV = b"crm_row_id,customer_id,amount\n1,C-1,100\n2,C-2,250\n"


class _CapturingRedis:
    def __init__(self) -> None:
        self.enqueued: list[tuple[str, tuple, dict]] = []

    async def enqueue_job(self, name: str, *args, **kwargs) -> None:
        self.enqueued.append((name, args, kwargs))


def _admin_headers() -> dict[str, str]:
    key = get_settings().supabase_service_role_key.get_secret_value()
    return {"apikey": key, "Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _create_user() -> uuid.UUID:
    settings = get_settings()
    response = httpx.post(
        f"{settings.supabase_url}/auth/v1/admin/users",
        headers=_admin_headers(),
        json={
            "email": f"lumen-glossary-{uuid.uuid4().hex[:12]}@example.com",
            "password": uuid.uuid4().hex + "Aa1!",
            "email_confirm": True,
            "user_metadata": {"display_name": "Glossary Tester", "org_name": "Glossary Org"},
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


async def _seed_scheduled_source(identity: Identity, name: str, content: bytes) -> tuple[uuid.UUID, uuid.UUID]:
    path = f"org/{identity.org_id}/uploads/{uuid.uuid4().hex}.csv"
    await SupabaseStorage().upload(path, content, "text/csv")

    source_id = uuid.uuid4()
    async with user_session(identity.user_id) as db:
        await db.execute(
            text(
                "insert into public.data_sources "
                "(id, org_id, name, kind, status, object_path, table_name) "
                "values (:id, :org, :name, 'csv', 'idle', :path, :name)"
            ),
            {"id": source_id, "org": identity.org_id, "name": name, "path": path},
        )
        schedule_id = (
            await db.execute(
                text(
                    "insert into public.data_source_schedules (org_id, source_id, created_by) "
                    "values (:org, :source, :user) returning id"
                ),
                {"org": identity.org_id, "source": source_id, "user": identity.user_id},
            )
        ).scalar_one()
    return source_id, schedule_id


async def _tick(redis: _CapturingRedis, schedule_id: uuid.UUID, identity: Identity, source_id: uuid.UUID) -> dict:
    return await process_schedule(
        {"redis": redis}, str(schedule_id), str(identity.org_id), str(source_id), str(identity.user_id)
    )


async def test_a_shared_column_across_sources_is_proposed_and_accepted(identity):
    redis = _CapturingRedis()
    source_b, schedule_b = await _seed_scheduled_source(identity, "crm.csv", SOURCE_B_CSV)
    await _tick(redis, schedule_b, identity, source_b)  # nothing to cluster against yet

    source_a, schedule_a = await _seed_scheduled_source(identity, "orders.csv", SOURCE_A_CSV)
    await _tick(redis, schedule_a, identity, source_a)  # customer_id now matches source_b's

    jobs = [job for job in redis.enqueued if job[0] == "propose_entity_mapping"]
    assert len(jobs) == 1, f"expected exactly one clustering candidate, got {redis.enqueued}"
    _, args, _ = jobs[0]
    result = await propose_entity_mapping({}, *args)
    assert result["status"] == "proposed"

    async with user_session(identity.user_id) as db:
        proposal = (
            await db.execute(
                text("select kind, status, spec from public.proposals where id = :id"),
                {"id": result["proposal_id"]},
            )
        ).mappings().first()
    assert proposal["kind"] == "entity_mapping"
    assert proposal["status"] == "awaiting_review"
    member_columns = {(m["source_id"], m["column"]) for m in proposal["spec"]["members"]}
    assert member_columns == {(str(source_a), "customer_id"), (str(source_b), "customer_id")}

    decided = await decide_proposal(
        uuid.UUID(result["proposal_id"]), DecisionRequest(decision="accept"), identity
    )
    assert decided["status"] == "applied"

    async with user_session(identity.user_id) as db:
        entity = (
            await db.execute(
                text("select id, status from public.canonical_entities where id = :id"),
                {"id": uuid.UUID(decided["entity"]["id"])},
            )
        ).mappings().first()
        members = (
            await db.execute(
                text(
                    "select source_id, column_name from public.canonical_entity_members "
                    "where entity_id = :id"
                ),
                {"id": entity["id"]},
            )
        ).mappings().all()
    assert entity["status"] == "approved"
    assert {(m["source_id"], m["column_name"]) for m in members} == {(source_a, "customer_id"), (source_b, "customer_id")}


async def test_an_already_covered_cluster_is_not_re_proposed(identity):
    redis = _CapturingRedis()
    source_b, schedule_b = await _seed_scheduled_source(identity, "crm2.csv", SOURCE_B_CSV)
    await _tick(redis, schedule_b, identity, source_b)
    source_a, schedule_a = await _seed_scheduled_source(identity, "orders2.csv", SOURCE_A_CSV)
    await _tick(redis, schedule_a, identity, source_a)

    first_job = next(job for job in redis.enqueued if job[0] == "propose_entity_mapping")
    first_result = await propose_entity_mapping({}, *first_job[1])
    assert first_result["status"] == "proposed"

    # Re-running the same seed candidate (as a second tick that found the
    # same match would) must not create a second proposal for a pair already
    # covered — "a rejected cluster is not re-proposed" generalizes to "an
    # already-decided-or-pending one isn't re-proposed either" (ADR-0009 §2).
    second_result = await propose_entity_mapping({}, *first_job[1])
    assert second_result["status"] == "skipped"


async def test_certification_reflects_a_resolved_entity(identity):
    redis = _CapturingRedis()
    source_b, schedule_b = await _seed_scheduled_source(identity, "crm3.csv", SOURCE_B_CSV)
    await _tick(redis, schedule_b, identity, source_b)
    source_a, schedule_a = await _seed_scheduled_source(identity, "orders3.csv", SOURCE_A_CSV)
    await _tick(redis, schedule_a, identity, source_a)

    before = await certification_for(identity.org_id, identity.user_id, source_a)
    # No CanonicalEntity yet — nothing to resolve, so nothing is unresolved
    # either; certification depends on drift/pending-review state at this
    # point, not on entity mapping at all.
    assert before.unresolved_entity_columns == []

    job = next(job for job in redis.enqueued if job[0] == "propose_entity_mapping")
    result = await propose_entity_mapping({}, *job[1])
    await decide_proposal(uuid.UUID(result["proposal_id"]), DecisionRequest(decision="accept"), identity)

    after = await certification_for(identity.org_id, identity.user_id, source_a)
    assert after.unresolved_entity_columns == []
    assert after.checked_by == "scheduled_tick"
