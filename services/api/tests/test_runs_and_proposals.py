"""The chat turn → proposal → accept loop, against the live project.

Runs on `MockProvider`, forced regardless of whatever LLM_MODE the developer's
own `.env` resolves to — deterministic and keyless, so these hold on a machine
with no Groq or Anthropic key configured, same as the product itself does.

This file exists because of two bugs that only a real run surfaced: an
"agent_events.type" value the database enum did not have (the whole stream
died on the user's own turn), and a cleaning step indexing a polars LazyFrame
the way only an eager DataFrame supports (accepting a proposal died on the
first column it touched). Neither was reachable by constructing objects and
asserting on them — both needed the full path to actually run.
"""

from __future__ import annotations

import uuid

import httpx
import pytest
import pytest_asyncio
from sqlalchemy import text

from lumen_api.agents.runner import stream_run
from lumen_api.auth.dependencies import Identity
from lumen_api.datasets.store import HandleStore, SupabaseStorage
from lumen_api.db.session import user_session
from lumen_api.proposals import DecisionRequest, decide_proposal
from lumen_api.settings import get_settings

pytestmark = pytest.mark.integration

# 40% null country_code, 'a1' repeated in email_hash — always gives the mock
# planner something to propose.
CSV = b"id,country_code,email_hash\n1,DE,a1\n2,,b2\n3,US,a1\n4,FR,c3\n5,,d4\n"


def _admin_headers() -> dict[str, str]:
    key = get_settings().supabase_service_role_key.get_secret_value()
    return {"apikey": key, "Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _create_user() -> uuid.UUID:
    settings = get_settings()
    response = httpx.post(
        f"{settings.supabase_url}/auth/v1/admin/users",
        headers=_admin_headers(),
        json={
            "email": f"lumen-runs-{uuid.uuid4().hex[:12]}@example.com",
            "password": uuid.uuid4().hex + "Aa1!",
            "email_confirm": True,
            "user_metadata": {"display_name": "Runs Tester", "org_name": "Runs Org"},
        },
        timeout=30,
    )
    response.raise_for_status()
    return uuid.UUID(response.json()["id"])


def _delete_user(user_id: uuid.UUID) -> None:
    settings = get_settings()
    httpx.delete(
        f"{settings.supabase_url}/auth/v1/admin/users/{user_id}",
        headers=_admin_headers(),
        timeout=30,
    )


async def _identity_of(user_id: uuid.UUID) -> Identity:
    async with user_session(user_id) as db:
        row = (
            await db.execute(text("select * from public.current_identity()"))
        ).mappings().first()
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


async def _seed_source(identity: Identity) -> str:
    """Upload a CSV and register it as a data source, as the upload endpoint will."""
    path = f"org/{identity.org_id}/uploads/{uuid.uuid4().hex}.csv"
    await SupabaseStorage().upload(path, CSV, "text/csv")

    source_id = uuid.uuid4()
    async with user_session(identity.user_id) as db:
        await db.execute(
            text(
                "insert into public.data_sources "
                "(id, org_id, name, kind, status, object_path, table_name) "
                "values (:id, :org, 'clean_me.csv', 'csv', 'idle', :path, 'clean_me')"
            ),
            {"id": source_id, "org": identity.org_id, "path": path},
        )
    return str(source_id)


@pytest_asyncio.fixture
async def identity():
    user_id = _create_user()
    try:
        yield await _identity_of(user_id)
    finally:
        _delete_user(user_id)


@pytest.fixture(autouse=True)
def _force_mock_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """A real run, but never a real model call — see the module docstring."""
    monkeypatch.setenv("LLM_MODE", "mock")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


async def _run_to_completion(identity: Identity, prompt: str, source_id: str) -> list[dict]:
    return [
        item async for item in stream_run(identity, prompt, source_id=uuid.UUID(source_id))
    ]


# The prompts below say "source", never "dataset", on purpose: MockProvider's
# RID_RE matches literal "dataset <hex>" to recognise a reference to an
# already-materialized handle. A source id is a UUID, not that rid format —
# "clean dataset <uuid>" matches the first 8 hex characters before the first
# hyphen, sends a bogus rid to profile_source, and the run still finishes
# correctly (it falls through to read_source next), but no longer proves the
# golden path with zero recoveries, which is what this file is checking for.


async def test_a_chat_turn_streams_and_persists_a_real_transcript(identity):
    source_id = await _seed_source(identity)
    events = await _run_to_completion(
        identity, f"Please read and clean source {source_id}", source_id
    )

    types = [event["type"] for event in events]
    assert types[0] == "run_started"
    assert types[-1] == "done"

    user_turn = next(e for e in events if e["type"] == "message" and e["payload"].get("role") == "user")
    assert user_turn["payload"]["text"] == f"Please read and clean source {source_id}"

    tool_calls = {e["payload"]["name"] for e in events if e["type"] == "tool_call"}
    assert {"read_source", "profile_source", "propose_cleaning_pipeline"} <= tool_calls
    assert all(
        e["payload"]["ok"] for e in events if e["type"] == "tool_result"
    ), "every tool in this path is expected to succeed on a clean CSV"

    done = events[-1]["payload"]
    assert done["error"] is None
    assert done["stop_reason"] == "done"

    thread_id = events[0]["payload"]["thread_id"]
    async with user_session(identity.user_id) as db:
        proposal = (
            await db.execute(
                text(
                    "select status, spec, rationale from public.proposals "
                    "where thread_id = :thread"
                ),
                {"thread": thread_id},
            )
        ).mappings().first()

        # The bug this regresses against: a run that could not even record the
        # user's own turn never reached this insert at all.
        persisted_types = (
            await db.execute(
                text(
                    "select distinct e.type from public.agent_events e "
                    "join public.runs r on r.id = e.run_id where r.thread_id = :thread"
                ),
                {"thread": thread_id},
            )
        ).scalars().all()

    assert proposal is not None, "propose_cleaning_pipeline must persist a row, not just answer"
    assert proposal["status"] == "awaiting_review"
    assert proposal["spec"]["rid"]
    assert len(proposal["spec"]["steps"]) >= 1
    assert "message" in persisted_types and "tool_call" in persisted_types


async def test_accepting_a_proposal_actually_runs_the_pipeline(identity):
    source_id = await _seed_source(identity)
    events = await _run_to_completion(
        identity, f"Please read and clean source {source_id}", source_id
    )
    thread_id = events[0]["payload"]["thread_id"]

    async with user_session(identity.user_id) as db:
        proposal_id = (
            await db.execute(
                text("select id from public.proposals where thread_id = :thread"),
                {"thread": thread_id},
            )
        ).scalar_one()

    result = await decide_proposal(proposal_id, DecisionRequest(decision="accept"), identity)

    assert result["status"] == "applied"
    # remove_duplicates_rows drops exact row matches; two rows here share an
    # email_hash but differ elsewhere, so the count itself does not move —
    # the regression this guards is that this whole call used to be a 500,
    # not that a row goes missing. The null count is the unambiguous signal
    # that the pipeline actually ran rather than being marked applied for free.
    assert result["result"]["row_count"] == 5

    cleaned = await HandleStore(identity.org_id, identity.user_id).resolve(result["result"]["rid"])
    cleaned = cleaned.collect() if hasattr(cleaned, "collect") else cleaned
    assert cleaned["country_code"].null_count() == 0, "imputation must have filled every null"

    async with user_session(identity.user_id) as db:
        status = (
            await db.execute(
                text("select status from public.proposals where id = :id"),
                {"id": proposal_id},
            )
        ).scalar_one()
    assert status == "applied"


async def test_rejecting_a_proposal_leaves_the_dataset_untouched(identity):
    source_id = await _seed_source(identity)
    events = await _run_to_completion(
        identity, f"Please read and clean source {source_id}", source_id
    )
    thread_id = events[0]["payload"]["thread_id"]

    async with user_session(identity.user_id) as db:
        proposal_id = (
            await db.execute(
                text("select id from public.proposals where thread_id = :thread"),
                {"thread": thread_id},
            )
        ).scalar_one()

    result = await decide_proposal(proposal_id, DecisionRequest(decision="reject"), identity)
    assert result["status"] == "rejected"

    async with user_session(identity.user_id) as db:
        handle_count = (
            await db.execute(
                text(
                    "select count(*) from public.dataset_handles where object_path like :like"
                ),
                {"like": f"org/{identity.org_id}/%"},
            )
        ).scalar_one()
    # The original read_source handle, and nothing produced by an apply.
    assert handle_count == 1
