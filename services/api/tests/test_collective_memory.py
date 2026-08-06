"""record_occurrence / record_outcome / lookup (ADR-0012), and decide_proposal
's wiring to them, against the live project: the opt-in gate, the per-org
one-contribution-ever ledger cap (§4's poisoning resistance), and cross-org
corroboration all read wrong from a mocked session — this exercises the real
tables and the real accept/reject path.

`global_patterns` has no org_id and no foreign key to anything this test's
`identity` fixture deletes on teardown, unlike every other table this session
has written to — its rows are cleaned up explicitly, by exact
(pattern_signature, fix_signature) pair, never a broad delete.
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid

import httpx
import pytest
import pytest_asyncio
from sqlalchemy import text

from lumen.agents.master_factory import AgentMasterFactory
from lumen_api.auth.dependencies import Identity
from lumen_api.datasets.store import HandleStore, SupabaseStorage
from lumen_api.db.session import service_session, user_session
from lumen_api.global_patterns import lookup, record_occurrence, record_outcome
from lumen_api.proposals import DecisionRequest, decide_proposal
from lumen_api.settings import get_settings

pytestmark = pytest.mark.integration

CSV = b"id,customer_id\n1,c-1\n2,c-2\n"


def _admin_headers() -> dict[str, str]:
    key = get_settings().supabase_service_role_key.get_secret_value()
    return {"apikey": key, "Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _create_user(display_name: str = "Collective Memory Tester", org_name: str = "Collective Org") -> uuid.UUID:
    settings = get_settings()
    response = httpx.post(
        f"{settings.supabase_url}/auth/v1/admin/users",
        headers=_admin_headers(),
        json={
            "email": f"lumen-gp-{uuid.uuid4().hex[:12]}@example.com",
            "password": uuid.uuid4().hex + "Aa1!",
            "email_confirm": True,
            "user_metadata": {"display_name": display_name, "org_name": org_name},
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


@pytest_asyncio.fixture
async def other_org_identity():
    """A second, unrelated org — for proving corroboration is a cross-org
    property, not a per-call counter."""
    user_id = _create_user(display_name="Second Org Tester", org_name="Second Collective Org")
    try:
        yield await _identity_of(user_id)
    finally:
        _delete_user(user_id)


@pytest_asyncio.fixture
async def patterns():
    """Registers (pattern_signature, fix_signature) pairs a test wrote to the
    org-less `global_patterns` table, and deletes exactly those rows on
    teardown — nothing here cascades from deleting a test user."""
    pairs: list[tuple[str, str]] = []
    yield pairs
    if not pairs:
        return
    async with service_session() as db:
        for pattern, fix in pairs:
            await db.execute(
                text(
                    "delete from public.global_patterns "
                    "where pattern_signature = :pattern and fix_signature = :fix"
                ),
                {"pattern": pattern, "fix": fix},
            )


def _signature_pair() -> tuple[str, str]:
    """A unique-per-call (pattern, fix) pair. The fix half is fixed at the
    literal `structural_shape("pipeline_patch", ...)` actually produces for
    `_propose_pipeline_patch`'s hardcoded single impute_categorical step —
    the decide_proposal tests below never fabricate one, they let
    decide_proposal compute it for real, so this has to match. The pattern
    half carries the entropy: unique per call, so the composite key never
    collides between tests even though `fix` repeats."""
    token = uuid.uuid4().hex[:10]
    return f"schema_change:added:{token}", "pipeline_patch:imputation"


async def _set_contribution(identity: Identity, enabled: bool) -> None:
    async with user_session(identity.user_id) as db:
        await db.execute(
            text("update public.organizations set pattern_contribution_enabled = :enabled where id = :org"),
            {"enabled": enabled, "org": identity.org_id},
        )


async def _seed_source(identity: Identity) -> uuid.UUID:
    path = f"org/{identity.org_id}/uploads/{uuid.uuid4().hex}.csv"
    await SupabaseStorage().upload(path, CSV, "text/csv")
    source_id = uuid.uuid4()
    async with user_session(identity.user_id) as db:
        await db.execute(
            text(
                "insert into public.data_sources (id, org_id, name, kind, status, object_path, table_name) "
                "values (:id, :org, 'gp.csv', 'csv', 'idle', :path, 'gp')"
            ),
            {"id": source_id, "org": identity.org_id, "path": path},
        )
    payload = await SupabaseStorage().download(path)
    directory = tempfile.mkdtemp(prefix="lumen-gp-test-")
    local = os.path.join(directory, "source.csv")
    with open(local, "wb") as file:
        file.write(payload)
    frame = AgentMasterFactory("polars").readers().read(local)
    await HandleStore(identity.org_id, identity.user_id).put(frame, label="gp", source_id=source_id)
    return source_id


async def _propose_pipeline_patch(
    identity: Identity, source_id: uuid.UUID, *, drift_pattern_signature: str | None
) -> uuid.UUID:
    """A pipeline_patch proposal shaped exactly as sentinel.py's propose_patch
    writes one — including (or, for the "no drift pattern" test, omitting)
    the ADR-0012 stamp in its spec."""
    async with user_session(identity.user_id) as db:
        rid = (
            await db.execute(
                text("select rid from public.dataset_handles where source_id = :source limit 1"),
                {"source": source_id},
            )
        ).scalar_one()
        run_id = (
            await db.execute(
                text(
                    "insert into public.runs (org_id, source_id, thread_id, kind, status, backend) "
                    "values (:org, :source, :thread, 'chat', 'succeeded', 'polars') returning id"
                ),
                {"org": identity.org_id, "source": source_id, "thread": uuid.uuid4()},
            )
        ).scalar_one()
        spec = {"rid": rid, "steps": [{"impute_categorical": {"columns": ["customer_id"], "strategy": "fixed"}}]}
        if drift_pattern_signature is not None:
            spec["drift_pattern_signature"] = drift_pattern_signature
        proposal_id = (
            await db.execute(
                text(
                    "insert into public.proposals "
                    "(org_id, run_id, thread_id, author_agent, kind, spec, rationale) "
                    "select org_id, :run, thread_id, 'sentinel', 'pipeline_patch', "
                    "       cast(:spec as jsonb), 'test' "
                    "from public.runs where id = :run returning id"
                ),
                {"run": run_id, "spec": json.dumps(spec)},
            )
        ).scalar_one()
    return proposal_id


async def test_contribution_is_a_no_op_until_the_org_opts_in(identity, patterns):
    pattern, fix = _signature_pair()
    patterns.append((pattern, fix))
    # pattern_contribution_enabled defaults false — no explicit opt-out needed.
    async with user_session(identity.user_id) as db:
        await record_occurrence(db, identity.org_id, pattern, fix)
        await record_outcome(db, identity.org_id, pattern, fix, applied=True)
    assert await lookup(pattern, fix) is None


async def test_opting_in_lets_an_occurrence_reach_the_shared_pool(identity, patterns):
    pattern, fix = _signature_pair()
    patterns.append((pattern, fix))
    await _set_contribution(identity, True)

    async with user_session(identity.user_id) as db:
        await record_occurrence(db, identity.org_id, pattern, fix)

    match = await lookup(pattern, fix)
    assert match is not None
    assert match.occurrences == 1
    assert match.applied_count == 0 and match.rejected_count == 0


async def test_a_single_org_cannot_inflate_occurrences_by_repeating_itself(identity, patterns):
    # ADR-0012 §4's poisoning-resistance property: a buggy or bad-faith org
    # re-triggering the same diagnosis contributes one occurrence, not one
    # per attempt.
    pattern, fix = _signature_pair()
    patterns.append((pattern, fix))
    await _set_contribution(identity, True)

    async with user_session(identity.user_id) as db:
        for _ in range(5):
            await record_occurrence(db, identity.org_id, pattern, fix)

    match = await lookup(pattern, fix)
    assert match.occurrences == 1


async def test_a_second_distinct_org_still_adds_a_genuine_occurrence(identity, other_org_identity, patterns):
    # The other half of the same property: corroboration must still be able
    # to accumulate when it is *actually* coming from distinct orgs.
    pattern, fix = _signature_pair()
    patterns.append((pattern, fix))
    await _set_contribution(identity, True)
    await _set_contribution(other_org_identity, True)

    async with user_session(identity.user_id) as db:
        await record_occurrence(db, identity.org_id, pattern, fix)
    async with user_session(other_org_identity.user_id) as db:
        await record_occurrence(db, other_org_identity.org_id, pattern, fix)

    match = await lookup(pattern, fix)
    assert match.occurrences == 2


async def test_record_outcome_updates_counts_and_recomputes_success_rate(identity, other_org_identity, patterns):
    pattern, fix = _signature_pair()
    patterns.append((pattern, fix))
    await _set_contribution(identity, True)
    await _set_contribution(other_org_identity, True)

    # Two distinct orgs applying it, a third decision (a second signature-
    # pair-sharing decision from the *same* org as one of the two) must not
    # count again — outcomes are claimed per-org exactly like occurrences.
    async with user_session(identity.user_id) as db:
        await record_outcome(db, identity.org_id, pattern, fix, applied=True)
        await record_outcome(db, identity.org_id, pattern, fix, applied=False)  # ignored: already claimed
    async with user_session(other_org_identity.user_id) as db:
        await record_outcome(db, other_org_identity.org_id, pattern, fix, applied=False)

    match = await lookup(pattern, fix)
    assert match.applied_count == 1
    assert match.rejected_count == 1
    assert match.success_rate == pytest.approx(0.5)


async def test_occurrence_and_outcome_are_claimed_independently(identity, patterns):
    pattern, fix = _signature_pair()
    patterns.append((pattern, fix))
    await _set_contribution(identity, True)

    async with user_session(identity.user_id) as db:
        await record_occurrence(db, identity.org_id, pattern, fix)
        await record_outcome(db, identity.org_id, pattern, fix, applied=True)

    match = await lookup(pattern, fix)
    # Both landed — the ledger's two flags do not block each other.
    assert match.occurrences == 1
    assert match.applied_count == 1


async def test_decide_proposal_accept_records_an_outcome_for_a_pipeline_patch(identity, patterns):
    pattern, fix = _signature_pair()
    patterns.append((pattern, fix))
    await _set_contribution(identity, True)
    source_id = await _seed_source(identity)
    proposal_id = await _propose_pipeline_patch(identity, source_id, drift_pattern_signature=pattern)

    result = await decide_proposal(proposal_id, DecisionRequest(decision="accept"), identity)
    assert result["status"] == "applied"

    match = await lookup(pattern, fix)
    assert match is not None
    assert match.applied_count == 1
    assert match.rejected_count == 0


async def test_decide_proposal_reject_records_an_outcome(identity, patterns):
    pattern, fix = _signature_pair()
    patterns.append((pattern, fix))
    await _set_contribution(identity, True)
    source_id = await _seed_source(identity)
    proposal_id = await _propose_pipeline_patch(identity, source_id, drift_pattern_signature=pattern)

    result = await decide_proposal(proposal_id, DecisionRequest(decision="reject"), identity)
    assert result["status"] == "rejected"

    match = await lookup(pattern, fix)
    assert match is not None
    assert match.applied_count == 0
    assert match.rejected_count == 1


async def test_decide_proposal_does_not_contribute_when_the_org_has_not_opted_in(identity, patterns):
    pattern, fix = _signature_pair()
    patterns.append((pattern, fix))
    # No _set_contribution call — pattern_contribution_enabled stays false.
    source_id = await _seed_source(identity)
    proposal_id = await _propose_pipeline_patch(identity, source_id, drift_pattern_signature=pattern)

    result = await decide_proposal(proposal_id, DecisionRequest(decision="accept"), identity)
    assert result["status"] == "applied"

    assert await lookup(pattern, fix) is None


async def test_a_pipeline_patch_without_a_drift_pattern_signature_contributes_nothing(identity, patterns):
    # Defends the kind+field guard in proposals.py: an older proposal
    # created before this ADR (or any non-Sentinel pipeline_patch, if one
    # ever existed) has no drift_pattern_signature and must not attempt a
    # lookup keyed on a signature that was never computed.
    _, fix = _signature_pair()
    await _set_contribution(identity, True)
    source_id = await _seed_source(identity)
    proposal_id = await _propose_pipeline_patch(identity, source_id, drift_pattern_signature=None)

    result = await decide_proposal(proposal_id, DecisionRequest(decision="accept"), identity)
    assert result["status"] == "applied"

    # There is no pattern signature to look this fix up under at all — the
    # only assertion that makes sense is that nothing raised, and that a
    # lookup keyed on a signature this proposal never declared finds nothing.
    assert await lookup("schema_change:added", fix) is None
