"""prepare_and_update_column_baselines / compute_and_remember_source_baseline
(ADR-0013) against the live project: calibration, cold-start silence, the
manual-override escape hatch, and the null-rate baselines handed back for
detect_drift all read wrong from a mocked session.

Frames are built directly with polars rather than uploaded through Storage —
these tests are about the baseline logic, not the file-handling pipeline
that test_sentinel_diagnosis.py's full-tick tests already cover.
"""

from __future__ import annotations

import uuid

import httpx
import polars as pl
import pytest
import pytest_asyncio
from sqlalchemy import text

from lumen.sentinel import MIN_SAMPLE_SIZE
from lumen_api.auth.dependencies import Identity
from lumen_api.baselines import compute_and_remember_source_baseline, prepare_and_update_column_baselines
from lumen_api.db.session import user_session
from lumen_api.settings import get_settings

pytestmark = pytest.mark.integration


def _admin_headers() -> dict[str, str]:
    key = get_settings().supabase_service_role_key.get_secret_value()
    return {"apikey": key, "Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _create_user() -> uuid.UUID:
    settings = get_settings()
    response = httpx.post(
        f"{settings.supabase_url}/auth/v1/admin/users",
        headers=_admin_headers(),
        json={
            "email": f"lumen-baseline-{uuid.uuid4().hex[:12]}@example.com",
            "password": uuid.uuid4().hex + "Aa1!",
            "email_confirm": True,
            "user_metadata": {"display_name": "Baseline Tester", "org_name": "Baseline Org"},
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


async def _seed_source(identity: Identity) -> uuid.UUID:
    """A minimal `data_sources` row — column_baselines/source_baselines both
    foreign-key into it, but these tests call the baseline functions
    directly with hand-built frames, never through the upload/Storage path,
    so no real file needs to exist behind it."""
    source_id = uuid.uuid4()
    async with user_session(identity.user_id) as db:
        await db.execute(
            text(
                "insert into public.data_sources (id, org_id, name, kind, status) "
                "values (:id, :org, 'baseline-test.csv', 'csv', 'idle')"
            ),
            {"id": source_id, "org": identity.org_id},
        )
    return source_id


SCHEMA = {"amount": "Float64", "country": "String"}


def _frame(amount: list[float], country: list[str]) -> pl.DataFrame:
    return pl.DataFrame({"amount": amount, "country": country})


async def _scan(identity, source_id, amount, country, *, null_rates=None):
    frame = _frame(amount, country)
    async with user_session(identity.user_id) as db:
        return await prepare_and_update_column_baselines(
            db,
            identity.org_id,
            source_id,
            frame,
            "polars",
            SCHEMA,
            row_count=len(amount),
            null_rates=null_rates or {"amount": 0.0, "country": 0.0},
        )


async def test_a_cold_start_column_never_produces_a_finding(identity):
    source_id = await _seed_source(identity)
    _, findings = await _scan(identity, source_id, [100.0, 100.0, 100.0], ["US", "US", "US"])
    assert findings == []

    # Even wildly different from the (short) history so far.
    _, findings = await _scan(identity, source_id, [999_999.0], ["ZZ"])
    assert findings == []


async def test_a_stable_numeric_column_flags_a_sharp_deviation_once_calibrated(identity):
    source_id = await _seed_source(identity)
    for _ in range(MIN_SAMPLE_SIZE):
        await _scan(identity, source_id, [100.0, 100.0], ["US", "US"])

    _, findings = await _scan(identity, source_id, [500.0, 500.0], ["US", "US"])
    value_findings = [f for f in findings if f["column"] == "amount"]
    assert len(value_findings) == 1
    assert value_findings[0]["baseline_kind"] == "numeric"


async def test_a_calibrated_categorical_column_flags_a_new_category_at_volume(identity):
    source_id = await _seed_source(identity)
    for _ in range(MIN_SAMPLE_SIZE):
        # amount and country must be the same length - one DataFrame.
        await _scan(identity, source_id, [1.0] * 20, ["US", "US", "US", "DE"] * 5)

    _, findings = await _scan(identity, source_id, [1.0] * 6, ["US", "US", "BR", "BR", "BR", "BR"])
    country_findings = [f for f in findings if f["column"] == "country"]
    assert any(f["shift_kind"] == "new_category" and f["category"] == "BR" for f in country_findings)


async def test_baseline_kind_stays_fixed_once_a_column_has_a_row(identity):
    # A small scan's own cardinality ratio is noisy — 2 distinct values out
    # of 6 rows (0.33) crosses the categorical/freeform_string boundary
    # (0.2) that the SAME column's first, larger scan (2 of 20 -> 0.1) did
    # not. Reclassifying every scan would flip this column to
    # freeform_string here and silently discard its whole categorical
    # calibration history, since a numeric-shaped model has no frequencies
    # a categorical comparison could ever read back.
    source_id = await _seed_source(identity)
    for _ in range(MIN_SAMPLE_SIZE):
        await _scan(identity, source_id, [1.0] * 20, ["US", "US", "US", "DE"] * 5)

    async with user_session(identity.user_id) as db:
        row = (
            await db.execute(
                text("select baseline_kind from public.column_baselines where source_id = :source and column_name = 'country'"),
                {"source": source_id},
            )
        ).mappings().first()
    assert row["baseline_kind"] == "categorical"

    await _scan(identity, source_id, [1.0] * 6, ["US", "BR", "BR", "BR", "BR", "BR"])

    async with user_session(identity.user_id) as db:
        row = (
            await db.execute(
                text("select baseline_kind from public.column_baselines where source_id = :source and column_name = 'country'"),
                {"source": source_id},
            )
        ).mappings().first()
    assert row["baseline_kind"] == "categorical"


async def test_null_rate_baselines_returned_are_the_state_before_this_scan(identity):
    source_id = await _seed_source(identity)
    for _ in range(MIN_SAMPLE_SIZE):
        await _scan(identity, source_id, [1.0, 2.0], ["US", "US"], null_rates={"amount": 0.02, "country": 0.0})

    before, _ = await _scan(
        identity, source_id, [1.0, 2.0], ["US", "US"], null_rates={"amount": 0.50, "country": 0.0}
    )
    # The baseline handed back must reflect the calibrated ~2% history, not
    # the 50% this exact call just observed — detect_drift compares the new
    # reading against this, and a baseline that already absorbed the new
    # reading could never flag it.
    assert before["amount"].is_calibrated()
    assert before["amount"].mean == pytest.approx(0.02, abs=1e-6)
    assert before["amount"].is_within_band(0.50) is False


async def test_an_override_widens_the_band_for_one_column(identity):
    source_id = await _seed_source(identity)
    # Alternating 50/150 gives this column real natural variance to widen -
    # a perfectly flat history (identical value every scan) has a
    # zero-width band that no z multiplier can widen, since z times zero is
    # still zero.
    for i in range(MIN_SAMPLE_SIZE):
        value = 50.0 if i % 2 == 0 else 150.0
        await _scan(identity, source_id, [value, value], ["US", "US"])

    async with user_session(identity.user_id) as db:
        await db.execute(
            text(
                "update public.column_baselines set override = '{\"z\": 20.0}'::jsonb "
                "where source_id = :source and column_name = 'amount'"
            ),
            {"source": source_id},
        )

    # Same deviation that test_a_stable_numeric_column_flags_a_sharp_deviation
    # confirms *does* get flagged at the default z - must not, at z=20.
    _, findings = await _scan(identity, source_id, [500.0, 500.0], ["US", "US"])
    assert [f for f in findings if f["column"] == "amount"] == []


async def test_source_row_count_delta_flags_a_stalled_source(identity):
    source_id = await _seed_source(identity)
    async with user_session(identity.user_id) as db:
        row_count = 100
        # The very first call only ever seeds `last_row_count` - there is
        # nothing yet to diff against, so it contributes no delta
        # observation. MIN_SAMPLE_SIZE + 1 calls are needed for
        # MIN_SAMPLE_SIZE actual deltas to have been observed.
        for _ in range(MIN_SAMPLE_SIZE + 1):
            row_count += 120
            finding = await compute_and_remember_source_baseline(db, identity.org_id, source_id, row_count)
        assert finding is None  # still calibrating, or exactly on-trend

        # The upstream job stops: row count does not move at all.
        finding = await compute_and_remember_source_baseline(db, identity.org_id, source_id, row_count)
    assert finding is not None
    assert finding["observed_delta"] == 0
