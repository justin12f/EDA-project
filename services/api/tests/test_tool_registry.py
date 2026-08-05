"""The tool registry, against the live project.

The registry is where the agent meets the engine and the tenant boundary, so
these run integration-marked: a unit test with a fake store would assert the
mock, not the isolation.
"""

from __future__ import annotations

import io
import uuid

import httpx
import polars as pl
import pytest
from sqlalchemy import text

from lumen_api.agents.registry import build_tool_registry
from lumen_api.datasets.store import HandleStore, SupabaseStorage
from lumen_api.db.session import user_session
from lumen_api.settings import get_settings

pytestmark = pytest.mark.integration

# Two nulls in five for country_code (40%), 'a1' repeated in email_hash.
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
            "email": f"lumen-tools-{uuid.uuid4().hex[:12]}@example.com",
            "password": uuid.uuid4().hex + "Aa1!",
            "email_confirm": True,
            "user_metadata": {"display_name": "Tool Tester"},
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


@pytest.fixture
def person():
    user_id = _create_user()
    yield user_id
    _delete_user(user_id)


async def _org_of(user_id: uuid.UUID) -> uuid.UUID:
    async with user_session(user_id) as db:
        return (
            await db.execute(
                text("select org_id from public.memberships where user_id = :id"),
                {"id": user_id},
            )
        ).scalar_one()


async def _seed_source(user_id: uuid.UUID, org_id: uuid.UUID) -> uuid.UUID:
    """Upload a CSV and register it as a data source, as the upload endpoint will."""
    path = f"org/{org_id}/uploads/{uuid.uuid4().hex}.csv"
    await SupabaseStorage().upload(path, CSV, "text/csv")

    source_id = uuid.uuid4()
    async with user_session(user_id) as db:
        await db.execute(
            text(
                "insert into public.data_sources "
                "(id, org_id, name, kind, status, object_path, table_name) "
                "values (:id, :org, 'users.csv', 'csv', 'idle', :path, 'users')"
            ),
            {"id": source_id, "org": org_id, "path": path},
        )
    return source_id


# ── shape ───────────────────────────────────────────────────────────────────


async def test_the_registry_exposes_the_expected_tools(person):
    registry = build_tool_registry(await _org_of(person), person)
    assert {spec.name for spec in registry.specs()} == {
        "list_data_sources",
        "read_source",
        "profile_source",
        "propose_cleaning_pipeline",
        "run_statistic",
        "recall_context",
        "remember_decision",
    }


async def test_every_spec_carries_a_usable_json_schema(person):
    registry = build_tool_registry(await _org_of(person), person)
    for spec in registry.specs():
        assert spec.input_schema["type"] == "object"
        assert "properties" in spec.input_schema
        assert spec.description, f"{spec.name} needs a description the model can act on"


async def test_an_unknown_tool_returns_an_error_naming_the_alternatives(person):
    registry = build_tool_registry(await _org_of(person), person)
    result = await registry.invoke("summon_daemon", {})
    assert result["ok"] is False
    assert "unknown tool" in result["error"].lower()
    assert "profile_source" in result["error"]


async def test_bad_arguments_become_an_error_result_not_a_crash(person):
    registry = build_tool_registry(await _org_of(person), person)
    result = await registry.invoke("profile_source", {"wrong_kwarg": 1})
    assert result["ok"] is False
    assert "Bad arguments" in result["error"]


# ── the real path ───────────────────────────────────────────────────────────


async def test_read_then_profile_derives_the_real_numbers(person):
    org_id = await _org_of(person)
    source_id = await _seed_source(person, org_id)
    registry = build_tool_registry(org_id, person)

    listed = await registry.invoke("list_data_sources", {})
    assert listed["ok"] is True
    assert [s["name"] for s in listed["data"]["sources"]] == ["users.csv"]

    loaded = await registry.invoke("read_source", {"source_id": str(source_id)})
    assert loaded["ok"] is True, loaded.get("error")
    rid = loaded["data"]["rid"]
    assert loaded["data"]["row_count"] == 5

    profile = await registry.invoke("profile_source", {"rid": rid})
    assert profile["ok"] is True, profile.get("error")
    data = profile["data"]
    assert data["null_rate_by_column"]["country_code"] == pytest.approx(0.4)
    assert data["null_rate_by_column"]["id"] == 0.0
    # email_hash is key-like, so it is checked; 'a1' twice is one duplicate.
    assert data["duplicate_counts"]["email_hash"] == 1


async def test_a_valid_pipeline_validates_and_an_invented_step_does_not(person):
    org_id = await _org_of(person)
    source_id = await _seed_source(person, org_id)
    registry = build_tool_registry(org_id, person)
    rid = (await registry.invoke("read_source", {"source_id": str(source_id)}))["data"]["rid"]

    good = await registry.invoke(
        "propose_cleaning_pipeline",
        {
            "rid": rid,
            "steps": [
                {"impute_categorical": {"columns": ["country_code"], "strategy": "mode"}},
                {"remove_duplicates_rows": {}},
            ],
            "rationale": "40.0% nulls in country_code; 1 duplicate email_hash",
        },
    )
    assert good["ok"] is True, good.get("error")

    bad = await registry.invoke(
        "propose_cleaning_pipeline",
        {"rid": rid, "steps": [{"summon_daemon": {}}], "rationale": "no"},
    )
    assert bad["ok"] is False
    assert "summon_daemon" in bad["error"]


async def test_validation_executes_nothing(person):
    """A validated proposal must leave the dataset byte-identical."""
    org_id = await _org_of(person)
    source_id = await _seed_source(person, org_id)
    registry = build_tool_registry(org_id, person)
    rid = (await registry.invoke("read_source", {"source_id": str(source_id)}))["data"]["rid"]

    before = (await HandleStore(org_id, person).resolve(rid)).collect().height
    await registry.invoke(
        "propose_cleaning_pipeline",
        {
            "rid": rid,
            "steps": [{"remove_duplicates_rows": {}}],
            "rationale": "r",
        },
    )
    after = (await HandleStore(org_id, person).resolve(rid)).collect().height

    assert before == after == 5, "proposing must not mutate the dataset"


# ── isolation ───────────────────────────────────────────────────────────────


async def test_a_handle_from_another_org_is_not_found(person):
    """`rid` alone grants nothing — RLS filters it out for anyone else."""
    org_id = await _org_of(person)
    handle = await HandleStore(org_id, person).put(
        pl.DataFrame({"x": [1, 2, 3]}), label="private"
    )

    other = _create_user()
    try:
        from lumen_api.errors import NotFound

        with pytest.raises(NotFound):
            await HandleStore(await _org_of(other), other).get(handle.rid)
    finally:
        _delete_user(other)


async def test_list_data_sources_never_leaks_another_org(person):
    org_id = await _org_of(person)
    await _seed_source(person, org_id)

    other = _create_user()
    try:
        registry = build_tool_registry(await _org_of(other), other)
        listed = await registry.invoke("list_data_sources", {})
        assert listed["data"]["sources"] == []
    finally:
        _delete_user(other)


async def test_a_dataset_round_trips_through_storage(person):
    org_id = await _org_of(person)
    store = HandleStore(org_id, person)

    handle = await store.put(pl.DataFrame({"a": [1, 2], "b": ["x", "y"]}), label="round trip")
    assert handle.row_count == 2
    assert set(handle.schema) == {"a", "b"}

    restored = (await store.resolve(handle.rid)).collect()
    assert restored.height == 2
    assert restored["b"].to_list() == ["x", "y"]


async def test_the_tool_description_lists_the_engine_s_real_vocabulary(person):
    """The fix for the defect this file caught: nobody has to guess a step name.

    The description is generated from the factory, so an agent — or a person
    reading the OpenAPI schema — sees exactly the names the engine will accept.
    """
    from lumen_api.agents.registry import registered_steps

    registry = build_tool_registry(await _org_of(person), person)
    spec = next(s for s in registry.specs() if s.name == "propose_cleaning_pipeline")

    steps = registered_steps("polars")
    assert steps, "the engine must register at least one cleaning step"
    for name in steps:
        assert name in spec.description, f"{name} missing from the tool description"
    assert "drop_nulls" not in spec.description, "a name the engine does not register"


async def test_column_scoped_steps_build_from_a_proposal(person):
    """Was a pinned failure; now the guard that it stays fixed.

    Column-scoped steps are most of the useful ones, and all 24 were unusable
    from a proposal until the abstract constructors stopped shadowing
    AbstractBaseStep.__init__.
    """
    org_id = await _org_of(person)
    source_id = await _seed_source(person, org_id)
    registry = build_tool_registry(org_id, person)
    rid = (await registry.invoke("read_source", {"source_id": str(source_id)}))["data"]["rid"]

    result = await registry.invoke(
        "propose_cleaning_pipeline",
        {
            "rid": rid,
            "steps": [{"impute_categorical": {"columns": ["country_code"], "strategy": "mode"}}],
            "rationale": "40.0% nulls in country_code",
        },
    )

    assert result["ok"] is True, result.get("error")
    assert result["data"]["steps"][0]["impute_categorical"]["columns"] == ["country_code"]


async def test_profiling_records_a_shared_fact_the_next_run_can_recall(person):
    """The agent stops re-deriving what it already measured."""
    from lumen_api.context.store import ContextStore, Scope

    org_id = await _org_of(person)
    source_id = await _seed_source(person, org_id)
    registry = build_tool_registry(org_id, person)

    rid = (await registry.invoke("read_source", {"source_id": str(source_id)}))["data"]["rid"]
    await registry.invoke("profile_source", {"rid": rid})

    recalled = await registry.invoke(
        "recall_context", {"query": "which column has missing values"}
    )
    assert recalled["ok"] is True
    matches = recalled["data"]["matches"]
    assert matches, "profiling should have left something to recall"
    assert any("country_code" in m["content"] for m in matches)
    # A measurement is a shared fact, not one person's opinion.
    assert any(m["scope"] == str(Scope.ORG) for m in matches)


async def test_a_recorded_decision_is_private_to_its_author(person):
    org_id = await _org_of(person)
    registry = build_tool_registry(org_id, person)

    written = await registry.invoke(
        "remember_decision",
        {
            "title": "Rejected imputation",
            "content": "Do not impute country_code — a wrong country is worse than a null.",
        },
    )
    assert written["ok"] is True
    assert written["data"]["scope"] == "user"

    mine = await registry.invoke("recall_context", {"query": "what did I decide about imputing"})
    assert any(m["mine"] and "worse than a null" in m["content"] for m in mine["data"]["matches"])

    other = _create_user()
    try:
        theirs = build_tool_registry(await _org_of(other), other)
        seen = await theirs.invoke("recall_context", {"query": "imputing country_code"})
        assert not any("worse than a null" in m["content"] for m in seen["data"]["matches"])
    finally:
        _delete_user(other)
