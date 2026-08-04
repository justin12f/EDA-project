"""The agent loop's guarantees: it terminates, it survives tool failure, it reports."""

from __future__ import annotations

import json

import pytest

from lumen.agents.loop import AgentLoop, CollectingSink
from lumen.llm.base import ChatMessage, LLMResponse, TokenUsage, ToolCall, ToolSpec
from lumen.llm.mock_provider import MockProvider

SPEC = ToolSpec(name="t", description="d", input_schema={"type": "object", "properties": {}})


class ScriptedProvider:
    """Replays a list of responses, one per call, then repeats the last."""

    model = "test-model"

    def __init__(self, script: list[LLMResponse]) -> None:
        self._script = list(script)
        self.calls = 0
        self.last_messages: list[ChatMessage] = []

    async def complete(self, messages, tools, max_tokens=4096, temperature=0.0):
        self.calls += 1
        self.last_messages = list(messages)
        return self._script.pop(0) if len(self._script) > 1 else self._script[0]


class StubRegistry:
    def __init__(self, result: dict | None = None, raises: Exception | None = None) -> None:
        self.invocations: list[tuple[str, dict]] = []
        self._result = result or {"ok": True, "data": {"x": 1}}
        self._raises = raises

    def specs(self):
        return [SPEC]

    async def invoke(self, name, arguments):
        self.invocations.append((name, arguments))
        if self._raises is not None:
            raise self._raises
        return self._result


def text(message: str) -> LLMResponse:
    return LLMResponse(
        text=message, tool_calls=[], usage=TokenUsage(10, 5, "test-model"), stop_reason="end_turn"
    )


def call(name: str = "t", arguments: dict | None = None) -> LLMResponse:
    return LLMResponse(
        text=None,
        tool_calls=[ToolCall(id="tc_1", name=name, arguments=arguments or {})],
        usage=TokenUsage(10, 5, "test-model"),
        stop_reason="tool_use",
    )


# ── the happy path ──────────────────────────────────────────────────────────


async def test_a_plain_answer_finishes_in_one_iteration():
    result = await AgentLoop(ScriptedProvider([text("done")]), StubRegistry()).run("sys", "user")
    assert result.final_text == "done"
    assert result.iterations == 1
    assert result.stop_reason == "done"


async def test_a_tool_call_is_executed_and_fed_back():
    registry = StubRegistry()
    loop = AgentLoop(ScriptedProvider([call("t", {"a": 1}), text("finished")]), registry)

    result = await loop.run("sys", "user")

    assert registry.invocations == [("t", {"a": 1})]
    assert result.final_text == "finished"
    assert result.iterations == 2


async def test_the_tool_result_reaches_the_model_as_a_tool_message():
    provider = ScriptedProvider([call(), text("ok")])
    await AgentLoop(provider, StubRegistry({"ok": True, "data": {"rows": 42}})).run("sys", "user")

    tool_messages = [m for m in provider.last_messages if m.role == "tool"]
    assert len(tool_messages) == 1
    assert json.loads(tool_messages[0].content)["data"]["rows"] == 42
    assert tool_messages[0].is_error is False


# ── termination ─────────────────────────────────────────────────────────────


async def test_the_iteration_cap_stops_a_runaway_loop():
    loop = AgentLoop(ScriptedProvider([call()]), StubRegistry(), max_iterations=3)
    result = await loop.run("sys", "user")

    assert result.iterations == 3
    assert result.stop_reason == "max_iterations"


async def test_the_token_budget_stops_the_loop():
    loop = AgentLoop(
        ScriptedProvider([call()]), StubRegistry(), max_iterations=50, max_total_tokens=30
    )
    result = await loop.run("sys", "user")

    assert result.stop_reason == "budget"
    assert result.total_tokens >= 30


async def test_the_deadline_stops_the_loop():
    loop = AgentLoop(
        ScriptedProvider([call()]), StubRegistry(), max_iterations=50, deadline_seconds=-1
    )
    result = await loop.run("sys", "user")
    assert result.stop_reason == "deadline"


# ── failure is data, not a crash ────────────────────────────────────────────


async def test_a_failing_tool_is_returned_to_the_model_not_raised():
    loop = AgentLoop(
        ScriptedProvider([call(), text("recovered")]),
        StubRegistry({"ok": False, "error": "boom"}),
    )
    result = await loop.run("sys", "user")
    assert result.final_text == "recovered"


async def test_a_tool_that_raises_becomes_an_error_result():
    provider = ScriptedProvider([call(), text("recovered")])
    loop = AgentLoop(provider, StubRegistry(raises=RuntimeError("disk on fire")))

    result = await loop.run("sys", "user")

    tool_messages = [m for m in provider.last_messages if m.role == "tool"]
    payload = json.loads(tool_messages[0].content)
    assert payload["ok"] is False
    assert "disk on fire" in payload["error"]
    assert tool_messages[0].is_error is True
    assert result.final_text == "recovered"


async def test_an_unknown_tool_name_does_not_crash_the_loop():
    class Rejecting(StubRegistry):
        async def invoke(self, name, arguments):
            return {"ok": False, "error": f"Unknown tool '{name}'"}

    loop = AgentLoop(ScriptedProvider([call("ghost"), text("ok")]), Rejecting())
    assert (await loop.run("sys", "user")).final_text == "ok"


# ── observability ───────────────────────────────────────────────────────────


async def test_every_step_reaches_the_sink_in_order():
    sink = CollectingSink()
    loop = AgentLoop(ScriptedProvider([call(), text("done")]), StubRegistry(), sink=sink)

    await loop.run("sys", "user")

    assert [kind for kind, _ in sink.events] == ["tool_call", "tool_result", "message"]
    assert sink.events[0][1]["name"] == "t"
    assert sink.events[1][1]["ok"] is True


async def test_usage_is_recorded_once_per_model_call():
    sink = CollectingSink()
    await AgentLoop(ScriptedProvider([call(), text("done")]), StubRegistry(), sink=sink).run(
        "sys", "user"
    )

    assert len(sink.usage) == 2
    assert all(u.model == "test-model" for u in sink.usage)


async def test_the_loop_runs_with_no_sink_configured():
    """The default sink discards, so the loop works anywhere — including a script."""
    result = await AgentLoop(ScriptedProvider([text("fine")]), StubRegistry()).run("sys", "user")
    assert result.final_text == "fine"


# ── the keyless path, end to end ────────────────────────────────────────────


async def test_the_mock_provider_drives_a_full_profile_to_proposal_run():
    """No API key, no network: the loop reaches a validated proposal on real numbers."""
    profile = {
        "ok": True,
        "data": {
            "rid": "ab12cd34",
            "row_count": 5,
            "columns": {"id": "int64", "country_code": "str", "email_hash": "str"},
            "null_rate_by_column": {"id": 0.0, "country_code": 0.4, "email_hash": 0.0},
            "duplicate_counts": {"email_hash": 1},
        },
    }
    proposed: dict = {}

    class Registry:
        def specs(self):
            return [
                ToolSpec(
                    name="profile_source",
                    description="profile",
                    input_schema={"type": "object", "properties": {"rid": {"type": "string"}}},
                ),
                ToolSpec(
                    name="propose_cleaning_pipeline",
                    description="propose",
                    input_schema={"type": "object", "properties": {"rid": {"type": "string"}}},
                ),
            ]

        async def invoke(self, name, arguments):
            if name == "profile_source":
                return profile
            proposed.update(arguments)
            return {"ok": True, "data": dict(arguments)}

    sink = CollectingSink()
    result = await AgentLoop(MockProvider(), Registry(), sink=sink).run(
        "You are the Cleaning Agent.", "Propose a cleaning pipeline for dataset ab12cd34."
    )

    assert result.stop_reason == "done"
    steps = proposed["steps"]
    assert {
        "impute_categorical": {"columns": ["country_code"], "strategy": "mode"}
    } in steps
    assert {"remove_duplicates_rows": {}} in steps
    assert "40.0%" in proposed["rationale"]
    assert [kind for kind, _ in sink.events].count("tool_call") == 2


@pytest.mark.parametrize("iterations", [1, 5, 12])
async def test_the_loop_never_exceeds_its_iteration_cap(iterations):
    loop = AgentLoop(ScriptedProvider([call()]), StubRegistry(), max_iterations=iterations)
    assert (await loop.run("sys", "user")).iterations == iterations
