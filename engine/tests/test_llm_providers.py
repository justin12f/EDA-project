"""The LLM layer, including the keyless path the whole product depends on."""

from __future__ import annotations

import json

import pytest

from lumen.llm.base import ChatMessage, LLMResponse, TokenUsage, ToolCall, ToolSpec
from lumen.llm.mock_provider import MockProvider
from lumen.llm.registry import ModelTiers, get_provider, resolve_mode

TOOLS = [
    ToolSpec(
        name="read_source",
        description="load a source",
        input_schema={"type": "object", "properties": {"source_id": {"type": "string"}}},
    ),
    ToolSpec(
        name="profile_source",
        description="profile a dataset",
        input_schema={"type": "object", "properties": {"rid": {"type": "string"}}},
    ),
    ToolSpec(
        name="propose_cleaning_pipeline",
        description="propose steps",
        input_schema={"type": "object", "properties": {"rid": {"type": "string"}}},
    ),
]


def tool_result(call_id: str, payload: dict) -> ChatMessage:
    return ChatMessage(role="tool", tool_call_id=call_id, content=json.dumps(payload))


# ── MockProvider ────────────────────────────────────────────────────────────


async def test_first_move_is_read_source_with_the_id_from_the_prompt():
    response = await MockProvider().complete(
        [
            ChatMessage(role="system", content="You are the Cleaning Agent."),
            ChatMessage(
                role="user",
                content="Propose a cleaning pipeline for source "
                "4a1f2c9e-0000-4000-8000-000000000001.",
            ),
        ],
        TOOLS,
    )

    assert len(response.tool_calls) == 1
    call = response.tool_calls[0]
    assert call.name == "read_source"
    assert call.arguments == {"source_id": "4a1f2c9e-0000-4000-8000-000000000001"}
    assert response.stop_reason == "tool_use"


async def test_second_move_profiles_the_handle_the_first_move_returned():
    messages = [
        ChatMessage(role="user", content="Propose a cleaning pipeline for source s1."),
        ChatMessage(
            role="assistant",
            content="",
            tool_calls=[ToolCall(id="c1", name="read_source", arguments={"source_id": "s1"})],
        ),
        tool_result("c1", {"ok": True, "data": {"rid": "ab12cd34", "row_count": 5}}),
    ]
    response = await MockProvider().complete(messages, TOOLS)

    assert response.tool_calls[0].name == "profile_source"
    assert response.tool_calls[0].arguments == {"rid": "ab12cd34"}


async def test_steps_are_derived_from_the_observed_null_rates():
    profile = {
        "ok": True,
        "data": {
            "rid": "ab12cd34",
            "row_count": 1000,
            "columns": {"id": "int64", "country_code": "str", "note": "str"},
            "null_rate_by_column": {"id": 0.0, "country_code": 0.032, "note": 0.001},
            "duplicate_counts": {"email_hash": 412},
        },
    }
    response = await MockProvider().complete(
        [
            ChatMessage(role="user", content="Propose a cleaning pipeline for dataset ab12cd34."),
            tool_result("c2", profile),
        ],
        TOOLS,
    )

    call = response.tool_calls[0]
    assert call.name == "propose_cleaning_pipeline"
    steps = call.arguments["steps"]

    # country_code (3.2%) clears the 0.5% threshold; note (0.1%) and id (0%) do not.
    # The step names are the engine's own — asserted against the registry below,
    # because a proposal the engine cannot build is worthless however good it reads.
    assert {
        "impute_categorical": {"columns": ["country_code"], "strategy": "mode"}
    } in steps
    assert not any("note" in json.dumps(step) for step in steps)
    assert {"remove_duplicates_rows": {}} in steps
    # The rationale cites the number it actually saw.
    assert "3.2%" in call.arguments["rationale"]


async def test_a_clean_dataset_produces_no_proposal_and_says_so():
    profile = {
        "ok": True,
        "data": {
            "rid": "ab12cd34",
            "row_count": 1000,
            "columns": {"id": "int64"},
            "null_rate_by_column": {"id": 0.0},
            "duplicate_counts": {},
        },
    }
    response = await MockProvider().complete(
        [
            ChatMessage(role="user", content="Propose a cleaning pipeline for dataset ab12cd34."),
            tool_result("c2", profile),
        ],
        TOOLS,
    )

    assert response.tool_calls == []
    assert response.stop_reason == "end_turn"
    assert "no material" in (response.text or "").lower()


async def test_after_a_successful_proposal_it_finishes_with_a_summary():
    messages = [
        ChatMessage(role="user", content="Propose a cleaning pipeline for dataset ab12cd34."),
        tool_result("c2", {"ok": True, "data": {"null_rate_by_column": {"c": 0.03}, "row_count": 10}}),
        tool_result(
            "c3",
            {
                "ok": True,
                "data": {
                    "rid": "ab12cd34",
                    "steps": [{"drop_nulls": {"columns": ["c"]}}],
                    "rationale": "3.0% nulls in c",
                },
            },
        ),
    ]
    response = await MockProvider().complete(messages, TOOLS)

    assert response.tool_calls == []
    assert response.stop_reason == "end_turn"
    assert response.text and "1 cleaning step" in response.text


async def test_a_run_of_failures_stops_instead_of_retrying_forever():
    messages = [ChatMessage(role="user", content="Propose a cleaning pipeline for dataset ab12cd34.")]
    for index in range(4):
        messages.append(tool_result(f"c{index}", {"ok": False, "error": "boom"}))

    response = await MockProvider().complete(messages, TOOLS)

    assert response.tool_calls == []
    assert "could not" in (response.text or "").lower()


async def test_usage_is_reported_and_nonzero():
    response = await MockProvider().complete(
        [ChatMessage(role="user", content="Profile source s1.")], TOOLS
    )
    assert response.usage.model == "mock-agent-v1"
    assert response.usage.input_tokens > 0
    assert response.usage.output_tokens > 0
    assert response.usage.total == response.usage.input_tokens + response.usage.output_tokens


async def test_it_never_calls_a_tool_that_was_not_offered():
    response = await MockProvider().complete(
        [ChatMessage(role="user", content="Propose a cleaning pipeline for dataset ab12cd34.")],
        tools=[],
    )
    assert response.tool_calls == []


async def test_it_is_deterministic():
    messages = [ChatMessage(role="user", content="Propose a cleaning pipeline for source s1.")]
    first = await MockProvider().complete(messages, TOOLS)
    second = await MockProvider().complete(messages, TOOLS)
    assert first.tool_calls == second.tool_calls
    assert first.text == second.text


# ── Registry ────────────────────────────────────────────────────────────────


def test_auto_falls_back_to_mock_with_no_credentials():
    assert resolve_mode("auto", anthropic_key=None, groq_key=None) == "mock"
    provider = get_provider("specialist", anthropic_key=None, groq_key=None, mode="auto")
    assert isinstance(provider, MockProvider)


def test_auto_prefers_anthropic_then_groq():
    assert resolve_mode("auto", anthropic_key="sk-ant-x", groq_key="gsk_y") == "anthropic"
    assert resolve_mode("auto", anthropic_key=None, groq_key="gsk_y") == "groq"


def test_auto_uses_anthropic_when_a_key_is_present():
    from lumen.llm.anthropic_provider import AnthropicProvider

    provider = get_provider(
        "specialist", anthropic_key="sk-ant-test", groq_key=None, mode="auto"
    )
    assert isinstance(provider, AnthropicProvider)
    assert provider.model == ModelTiers().specialist


def test_reasoning_and_specialist_tiers_map_to_different_models():
    reasoning = get_provider("reasoning", anthropic_key="sk-ant-test", groq_key=None)
    specialist = get_provider("specialist", anthropic_key="sk-ant-test", groq_key=None)
    assert reasoning.model == "claude-opus-5"
    assert specialist.model == "claude-sonnet-5"


def test_an_explicit_mode_never_silently_degrades():
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        get_provider("specialist", anthropic_key=None, groq_key="gsk_y", mode="anthropic")

    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        get_provider("fast", anthropic_key="sk-ant-x", groq_key=None, mode="groq")


# ── AnthropicProvider wire format ───────────────────────────────────────────


class _Block:
    def __init__(self, type_: str, **attributes):
        self.type = type_
        for key, value in attributes.items():
            setattr(self, key, value)


class _Reply:
    def __init__(self, content, stop_reason="end_turn", input_tokens=10, output_tokens=5):
        self.content = content
        self.stop_reason = stop_reason
        self.usage = type(
            "U", (), {"input_tokens": input_tokens, "output_tokens": output_tokens}
        )()


class _FakeAnthropic:
    def __init__(self, reply):
        self._reply = reply
        self.last_kwargs: dict | None = None
        self.messages = self

    async def create(self, **kwargs):
        self.last_kwargs = kwargs
        return self._reply


async def test_anthropic_returns_text_and_usage():
    from lumen.llm.anthropic_provider import AnthropicProvider

    client = _FakeAnthropic(_Reply([_Block("text", text="hello")]))
    response = await AnthropicProvider(client=client, model="claude-sonnet-5").complete(
        [ChatMessage(role="user", content="hi")], tools=[], max_tokens=64
    )

    assert isinstance(response, LLMResponse)
    assert response.text == "hello"
    assert response.tool_calls == []
    assert response.usage == TokenUsage(10, 5, "claude-sonnet-5")


async def test_anthropic_extracts_tool_calls():
    from lumen.llm.anthropic_provider import AnthropicProvider

    block = _Block("tool_use", id="tu_1", name="read_source", input={"source_id": "abc"})
    client = _FakeAnthropic(_Reply([block], stop_reason="tool_use"))

    response = await AnthropicProvider(client=client, model="claude-sonnet-5").complete(
        [ChatMessage(role="user", content="read it")], tools=TOOLS, max_tokens=64
    )

    assert response.stop_reason == "tool_use"
    assert response.tool_calls == [
        ToolCall(id="tu_1", name="read_source", arguments={"source_id": "abc"})
    ]


async def test_anthropic_moves_system_messages_to_the_system_parameter():
    from lumen.llm.anthropic_provider import AnthropicProvider

    client = _FakeAnthropic(_Reply([_Block("text", text="ok")]))
    await AnthropicProvider(client=client, model="claude-sonnet-5").complete(
        [
            ChatMessage(role="system", content="You are Lumen."),
            ChatMessage(role="user", content="hi"),
        ],
        tools=[],
        max_tokens=64,
    )

    assert client.last_kwargs["system"] == "You are Lumen."
    assert [m["role"] for m in client.last_kwargs["messages"]] == ["user"]


async def test_anthropic_encodes_a_tool_result_as_a_user_block():
    from lumen.llm.anthropic_provider import AnthropicProvider

    client = _FakeAnthropic(_Reply([_Block("text", text="ok")]))
    await AnthropicProvider(client=client, model="claude-sonnet-5").complete(
        [
            ChatMessage(role="user", content="go"),
            ChatMessage(
                role="assistant",
                tool_calls=[ToolCall(id="tu_1", name="read_source", arguments={})],
            ),
            ChatMessage(role="tool", tool_call_id="tu_1", content='{"ok": true}'),
        ],
        tools=[],
        max_tokens=64,
    )

    sent = client.last_kwargs["messages"]
    assert sent[-1]["role"] == "user"
    assert sent[-1]["content"][0]["type"] == "tool_result"
    assert sent[-1]["content"][0]["tool_use_id"] == "tu_1"


def test_every_step_the_mock_proposes_is_registered_in_the_engine():
    """The guard against the defect this test file previously encoded.

    MockProvider used to propose `drop_nulls` and `drop_duplicates` — names that
    read naturally and that nothing has ever registered. The proposal validated
    nowhere and failed at the tool boundary. Assert against the registry so the
    two can never drift again.
    """
    from lumen.data_cleaning.step_factory import AbstractDataCleaningStepFactory as Factory
    from lumen.llm.mock_provider import _plan

    profile = {
        "row_count": 1000,
        "null_rate_by_column": {"a": 0.4, "b": 0.0},
        "duplicate_counts": {"c": 3},
    }
    steps, _ = _plan(profile, 0.005)
    assert steps, "a dirty profile must produce steps"

    for step in steps:
        (name,) = step.keys()
        assert Factory.is_registered(name, "polars"), f"{name} is not a registered step"
        assert Factory.is_registered(name, "pandas"), f"{name} is not registered for pandas"
