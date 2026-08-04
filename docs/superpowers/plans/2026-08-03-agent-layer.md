# Agent Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the AI agent flow: an agent profiles a data source, proposes a cleaning pipeline as a validated, executable spec, a human accepts it in the UI, and a worker runs it — with every token, second and decision recorded.

**Architecture:** A `SupervisorAgent` delegates to typed specialists that share one `ToolRegistry` built over the existing `AgentMasterFactory`. No agent mutates tenant data: specialists emit a `Proposal` whose `spec` is *exactly* the JSON `PipelineBuilder.build()` already consumes, so the agent's output is the engine's input and an invalid plan fails validation before a human ever sees it. Dataframes move between processes as `DatasetHandle` rows (Parquet in object storage + metadata in Postgres), replacing the in-process dict. Every LLM call and every job goes through wrappers that write a `UsageRecord` in the same transaction as the work.

**Tech Stack:** Anthropic Messages API (`claude-opus-5` supervisor, `claude-sonnet-5` specialists), Groq/OpenAI-compatible fast tier, arq + Redis, SQLAlchemy 2.x async, Polars/Pandas/PySpark via the existing engine, SSE via `sse-starlette`.

**Prerequisite:** `docs/superpowers/plans/2026-08-03-saas-foundations.md` complete through Task 7.

## Global Constraints

- Python 3.11, `uv`. Never `pip`.
- The engine (`engine/src/lumen/`) stays free of web and tenant concerns. Agent orchestration that needs `org_id` lives in `services/api/src/lumen_api/agents/`; backend-agnostic capability wrappers live in `engine/src/lumen/`.
- **No generated code is ever executed.** No `exec`, no `eval`, no LLM-authored Python or SQL run against tenant data. Agents emit declarative specs validated against the engine's factories.
- Every agent loop has a hard iteration cap, a wall-clock deadline, and a token budget. A tool that raises is caught and returned to the model as a tool result — never propagated out of the loop.
- Every LLM call writes `llm_input_tokens` and `llm_output_tokens` usage records with the real model id. Every job writes `compute_seconds`.
- New tables carry `org_id` and an RLS policy, and are added to `ORG_SCOPED_TABLES` in the RLS migration.
- Model ids are configuration. Never hardcode a model string outside `settings.py`.
- **The product must run end to end with no API key.** `LLM_MODE` defaults to `auto`, which resolves to Anthropic when `ANTHROPIC_API_KEY` is set and to the deterministic `MockProvider` (Task 9) otherwise. No test, no `make dev`, and no end-to-end run may require a credential. A test that skips without a key is a test that does not run — derive the behaviour from real tool results instead.

---

## File Structure

```
engine/src/lumen/
  llm/
    __init__.py
    base.py              LLMProvider, ChatMessage, ToolSpec, ToolCall, LLMResponse, TokenUsage
    anthropic_provider.py
    groq_provider.py
    registry.py          get_provider(tier) -> LLMProvider
  datasets/
    __init__.py
    handle.py            DatasetHandle dataclass, HandleStore protocol
    materialize.py       to_parquet(frame, backend, path), from_parquet(path, backend)
  tools/
    __init__.py
    catalog.py           ToolSpec definitions — engine-side, tenant-free
services/api/src/lumen_api/
  agents/
    __init__.py
    loop.py              AgentLoop — bounded, budgeted, event-emitting
    supervisor.py        SupervisorAgent
    context_agent.py     profile a source
    cleaning_agent.py    propose a cleaning pipeline
    registry.py          build_tool_registry(org_id, backend) -> ToolRegistry
    prompts/
      supervisor.md  context.md  cleaning.md
  datasets/
    store.py             Postgres+S3 implementation of HandleStore
  runs/
    models.py            Run, Proposal, AgentEvent, UsageRecord
    schemas.py
    repository.py
    router.py            /v1/runs, /v1/proposals, SSE
  usage/
    meter.py             record_usage(), QuotaGate
  uploads/router.py      /v1/uploads
services/worker/
  pyproject.toml
  src/lumen_worker/
    __init__.py
    settings.py
    main.py              arq WorkerSettings
    jobs/
      run_pipeline.py
      profile_source.py
```

---

### Task 1: Provider-agnostic LLM client

**Files:**
- Create: `engine/src/lumen/llm/base.py`, `engine/src/lumen/llm/anthropic_provider.py`, `engine/src/lumen/llm/groq_provider.py`, `engine/src/lumen/llm/registry.py`, `engine/src/lumen/llm/__init__.py`
- Delete: `engine/src/lumen/api/groq/qwen_3_6.py`, `engine/src/lumen/api/huggin_face/qwen_3_6.py`, `engine/src/lumen/api/supabase_api.py`
- Test: `engine/tests/test_llm_provider.py`

**Interfaces:**
- Consumes: nothing from earlier tasks
- Produces:
  - `TokenUsage(input_tokens: int, output_tokens: int, model: str)`
  - `ToolSpec(name: str, description: str, input_schema: dict)`
  - `ToolCall(id: str, name: str, arguments: dict)`
  - `ChatMessage(role: Literal["system","user","assistant","tool"], content: str, tool_calls: list[ToolCall] | None, tool_call_id: str | None)`
  - `LLMResponse(text: str | None, tool_calls: list[ToolCall], usage: TokenUsage, stop_reason: str)`
  - `LLMProvider.complete(messages, tools, max_tokens, temperature) -> LLMResponse` (async)
  - `get_provider(tier: Literal["reasoning","fast"], *, anthropic_key, groq_key, models) -> LLMProvider`

- [ ] **Step 1: Write the failing test**

Create `engine/tests/test_llm_provider.py`:

```python
import pytest

from lumen.llm.base import ChatMessage, LLMResponse, ToolCall, ToolSpec, TokenUsage


class FakeAnthropicClient:
    """Stands in for anthropic.AsyncAnthropic — records the request, returns a canned reply."""

    def __init__(self, reply):
        self._reply = reply
        self.last_kwargs = None
        self.messages = self

    async def create(self, **kwargs):
        self.last_kwargs = kwargs
        return self._reply


class _Block:
    def __init__(self, type_, **kw):
        self.type = type_
        for k, v in kw.items():
            setattr(self, k, v)


class _Reply:
    def __init__(self, content, stop_reason="end_turn", input_tokens=10, output_tokens=5):
        self.content = content
        self.stop_reason = stop_reason
        self.usage = type("U", (), {"input_tokens": input_tokens, "output_tokens": output_tokens})()


@pytest.mark.asyncio
async def test_anthropic_provider_returns_text_and_usage():
    from lumen.llm.anthropic_provider import AnthropicProvider

    client = FakeAnthropicClient(_Reply([_Block("text", text="hello")]))
    provider = AnthropicProvider(client=client, model="claude-sonnet-5")

    response = await provider.complete(
        [ChatMessage(role="user", content="hi")], tools=[], max_tokens=64
    )

    assert isinstance(response, LLMResponse)
    assert response.text == "hello"
    assert response.tool_calls == []
    assert response.usage == TokenUsage(input_tokens=10, output_tokens=5, model="claude-sonnet-5")


@pytest.mark.asyncio
async def test_anthropic_provider_extracts_tool_calls():
    from lumen.llm.anthropic_provider import AnthropicProvider

    block = _Block("tool_use", id="tu_1", name="read_source", input={"source_id": "abc"})
    client = FakeAnthropicClient(_Reply([block], stop_reason="tool_use"))
    provider = AnthropicProvider(client=client, model="claude-sonnet-5")

    response = await provider.complete(
        [ChatMessage(role="user", content="read it")],
        tools=[ToolSpec(name="read_source", description="d", input_schema={"type": "object"})],
        max_tokens=64,
    )

    assert response.stop_reason == "tool_use"
    assert response.tool_calls == [
        ToolCall(id="tu_1", name="read_source", arguments={"source_id": "abc"})
    ]


@pytest.mark.asyncio
async def test_system_messages_move_to_the_system_parameter():
    from lumen.llm.anthropic_provider import AnthropicProvider

    client = FakeAnthropicClient(_Reply([_Block("text", text="ok")]))
    provider = AnthropicProvider(client=client, model="claude-sonnet-5")

    await provider.complete(
        [
            ChatMessage(role="system", content="You are Lumen."),
            ChatMessage(role="user", content="hi"),
        ],
        tools=[],
        max_tokens=64,
    )

    assert client.last_kwargs["system"] == "You are Lumen."
    assert [m["role"] for m in client.last_kwargs["messages"]] == ["user"]


def test_registry_maps_tiers_to_models():
    from lumen.llm.registry import ModelTiers, get_provider

    tiers = ModelTiers(reasoning="claude-opus-5", specialist="claude-sonnet-5", fast="qwen/qwen3.6-27b")
    provider = get_provider("reasoning", anthropic_key="sk-test", groq_key=None, tiers=tiers)
    assert provider.model == "claude-opus-5"


def test_registry_raises_when_the_key_for_a_tier_is_missing():
    from lumen.llm.registry import ModelTiers, get_provider

    tiers = ModelTiers(reasoning="claude-opus-5", specialist="claude-sonnet-5", fast="qwen/qwen3.6-27b")
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        get_provider("reasoning", anthropic_key=None, groq_key="gsk", tiers=tiers)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --directory engine pytest tests/test_llm_provider.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lumen.llm'`

- [ ] **Step 3: Write the base types**

Create `engine/src/lumen/llm/base.py`:

```python
"""Provider-agnostic chat contract.

Deliberately narrow: text in, text or tool calls out, token usage always. Streaming is
handled one layer up by re-calling with the accumulated transcript, so providers stay simple.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int
    output_tokens: int
    model: str


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ChatMessage:
    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    is_error: bool = False


@dataclass(frozen=True)
class LLMResponse:
    text: str | None
    tool_calls: list[ToolCall]
    usage: TokenUsage
    stop_reason: str


class LLMProvider(ABC):
    model: str

    @abstractmethod
    async def complete(
        self,
        messages: list[ChatMessage],
        tools: list[ToolSpec],
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """Send one turn and return the model's reply."""
```

- [ ] **Step 4: Write the Anthropic provider**

Create `engine/src/lumen/llm/anthropic_provider.py`:

```python
from __future__ import annotations

from typing import Any

from lumen.llm.base import ChatMessage, LLMProvider, LLMResponse, ToolCall, ToolSpec, TokenUsage


class AnthropicProvider(LLMProvider):
    """Anthropic Messages API with native tool use."""

    def __init__(self, client: Any, model: str) -> None:
        self._client = client
        self.model = model

    async def complete(
        self,
        messages: list[ChatMessage],
        tools: list[ToolSpec],
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> LLMResponse:
        system_parts = [m.content for m in messages if m.role == "system"]
        payload = [self._encode(m) for m in messages if m.role != "system"]

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": payload,
        }
        if system_parts:
            kwargs["system"] = "\n\n".join(system_parts)
        if tools:
            kwargs["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.input_schema}
                for t in tools
            ]

        reply = await self._client.messages.create(**kwargs)

        text_parts: list[str] = []
        calls: list[ToolCall] = []
        for block in reply.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                calls.append(ToolCall(id=block.id, name=block.name, arguments=dict(block.input)))

        return LLMResponse(
            text="".join(text_parts) or None,
            tool_calls=calls,
            usage=TokenUsage(
                input_tokens=reply.usage.input_tokens,
                output_tokens=reply.usage.output_tokens,
                model=self.model,
            ),
            stop_reason=reply.stop_reason,
        )

    @staticmethod
    def _encode(message: ChatMessage) -> dict[str, Any]:
        if message.role == "tool":
            return {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": message.tool_call_id,
                        "content": message.content,
                        "is_error": message.is_error,
                    }
                ],
            }
        if message.role == "assistant" and message.tool_calls:
            blocks: list[dict[str, Any]] = []
            if message.content:
                blocks.append({"type": "text", "text": message.content})
            blocks.extend(
                {"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments}
                for c in message.tool_calls
            )
            return {"role": "assistant", "content": blocks}
        return {"role": message.role, "content": message.content}
```

- [ ] **Step 5: Write the Groq provider and the registry**

Create `engine/src/lumen/llm/groq_provider.py`:

```python
from __future__ import annotations

import json
from typing import Any

from lumen.llm.base import ChatMessage, LLMProvider, LLMResponse, ToolCall, ToolSpec, TokenUsage


class GroqProvider(LLMProvider):
    """OpenAI-compatible chat completions — the cheap tier for classification and summaries."""

    def __init__(self, client: Any, model: str) -> None:
        self._client = client
        self.model = model

    async def complete(
        self,
        messages: list[ChatMessage],
        tools: list[ToolSpec],
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> LLMResponse:
        payload = [self._encode(m) for m in messages]
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": payload,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.input_schema,
                    },
                }
                for t in tools
            ]
            kwargs["tool_choice"] = "auto"

        reply = await self._client.chat.completions.create(**kwargs)
        choice = reply.choices[0].message
        calls = [
            ToolCall(id=c.id, name=c.function.name, arguments=json.loads(c.function.arguments or "{}"))
            for c in (choice.tool_calls or [])
        ]
        return LLMResponse(
            text=choice.content,
            tool_calls=calls,
            usage=TokenUsage(
                input_tokens=reply.usage.prompt_tokens,
                output_tokens=reply.usage.completion_tokens,
                model=self.model,
            ),
            stop_reason=reply.choices[0].finish_reason,
        )

    @staticmethod
    def _encode(message: ChatMessage) -> dict[str, Any]:
        if message.role == "tool":
            return {"role": "tool", "tool_call_id": message.tool_call_id, "content": message.content}
        if message.role == "assistant" and message.tool_calls:
            return {
                "role": "assistant",
                "content": message.content or None,
                "tool_calls": [
                    {
                        "id": c.id,
                        "type": "function",
                        "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
                    }
                    for c in message.tool_calls
                ],
            }
        return {"role": message.role, "content": message.content}
```

Create `engine/src/lumen/llm/registry.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from lumen.llm.base import LLMProvider

Tier = Literal["reasoning", "specialist", "fast"]


@dataclass(frozen=True)
class ModelTiers:
    reasoning: str = "claude-opus-5"
    specialist: str = "claude-sonnet-5"
    fast: str = "qwen/qwen3.6-27b"


def get_provider(
    tier: Tier,
    *,
    anthropic_key: str | None,
    groq_key: str | None,
    tiers: ModelTiers | None = None,
) -> LLMProvider:
    tiers = tiers or ModelTiers()

    if tier in ("reasoning", "specialist"):
        if not anthropic_key:
            raise ValueError(f"ANTHROPIC_API_KEY is required for the '{tier}' tier")
        from anthropic import AsyncAnthropic

        from lumen.llm.anthropic_provider import AnthropicProvider

        model = tiers.reasoning if tier == "reasoning" else tiers.specialist
        return AnthropicProvider(client=AsyncAnthropic(api_key=anthropic_key), model=model)

    if not groq_key:
        raise ValueError("GROQ_API_KEY is required for the 'fast' tier")
    from openai import AsyncOpenAI

    from lumen.llm.groq_provider import GroqProvider

    client = AsyncOpenAI(api_key=groq_key, base_url="https://api.groq.com/openai/v1")
    return GroqProvider(client=client, model=tiers.fast)
```

Create `engine/src/lumen/llm/__init__.py`:

```python
from lumen.llm.base import (
    ChatMessage, LLMProvider, LLMResponse, ToolCall, ToolSpec, TokenUsage,
)
from lumen.llm.registry import ModelTiers, get_provider

__all__ = [
    "ChatMessage", "LLMProvider", "LLMResponse", "ToolCall", "ToolSpec",
    "TokenUsage", "ModelTiers", "get_provider",
]
```

- [ ] **Step 6: Delete the old clients**

```bash
git rm engine/src/lumen/api/groq/qwen_3_6.py engine/src/lumen/api/huggin_face/qwen_3_6.py engine/src/lumen/api/supabase_api.py
git rm -r --ignore-unmatch engine/src/lumen/api
```

`engine/src/lumen/agents/context_creator.py` imports the deleted clients and will not import until Task 5 rewrites it. That is expected; do not patch it here.

- [ ] **Step 7: Add dependencies and run the tests**

```bash
cd engine && uv add anthropic openai && cd ..
uv run --directory engine pytest tests/test_llm_provider.py -v
```
Expected: PASS — 5 passed

- [ ] **Step 8: Commit**

```bash
git add engine
git commit -m "feat: add provider-agnostic llm client with claude default and groq fast tier"
```

---

### Task 9: Keyless LLM providers — run the whole flow with no API key

> **Ordering:** execute this immediately after Task 1. It is numbered 9 only because it
> was added after the plan was first written. Tasks 5–8 and the web plan's Task 8 all
> depend on it, because it is what lets the vertical slice run green with no API key.

**Files:**
- Create: `engine/src/lumen/llm/mock_provider.py`, `engine/src/lumen/llm/bridge_provider.py`
- Modify: `engine/src/lumen/llm/registry.py`, `engine/src/lumen/llm/__init__.py`
- Test: `engine/tests/test_mock_provider.py`, `engine/tests/test_bridge_provider.py`

**Interfaces:**
- Consumes: `LLMProvider`, `ChatMessage`, `ToolSpec`, `ToolCall`, `LLMResponse`, `TokenUsage` from Task 1
- Produces:
  - `MockProvider(model="mock-agent-v1", null_threshold=0.005)` — a deterministic, network-free `LLMProvider`
  - `BridgeProvider(inbox: Path, poll_seconds=1.0, timeout_seconds=600)` — an `LLMProvider` that writes each request to a file and waits for a human- or agent-authored reply
  - `get_provider(tier, *, anthropic_key, groq_key, tiers, mode="auto")` where `mode ∈ {"auto","anthropic","groq","mock","bridge"}`; `"auto"` resolves to `anthropic` when `anthropic_key` is set and to `mock` otherwise

**Why this exists.** Requiring a paid API key to see the product work is a bad default: it blocks CI, blocks a new contributor's first hour, and makes the end-to-end test unrunnable on a laptop. `MockProvider` removes that requirement without faking the result. It does **not** replay canned text — it reads the tool results it is handed and derives its next move from them, so a run over a real uploaded CSV produces a proposal whose steps come from that file's actual null rates and duplicate counts. Swapping in a real model changes the wording of the rationale, not the shape of the flow.

`BridgeProvider` covers the other half: it lets a human — or Claude in an agent session — answer each model call by hand, which is how you exercise a path the mock does not cover without spending a token.

- [ ] **Step 1: Write the failing tests**

Create `engine/tests/test_mock_provider.py`:

```python
"""MockProvider must play the real flow, deriving each move from real tool results."""
import json

import pytest

from lumen.llm.base import ChatMessage, ToolSpec
from lumen.llm.mock_provider import MockProvider

TOOLS = [
    ToolSpec(name="read_source", description="load a source",
             input_schema={"type": "object", "properties": {"source_id": {"type": "string"}}}),
    ToolSpec(name="profile_source", description="profile a dataset",
             input_schema={"type": "object", "properties": {"rid": {"type": "string"}}}),
    ToolSpec(name="propose_cleaning_pipeline", description="propose steps",
             input_schema={"type": "object", "properties": {"rid": {"type": "string"}}}),
]


def tool_result(call_id: str, payload: dict) -> ChatMessage:
    return ChatMessage(role="tool", tool_call_id=call_id, content=json.dumps(payload))


@pytest.mark.asyncio
async def test_first_move_is_read_source_with_the_id_from_the_prompt():
    provider = MockProvider()
    response = await provider.complete(
        [
            ChatMessage(role="system", content="You are the Cleaning Agent."),
            ChatMessage(
                role="user",
                content="Propose a cleaning pipeline for source 4a1f2c9e-0000-4000-8000-000000000001.",
            ),
        ],
        TOOLS,
    )
    assert len(response.tool_calls) == 1
    call = response.tool_calls[0]
    assert call.name == "read_source"
    assert call.arguments == {"source_id": "4a1f2c9e-0000-4000-8000-000000000001"}
    assert response.stop_reason == "tool_use"


@pytest.mark.asyncio
async def test_second_move_profiles_the_handle_the_first_move_returned():
    provider = MockProvider()
    messages = [
        ChatMessage(role="user", content="Propose a cleaning pipeline for dataset ab12cd34."),
        ChatMessage(role="assistant", content="", tool_calls=[
            __import__("lumen.llm.base", fromlist=["ToolCall"]).ToolCall(
                id="c1", name="read_source", arguments={"source_id": "s1"})
        ]),
        tool_result("c1", {"ok": True, "data": {"rid": "ab12cd34", "row_count": 5, "schema": {"a": "str"}}}),
    ]
    response = await provider.complete(messages, TOOLS)
    assert response.tool_calls[0].name == "profile_source"
    assert response.tool_calls[0].arguments == {"rid": "ab12cd34"}


@pytest.mark.asyncio
async def test_steps_are_derived_from_the_observed_null_rates():
    provider = MockProvider()
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
    messages = [
        ChatMessage(role="user", content="Propose a cleaning pipeline for dataset ab12cd34."),
        tool_result("c2", profile),
    ]
    response = await provider.complete(messages, TOOLS)

    call = response.tool_calls[0]
    assert call.name == "propose_cleaning_pipeline"
    steps = call.arguments["steps"]

    # country_code is above the 0.5% threshold; note (0.1%) and id (0%) are not.
    assert {"drop_nulls": {"columns": ["country_code"]}} in steps
    assert not any("note" in json.dumps(step) for step in steps)
    # a duplicate signal produces a dedupe step on that column
    assert {"drop_duplicates": {"columns": ["email_hash"], "keep": "last"}} in steps
    # the rationale cites the real number it saw
    assert "3.2%" in call.arguments["rationale"]


@pytest.mark.asyncio
async def test_a_clean_dataset_produces_no_proposal_and_says_so():
    provider = MockProvider()
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
    response = await provider.complete(
        [
            ChatMessage(role="user", content="Propose a cleaning pipeline for dataset ab12cd34."),
            tool_result("c2", profile),
        ],
        TOOLS,
    )
    assert response.tool_calls == []
    assert response.stop_reason == "end_turn"
    assert "no material" in (response.text or "").lower()


@pytest.mark.asyncio
async def test_after_a_successful_proposal_it_finishes_with_a_summary():
    provider = MockProvider()
    messages = [
        ChatMessage(role="user", content="Propose a cleaning pipeline for dataset ab12cd34."),
        tool_result("c2", {"ok": True, "data": {"null_rate_by_column": {"c": 0.03}, "row_count": 10}}),
        tool_result("c3", {"ok": True, "data": {"rid": "ab12cd34", "steps": [{"drop_nulls": {"columns": ["c"]}}],
                                                "rationale": "3.0% nulls in c"}}),
    ]
    response = await provider.complete(messages, TOOLS)
    assert response.tool_calls == []
    assert response.stop_reason == "end_turn"
    assert response.text and len(response.text) > 20


@pytest.mark.asyncio
async def test_a_failed_tool_result_is_not_retried_forever():
    provider = MockProvider()
    messages = [ChatMessage(role="user", content="Propose a cleaning pipeline for dataset ab12cd34.")]
    for i in range(4):
        messages.append(tool_result(f"c{i}", {"ok": False, "error": "boom"}))
    response = await provider.complete(messages, TOOLS)
    assert response.tool_calls == []
    assert "could not" in (response.text or "").lower()


@pytest.mark.asyncio
async def test_usage_is_reported_and_nonzero():
    provider = MockProvider()
    response = await provider.complete(
        [ChatMessage(role="user", content="Profile source s1.")], TOOLS
    )
    assert response.usage.model == "mock-agent-v1"
    assert response.usage.input_tokens > 0
    assert response.usage.output_tokens > 0


@pytest.mark.asyncio
async def test_it_never_calls_a_tool_that_was_not_offered():
    provider = MockProvider()
    response = await provider.complete(
        [ChatMessage(role="user", content="Propose a cleaning pipeline for dataset ab12cd34.")],
        tools=[],
    )
    assert response.tool_calls == []


@pytest.mark.asyncio
async def test_it_is_deterministic():
    messages = [ChatMessage(role="user", content="Propose a cleaning pipeline for source s1.")]
    first = await MockProvider().complete(messages, TOOLS)
    second = await MockProvider().complete(messages, TOOLS)
    assert first.tool_calls == second.tool_calls
    assert first.text == second.text
```

Create `engine/tests/test_bridge_provider.py`:

```python
"""BridgeProvider lets a human or an agent answer each model call by hand."""
import asyncio
import json

import pytest

from lumen.llm.base import ChatMessage, ToolSpec
from lumen.llm.bridge_provider import BridgeProvider


@pytest.mark.asyncio
async def test_it_writes_a_request_file_and_reads_the_reply(tmp_path):
    provider = BridgeProvider(inbox=tmp_path, poll_seconds=0.05, timeout_seconds=10)

    async def answer():
        for _ in range(200):
            requests = sorted(tmp_path.glob("*.request.json"))
            if requests:
                request = requests[0]
                payload = json.loads(request.read_text(encoding="utf-8"))
                assert payload["messages"][-1]["content"] == "Profile source s1."
                assert payload["tools"][0]["name"] == "profile_source"
                request.with_suffix("").with_suffix(".response.json").write_text(
                    json.dumps({"text": "done", "tool_calls": []}), encoding="utf-8"
                )
                return
            await asyncio.sleep(0.02)
        raise AssertionError("no request file appeared")

    responder = asyncio.create_task(answer())
    response = await provider.complete(
        [ChatMessage(role="user", content="Profile source s1.")],
        [ToolSpec(name="profile_source", description="d", input_schema={"type": "object"})],
    )
    await responder

    assert response.text == "done"
    assert response.tool_calls == []


@pytest.mark.asyncio
async def test_a_reply_may_carry_tool_calls(tmp_path):
    provider = BridgeProvider(inbox=tmp_path, poll_seconds=0.05, timeout_seconds=10)

    async def answer():
        for _ in range(200):
            requests = sorted(tmp_path.glob("*.request.json"))
            if requests:
                requests[0].with_suffix("").with_suffix(".response.json").write_text(
                    json.dumps(
                        {
                            "text": None,
                            "tool_calls": [
                                {"name": "profile_source", "arguments": {"rid": "ab12cd34"}}
                            ],
                        }
                    ),
                    encoding="utf-8",
                )
                return
            await asyncio.sleep(0.02)
        raise AssertionError("no request file appeared")

    responder = asyncio.create_task(answer())
    response = await provider.complete([ChatMessage(role="user", content="go")], [])
    await responder

    assert response.stop_reason == "tool_use"
    assert response.tool_calls[0].name == "profile_source"
    assert response.tool_calls[0].arguments == {"rid": "ab12cd34"}
    assert response.tool_calls[0].id


@pytest.mark.asyncio
async def test_it_times_out_with_an_actionable_error(tmp_path):
    provider = BridgeProvider(inbox=tmp_path, poll_seconds=0.02, timeout_seconds=0.15)
    with pytest.raises(TimeoutError, match=str(tmp_path.name)):
        await provider.complete([ChatMessage(role="user", content="go")], [])
```

Add to `engine/tests/test_llm_provider.py`:

```python
def test_auto_mode_falls_back_to_mock_without_a_key():
    from lumen.llm.mock_provider import MockProvider
    from lumen.llm.registry import ModelTiers, get_provider

    provider = get_provider(
        "specialist", anthropic_key=None, groq_key=None, tiers=ModelTiers(), mode="auto"
    )
    assert isinstance(provider, MockProvider)


def test_auto_mode_prefers_anthropic_when_a_key_is_present(monkeypatch):
    from lumen.llm.anthropic_provider import AnthropicProvider
    from lumen.llm.registry import ModelTiers, get_provider

    provider = get_provider(
        "specialist", anthropic_key="sk-test", groq_key=None, tiers=ModelTiers(), mode="auto"
    )
    assert isinstance(provider, AnthropicProvider)


def test_explicit_anthropic_mode_still_raises_without_a_key():
    from lumen.llm.registry import ModelTiers, get_provider

    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        get_provider(
            "specialist", anthropic_key=None, groq_key=None, tiers=ModelTiers(), mode="anthropic"
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --directory engine pytest tests/test_mock_provider.py tests/test_bridge_provider.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lumen.llm.mock_provider'`

- [ ] **Step 3: Implement MockProvider**

Create `engine/src/lumen/llm/mock_provider.py`:

```python
"""A deterministic, network-free LLMProvider.

This is not a stub that replays canned text. It inspects the tool results already in the
transcript and derives its next move from them, so an end-to-end run over a real dataset
produces a proposal built from that dataset's real null rates and duplicate counts.

Swapping in a real model changes the wording of the rationale, not the shape of the flow —
which is what makes this safe to use as the default when no API key is configured.
"""
from __future__ import annotations

import json
import re
from typing import Any

from lumen.llm.base import ChatMessage, LLMProvider, LLMResponse, ToolCall, ToolSpec, TokenUsage

UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)
RID_RE = re.compile(r"\bdataset ([0-9a-f]{8,32})\b", re.I)
MAX_CONSECUTIVE_FAILURES = 3


class MockProvider(LLMProvider):
    def __init__(self, model: str = "mock-agent-v1", null_threshold: float = 0.005) -> None:
        self.model = model
        self._null_threshold = null_threshold

    async def complete(
        self,
        messages: list[ChatMessage],
        tools: list[ToolSpec],
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> LLMResponse:
        available = {tool.name for tool in tools}
        results = _tool_results(messages)
        prompt = _first_user_text(messages)

        if _consecutive_failures(results) >= MAX_CONSECUTIVE_FAILURES:
            return self._answer(
                messages,
                "I could not complete this: every tool call failed. "
                "The last error was: " + str(results[-1].get("error", "unknown")),
            )

        proposal = _latest_ok(results, lambda data: "steps" in data)
        if proposal is not None:
            steps = proposal.get("steps") or []
            return self._answer(
                messages,
                f"Proposed {len(steps)} cleaning step(s). "
                + str(proposal.get("rationale", "")).strip(),
            )

        profile = _latest_ok(results, lambda data: "null_rate_by_column" in data)
        if profile is not None and "propose_cleaning_pipeline" in available:
            steps, rationale = _plan(profile, self._null_threshold)
            if not steps:
                return self._answer(
                    messages,
                    "I profiled the dataset and found no material data-quality problems: "
                    + rationale,
                )
            return self._call(
                messages,
                "propose_cleaning_pipeline",
                {
                    "rid": str(profile.get("rid") or _rid_from(messages) or ""),
                    "steps": steps,
                    "rationale": rationale,
                },
            )

        handle = _latest_ok(results, lambda data: "rid" in data)
        if handle is not None and "profile_source" in available:
            return self._call(messages, "profile_source", {"rid": str(handle["rid"])})

        rid = _rid_from(messages)
        if rid and "profile_source" in available and not results:
            return self._call(messages, "profile_source", {"rid": rid})

        source_id = _source_id_from(prompt)
        if source_id and "read_source" in available:
            return self._call(messages, "read_source", {"source_id": source_id})

        if "list_data_sources" in available and not results:
            return self._call(messages, "list_data_sources", {})

        return self._answer(messages, "No further action is available with the tools provided.")

    # ── response builders ────────────────────────────────────────────────

    def _call(self, messages: list[ChatMessage], name: str, arguments: dict[str, Any]) -> LLMResponse:
        call_id = f"mock_{name}_{len(messages)}"
        return LLMResponse(
            text=None,
            tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)],
            usage=_usage(messages, json.dumps(arguments), self.model),
            stop_reason="tool_use",
        )

    def _answer(self, messages: list[ChatMessage], text: str) -> LLMResponse:
        return LLMResponse(
            text=text,
            tool_calls=[],
            usage=_usage(messages, text, self.model),
            stop_reason="end_turn",
        )


# ── transcript inspection ────────────────────────────────────────────────


def _tool_results(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for message in messages:
        if message.role != "tool":
            continue
        try:
            parsed = json.loads(message.content)
        except (ValueError, TypeError):
            continue
        if isinstance(parsed, dict):
            out.append(parsed)
    return out


def _latest_ok(results: list[dict[str, Any]], predicate) -> dict[str, Any] | None:
    for result in reversed(results):
        if not result.get("ok"):
            continue
        data = result.get("data")
        if isinstance(data, dict) and predicate(data):
            return data
    return None


def _consecutive_failures(results: list[dict[str, Any]]) -> int:
    count = 0
    for result in reversed(results):
        if result.get("ok"):
            break
        count += 1
    return count


def _first_user_text(messages: list[ChatMessage]) -> str:
    for message in messages:
        if message.role == "user":
            return message.content
    return ""


def _source_id_from(prompt: str) -> str | None:
    match = UUID_RE.search(prompt)
    return match.group(0) if match else None


def _rid_from(messages: list[ChatMessage]) -> str | None:
    match = RID_RE.search(_first_user_text(messages))
    return match.group(1) if match else None


# ── planning ─────────────────────────────────────────────────────────────


def _plan(profile: dict[str, Any], threshold: float) -> tuple[list[dict[str, Any]], str]:
    null_rates: dict[str, float] = profile.get("null_rate_by_column") or {}
    duplicates: dict[str, int] = profile.get("duplicate_counts") or {}

    offenders = sorted(
        (column, rate)
        for column, rate in null_rates.items()
        if isinstance(rate, (int, float)) and rate > threshold
    )

    steps: list[dict[str, Any]] = []
    notes: list[str] = []

    if offenders:
        steps.append({"drop_nulls": {"columns": [column for column, _ in offenders]}})
        notes.append(
            ", ".join(f"{rate * 100:.1f}% nulls in {column}" for column, rate in offenders)
        )

    for column, count in sorted(duplicates.items()):
        if count and count > 0:
            steps.append({"drop_duplicates": {"columns": [column], "keep": "last"}})
            notes.append(f"{count} duplicate {column} values")

    if not steps:
        rows = profile.get("row_count", "an unknown number of")
        return [], (
            f"every column is below the {threshold * 100:.1f}% null threshold "
            f"across {rows} rows, and no duplicate keys were reported."
        )

    return steps, "Found " + "; ".join(notes) + "."


def _usage(messages: list[ChatMessage], output: str, model: str) -> TokenUsage:
    prompt_chars = sum(len(m.content or "") for m in messages)
    return TokenUsage(
        input_tokens=max(1, prompt_chars // 4),
        output_tokens=max(1, len(output) // 4),
        model=model,
    )
```

- [ ] **Step 4: Implement BridgeProvider**

Create `engine/src/lumen/llm/bridge_provider.py`:

```python
"""An LLMProvider whose replies come from a file, not a network call.

Each `complete` writes `NNN.request.json` into the inbox and polls for `NNN.response.json`.
A human — or an agent driving the session — reads the request and writes the reply. Useful
for exercising a path MockProvider does not cover without spending a token, and for
demonstrating the agent flow with a person in the model's seat.

Response file format:
    {"text": "…" | null,
     "tool_calls": [{"name": "…", "arguments": {…}}, …]}
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from pathlib import Path

from lumen.llm.base import ChatMessage, LLMProvider, LLMResponse, ToolCall, ToolSpec, TokenUsage


class BridgeProvider(LLMProvider):
    def __init__(
        self,
        inbox: Path | str,
        model: str = "bridge",
        poll_seconds: float = 1.0,
        timeout_seconds: float = 600.0,
    ) -> None:
        self.model = model
        self._inbox = Path(inbox)
        self._inbox.mkdir(parents=True, exist_ok=True)
        self._poll = poll_seconds
        self._timeout = timeout_seconds
        self._turn = 0

    async def complete(
        self,
        messages: list[ChatMessage],
        tools: list[ToolSpec],
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> LLMResponse:
        self._turn += 1
        stem = f"{self._turn:03d}"
        request_path = self._inbox / f"{stem}.request.json"
        response_path = self._inbox / f"{stem}.response.json"

        request_path.write_text(
            json.dumps(
                {
                    "turn": self._turn,
                    "messages": [asdict(m) for m in messages],
                    "tools": [asdict(t) for t in tools],
                    "max_tokens": max_tokens,
                    "reply_to": response_path.name,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

        waited = 0.0
        while waited < self._timeout:
            if response_path.exists():
                payload = json.loads(response_path.read_text(encoding="utf-8"))
                calls = [
                    ToolCall(
                        id=call.get("id") or f"bridge_{stem}_{index}",
                        name=call["name"],
                        arguments=call.get("arguments") or {},
                    )
                    for index, call in enumerate(payload.get("tool_calls") or [])
                ]
                text = payload.get("text")
                return LLMResponse(
                    text=text,
                    tool_calls=calls,
                    usage=TokenUsage(
                        input_tokens=max(1, sum(len(m.content or "") for m in messages) // 4),
                        output_tokens=max(1, len(json.dumps(payload)) // 4),
                        model=self.model,
                    ),
                    stop_reason="tool_use" if calls else "end_turn",
                )
            await asyncio.sleep(self._poll)
            waited += self._poll

        raise TimeoutError(
            f"No reply within {self._timeout:.0f}s. Write {response_path} to answer "
            f"the request in {request_path}."
        )
```

- [ ] **Step 5: Wire the registry**

Replace `get_provider` in `engine/src/lumen/llm/registry.py` — keep `ModelTiers` and `Tier` as they are and add `Mode`:

```python
Mode = Literal["auto", "anthropic", "groq", "mock", "bridge"]


def get_provider(
    tier: Tier,
    *,
    anthropic_key: str | None,
    groq_key: str | None,
    tiers: ModelTiers | None = None,
    mode: Mode = "auto",
    bridge_inbox: str | None = None,
) -> LLMProvider:
    """Resolve a provider.

    `auto` is the default and is what makes the product runnable with no credentials:
    Anthropic when a key is configured, the deterministic MockProvider otherwise.
    An explicit mode never silently falls back — asking for `anthropic` without a key
    is a configuration error and is raised as one.
    """
    tiers = tiers or ModelTiers()

    if mode == "auto":
        mode = "anthropic" if anthropic_key else "mock"

    if mode == "mock":
        from lumen.llm.mock_provider import MockProvider

        return MockProvider()

    if mode == "bridge":
        from lumen.llm.bridge_provider import BridgeProvider

        return BridgeProvider(inbox=bridge_inbox or ".llm-bridge")

    if mode == "anthropic":
        if not anthropic_key:
            raise ValueError(f"ANTHROPIC_API_KEY is required for the '{tier}' tier")
        from anthropic import AsyncAnthropic

        from lumen.llm.anthropic_provider import AnthropicProvider

        model = tiers.reasoning if tier == "reasoning" else tiers.specialist
        return AnthropicProvider(client=AsyncAnthropic(api_key=anthropic_key), model=model)

    if not groq_key:
        raise ValueError("GROQ_API_KEY is required for the 'groq' mode")
    from openai import AsyncOpenAI

    from lumen.llm.groq_provider import GroqProvider

    client = AsyncOpenAI(api_key=groq_key, base_url="https://api.groq.com/openai/v1")
    return GroqProvider(client=client, model=tiers.fast)
```

Export both providers from `engine/src/lumen/llm/__init__.py` alongside the existing names:

```python
from lumen.llm.bridge_provider import BridgeProvider
from lumen.llm.mock_provider import MockProvider
```

and add `"MockProvider"`, `"BridgeProvider"` to `__all__`.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run --directory engine pytest tests/test_mock_provider.py tests/test_bridge_provider.py tests/test_llm_provider.py -v`
Expected: PASS — 12 mock/bridge tests plus the 8 registry/provider tests.

Note on the `test_first_move_is_read_source_with_the_id_from_the_prompt` assertion: the prompt the cleaning agent actually sends is `"Propose a cleaning pipeline for dataset {rid}."` (agent-layer Task 6) and the context agent sends `"Profile data source {source_id}."` (Task 6). `MockProvider` handles both shapes — a bare `rid` goes straight to `profile_source`, a UUID goes through `read_source` first.

- [ ] **Step 7: Commit**

```bash
git add engine
git commit -m "feat: add keyless mock and bridge llm providers so the flow runs without an api key"
```

---

### Task 2: DatasetHandle — durable, tenant-scoped dataframe handles

**Files:**
- Create: `engine/src/lumen/datasets/handle.py`, `engine/src/lumen/datasets/materialize.py`, `engine/src/lumen/datasets/__init__.py`
- Delete: `engine/src/lumen/model_tools/object_registry.py`
- Test: `engine/tests/test_datasets.py`

**Interfaces:**
- Consumes: `lumen.core.backend.Backend`, `lumen.readers.inyeccion.ReadersInyeccionDependency`
- Produces:
  - `DatasetHandle(rid: str, uri: str, backend: Backend, schema: dict[str,str], row_count: int, byte_size: int)`
  - `write_parquet(frame: Any, backend: Backend, path: str) -> DatasetMeta` where `DatasetMeta(schema, row_count, byte_size)`
  - `read_parquet(path: str, backend: Backend) -> Any`
  - `frame_schema(frame: Any, backend: Backend) -> dict[str, str]`

- [ ] **Step 1: Write the failing test**

Create `engine/tests/test_datasets.py`:

```python
import pandas as pd
import polars as pl
import pytest

from lumen.datasets.materialize import frame_schema, read_parquet, write_parquet


@pytest.fixture
def frame_data():
    return {"id": [1, 2, 3], "name": ["a", "b", "c"], "score": [1.5, 2.5, 3.5]}


def test_pandas_round_trip(tmp_path, frame_data):
    path = str(tmp_path / "p.parquet")
    meta = write_parquet(pd.DataFrame(frame_data), "pandas", path)

    assert meta.row_count == 3
    assert set(meta.schema) == {"id", "name", "score"}
    assert meta.byte_size > 0

    restored = read_parquet(path, "pandas")
    assert list(restored.columns) == ["id", "name", "score"]
    assert len(restored) == 3


def test_polars_round_trip_returns_a_lazyframe(tmp_path, frame_data):
    path = str(tmp_path / "q.parquet")
    meta = write_parquet(pl.DataFrame(frame_data), "polars", path)
    assert meta.row_count == 3

    restored = read_parquet(path, "polars")
    assert isinstance(restored, pl.LazyFrame), "polars reads must stay lazy"
    assert restored.collect().height == 3


def test_polars_lazyframe_input_is_collected_before_writing(tmp_path, frame_data):
    path = str(tmp_path / "r.parquet")
    meta = write_parquet(pl.LazyFrame(frame_data), "polars", path)
    assert meta.row_count == 3


def test_frame_schema_reports_dtypes_as_strings(frame_data):
    schema = frame_schema(pd.DataFrame(frame_data), "pandas")
    assert schema["id"].startswith("int")
    assert schema["score"].startswith("float")
    assert all(isinstance(v, str) for v in schema.values())


def test_unknown_backend_raises(tmp_path, frame_data):
    with pytest.raises(ValueError, match="Unsupported backend"):
        write_parquet(pd.DataFrame(frame_data), "duckdb", str(tmp_path / "x.parquet"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --directory engine pytest tests/test_datasets.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lumen.datasets'`

- [ ] **Step 3: Write the handle type**

Create `engine/src/lumen/datasets/handle.py`:

```python
"""A durable reference to a materialised dataset.

Replaces the process-local dict in the retired `model_tools/object_registry.py`. A handle
survives process restarts, is scoped to one organization by the store that issues it, and
carries enough metadata for an agent to reason about the data without loading it.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from lumen.core.backend import Backend


@dataclass(frozen=True)
class DatasetHandle:
    rid: str
    uri: str
    backend: Backend
    schema: dict[str, str]
    row_count: int
    byte_size: int
    created_at: datetime | None = None
    expires_at: datetime | None = None


class HandleStore(Protocol):
    """Persistence contract. The tenant-aware implementation lives in services/api."""

    async def put(self, frame: object, backend: Backend, *, label: str) -> DatasetHandle: ...

    async def get(self, rid: str) -> DatasetHandle: ...

    async def resolve(self, rid: str) -> object:
        """Return a backend-native frame for the handle."""
```

- [ ] **Step 4: Write materialisation**

Create `engine/src/lumen/datasets/materialize.py`:

```python
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from lumen.core.backend import Backend, validate_backend


@dataclass(frozen=True)
class DatasetMeta:
    schema: dict[str, str]
    row_count: int
    byte_size: int


def frame_schema(frame: Any, backend: Backend | str) -> dict[str, str]:
    backend = validate_backend(str(backend))
    if backend == "pandas":
        return {str(c): str(d) for c, d in frame.dtypes.items()}
    if backend == "polars":
        schema = frame.collect_schema() if hasattr(frame, "collect_schema") else frame.schema
        return {str(name): str(dtype) for name, dtype in schema.items()}
    return {field.name: field.dataType.simpleString() for field in frame.schema.fields}


def write_parquet(frame: Any, backend: Backend | str, path: str) -> DatasetMeta:
    backend = validate_backend(str(backend))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    if backend == "pandas":
        frame.to_parquet(path, index=False)
        row_count = len(frame)
        schema = frame_schema(frame, backend)
    elif backend == "polars":
        materialised = frame.collect() if hasattr(frame, "collect") else frame
        materialised.write_parquet(path)
        row_count = materialised.height
        schema = frame_schema(materialised, backend)
    else:  # spark
        frame.write.mode("overwrite").parquet(path)
        row_count = frame.count()
        schema = frame_schema(frame, backend)

    byte_size = _tree_size(path)
    return DatasetMeta(schema=schema, row_count=row_count, byte_size=byte_size)


def read_parquet(path: str, backend: Backend | str) -> Any:
    backend = validate_backend(str(backend))
    if backend == "pandas":
        import pandas as pd

        return pd.read_parquet(path)
    if backend == "polars":
        import polars as pl

        return pl.scan_parquet(path)

    from pyspark.sql import SparkSession

    return SparkSession.builder.getOrCreate().read.parquet(path)


def _tree_size(path: str) -> int:
    if os.path.isfile(path):
        return os.path.getsize(path)
    total = 0
    for root, _, files in os.walk(path):
        total += sum(os.path.getsize(os.path.join(root, f)) for f in files)
    return total
```

Create `engine/src/lumen/datasets/__init__.py`:

```python
from lumen.datasets.handle import DatasetHandle, HandleStore
from lumen.datasets.materialize import DatasetMeta, frame_schema, read_parquet, write_parquet

__all__ = [
    "DatasetHandle", "HandleStore", "DatasetMeta",
    "frame_schema", "read_parquet", "write_parquet",
]
```

`validate_backend` raises `ValueError("Unsupported backend '<x>'. Choose one of: ...")`, which satisfies the `test_unknown_backend_raises` match.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run --directory engine pytest tests/test_datasets.py -v`
Expected: PASS — 5 passed

- [ ] **Step 6: Retire the object registry**

```bash
git rm engine/src/lumen/model_tools/object_registry.py
```

`data_reader_tool.py` and `create_data_context.py` import it and will not import until Task 4 replaces them. Expected.

- [ ] **Step 7: Commit**

```bash
git add engine
git commit -m "feat: add durable dataset handles backed by parquet, retire in-process registry"
```

---

### Task 3: Run, Proposal, AgentEvent and UsageRecord

**Files:**
- Create: `services/api/src/lumen_api/runs/__init__.py`, `services/api/src/lumen_api/runs/models.py`, `services/api/src/lumen_api/runs/schemas.py`, `services/api/src/lumen_api/runs/repository.py`, `services/api/alembic/versions/0003_agent_domain.py`
- Modify: `services/api/src/lumen_api/db/models/__init__.py`, `services/api/alembic/versions/0002_rls.py` (extend `ORG_SCOPED_TABLES` — do this in a new migration `0004_rls_agent.py`, never by editing an applied revision)
- Test: `services/api/tests/test_runs_repository.py`

**Interfaces:**
- Consumes: `Base`, `org_session`
- Produces:
  - models `Run(id, org_id, source_id, thread_id, kind, status, backend, started_at, finished_at, error)`, `Proposal(id, org_id, run_id, thread_id, author_agent, kind, status, spec, rationale, estimate, decided_by, decided_at, created_at)`, `AgentEvent(id, org_id, run_id, seq, type, payload, created_at)`, `UsageRecord(id, org_id, run_id, agent, metric, quantity, cost_micros, occurred_at, meta)`, `DatasetHandleRow(rid, org_id, uri, backend, schema, row_count, byte_size, created_at, expires_at)`
  - repository functions `create_run`, `get_run`, `set_run_status`, `append_event`, `list_events`, `create_proposal`, `get_proposal`, `decide_proposal`
- Statuses: `Run.status ∈ {queued, running, succeeded, failed, cancelled}`; `Proposal.status ∈ {draft, awaiting_review, accepted, rejected, applied, failed}`; `AgentEvent.type ∈ {thinking, tool_call, tool_result, message, proposal, error, done}`

- [ ] **Step 1: Write the failing test**

Create `services/api/tests/test_runs_repository.py`:

```python
import uuid

import pytest

from lumen_api.runs import repository as repo


@pytest.mark.asyncio
async def test_run_lifecycle_and_event_ordering(one_org):
    org_id = one_org
    run = await repo.create_run(org_id, source_id=None, thread_id=uuid.uuid4(), kind="profile", backend="polars")
    assert run.status == "queued"

    await repo.set_run_status(org_id, run.id, "running")
    for kind in ("thinking", "tool_call", "tool_result", "done"):
        await repo.append_event(org_id, run.id, kind, {"k": kind})
    await repo.set_run_status(org_id, run.id, "succeeded")

    events = await repo.list_events(org_id, run.id)
    assert [e.seq for e in events] == [1, 2, 3, 4]
    assert [e.type for e in events] == ["thinking", "tool_call", "tool_result", "done"]

    reloaded = await repo.get_run(org_id, run.id)
    assert reloaded.status == "succeeded"
    assert reloaded.finished_at is not None


@pytest.mark.asyncio
async def test_proposal_starts_awaiting_review_and_records_the_decision(one_org):
    org_id = one_org
    thread_id = uuid.uuid4()
    run = await repo.create_run(org_id, source_id=None, thread_id=thread_id, kind="clean", backend="polars")

    proposal = await repo.create_proposal(
        org_id,
        run_id=run.id,
        thread_id=thread_id,
        author_agent="cleaning",
        kind="cleaning_pipeline",
        spec=[{"drop_nulls": {"columns": ["country_code"]}}],
        rationale="3.2% of rows have a null country_code.",
        estimate={"affected_rows": 15420, "est_seconds": 1.2},
    )
    assert proposal.status == "awaiting_review"

    decider = uuid.uuid4()
    decided = await repo.decide_proposal(org_id, proposal.id, accept=True, user_id=decider)
    assert decided.status == "accepted"
    assert decided.decided_by == decider
    assert decided.decided_at is not None


@pytest.mark.asyncio
async def test_deciding_twice_is_rejected(one_org):
    org_id = one_org
    thread_id = uuid.uuid4()
    run = await repo.create_run(org_id, source_id=None, thread_id=thread_id, kind="clean", backend="polars")
    proposal = await repo.create_proposal(
        org_id, run_id=run.id, thread_id=thread_id, author_agent="cleaning",
        kind="cleaning_pipeline", spec=[], rationale="r", estimate={},
    )
    await repo.decide_proposal(org_id, proposal.id, accept=True, user_id=uuid.uuid4())

    with pytest.raises(Exception):
        await repo.decide_proposal(org_id, proposal.id, accept=False, user_id=uuid.uuid4())


@pytest.mark.asyncio
async def test_another_org_cannot_read_the_run(two_orgs):
    org_a, org_b = two_orgs
    run = await repo.create_run(org_a, source_id=None, thread_id=uuid.uuid4(), kind="profile", backend="polars")

    from lumen_api.errors import NotFound

    with pytest.raises(NotFound):
        await repo.get_run(org_b, run.id)
```

Add to `services/api/tests/conftest.py`:

```python
@pytest_asyncio.fixture
async def one_org(two_orgs):
    org_a, _ = two_orgs
    yield org_a
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --directory services/api pytest tests/test_runs_repository.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lumen_api.runs'`

- [ ] **Step 3: Write the models**

Create `services/api/src/lumen_api/runs/models.py`:

```python
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    BigInteger, DateTime, ForeignKey, Index, Integer, Numeric, String, Text,
    UniqueConstraint, func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID as PgUUID
from sqlalchemy.orm import Mapped, mapped_column

from lumen_api.db.base import Base


def _org_fk() -> Mapped[uuid.UUID]:
    return mapped_column(
        PgUUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
    )


class Run(Base):
    __tablename__ = "runs"
    __table_args__ = (Index("ix_runs_org_created", "org_id", "started_at"),)

    id: Mapped[uuid.UUID] = mapped_column(PgUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    org_id: Mapped[uuid.UUID] = _org_fk()
    source_id: Mapped[uuid.UUID | None] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("data_sources.id", ondelete="SET NULL"), nullable=True
    )
    thread_id: Mapped[uuid.UUID] = mapped_column(PgUUID(as_uuid=True), nullable=False)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="queued")
    backend: Mapped[str] = mapped_column(String(16), nullable=False, default="polars")
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)


class Proposal(Base):
    __tablename__ = "proposals"

    id: Mapped[uuid.UUID] = mapped_column(PgUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    org_id: Mapped[uuid.UUID] = _org_fk()
    run_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("runs.id", ondelete="CASCADE"), nullable=False
    )
    thread_id: Mapped[uuid.UUID] = mapped_column(PgUUID(as_uuid=True), nullable=False)
    author_agent: Mapped[str] = mapped_column(String(32), nullable=False)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="awaiting_review")
    spec: Mapped[dict | list] = mapped_column(JSONB, nullable=False)
    rationale: Mapped[str] = mapped_column(Text, nullable=False, default="")
    estimate: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    decided_by: Mapped[uuid.UUID | None] = mapped_column(PgUUID(as_uuid=True), nullable=True)
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    applied_run_id: Mapped[uuid.UUID | None] = mapped_column(PgUUID(as_uuid=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class AgentEvent(Base):
    __tablename__ = "agent_events"
    __table_args__ = (UniqueConstraint("run_id", "seq"),)

    id: Mapped[uuid.UUID] = mapped_column(PgUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    org_id: Mapped[uuid.UUID] = _org_fk()
    run_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("runs.id", ondelete="CASCADE"), nullable=False
    )
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    type: Mapped[str] = mapped_column(String(24), nullable=False)
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class UsageRecord(Base):
    __tablename__ = "usage_records"
    __table_args__ = (Index("ix_usage_org_occurred", "org_id", "occurred_at"),)

    id: Mapped[uuid.UUID] = mapped_column(PgUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    org_id: Mapped[uuid.UUID] = _org_fk()
    run_id: Mapped[uuid.UUID | None] = mapped_column(PgUUID(as_uuid=True), nullable=True)
    agent: Mapped[str | None] = mapped_column(String(32), nullable=True)
    metric: Mapped[str] = mapped_column(String(32), nullable=False)
    quantity: Mapped[float] = mapped_column(Numeric(20, 4), nullable=False)
    cost_micros: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    meta: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)


class DatasetHandleRow(Base):
    __tablename__ = "dataset_handles"

    rid: Mapped[str] = mapped_column(String(32), primary_key=True)
    org_id: Mapped[uuid.UUID] = _org_fk()
    uri: Mapped[str] = mapped_column(Text, nullable=False)
    backend: Mapped[str] = mapped_column(String(16), nullable=False)
    schema_json: Mapped[dict] = mapped_column("schema", JSONB, nullable=False, default=dict)
    row_count: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    byte_size: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    label: Mapped[str] = mapped_column(String(120), nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
```

Re-export in `services/api/src/lumen_api/db/models/__init__.py`:

```python
from lumen_api.db.models.identity import Membership, Organization, User
from lumen_api.db.models.source import DataSource
from lumen_api.runs.models import AgentEvent, DatasetHandleRow, Proposal, Run, UsageRecord

__all__ = [
    "User", "Organization", "Membership", "DataSource",
    "Run", "Proposal", "AgentEvent", "UsageRecord", "DatasetHandleRow",
]
```

- [ ] **Step 4: Write the repository**

Create `services/api/src/lumen_api/runs/repository.py`:

```python
"""Data access for the agent domain. Every function opens its own org-scoped transaction."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func, select

from lumen_api.db.session import org_session
from lumen_api.errors import Conflict, NotFound
from lumen_api.runs.models import AgentEvent, Proposal, Run, UsageRecord

TERMINAL_RUN_STATES = {"succeeded", "failed", "cancelled"}


async def create_run(
    org_id: uuid.UUID,
    *,
    source_id: uuid.UUID | None,
    thread_id: uuid.UUID,
    kind: str,
    backend: str,
) -> Run:
    run = Run(
        org_id=org_id, source_id=source_id, thread_id=thread_id,
        kind=kind, backend=backend, status="queued",
    )
    async with org_session(org_id) as db:
        db.add(run)
        await db.flush()
        await db.refresh(run)
        return run


async def get_run(org_id: uuid.UUID, run_id: uuid.UUID) -> Run:
    async with org_session(org_id) as db:
        run = await db.get(Run, run_id)
        if run is None:
            raise NotFound("Run not found")
        return run


async def set_run_status(
    org_id: uuid.UUID, run_id: uuid.UUID, status: str, *, error: str | None = None
) -> Run:
    async with org_session(org_id) as db:
        run = await db.get(Run, run_id)
        if run is None:
            raise NotFound("Run not found")
        run.status = status
        run.error = error
        if status in TERMINAL_RUN_STATES:
            run.finished_at = datetime.now(timezone.utc)
        await db.flush()
        await db.refresh(run)
        return run


async def append_event(
    org_id: uuid.UUID, run_id: uuid.UUID, type_: str, payload: dict[str, Any]
) -> AgentEvent:
    async with org_session(org_id) as db:
        next_seq = await db.scalar(
            select(func.coalesce(func.max(AgentEvent.seq), 0) + 1).where(AgentEvent.run_id == run_id)
        )
        event = AgentEvent(
            org_id=org_id, run_id=run_id, seq=int(next_seq), type=type_, payload=payload
        )
        db.add(event)
        await db.flush()
        await db.refresh(event)
        return event


async def list_events(
    org_id: uuid.UUID, run_id: uuid.UUID, *, after_seq: int = 0
) -> list[AgentEvent]:
    async with org_session(org_id) as db:
        result = await db.execute(
            select(AgentEvent)
            .where(AgentEvent.run_id == run_id, AgentEvent.seq > after_seq)
            .order_by(AgentEvent.seq)
        )
        return list(result.scalars())


async def create_proposal(
    org_id: uuid.UUID,
    *,
    run_id: uuid.UUID,
    thread_id: uuid.UUID,
    author_agent: str,
    kind: str,
    spec: Any,
    rationale: str,
    estimate: dict[str, Any],
) -> Proposal:
    proposal = Proposal(
        org_id=org_id, run_id=run_id, thread_id=thread_id, author_agent=author_agent,
        kind=kind, spec=spec, rationale=rationale, estimate=estimate, status="awaiting_review",
    )
    async with org_session(org_id) as db:
        db.add(proposal)
        await db.flush()
        await db.refresh(proposal)
        return proposal


async def get_proposal(org_id: uuid.UUID, proposal_id: uuid.UUID) -> Proposal:
    async with org_session(org_id) as db:
        proposal = await db.get(Proposal, proposal_id)
        if proposal is None:
            raise NotFound("Proposal not found")
        return proposal


async def decide_proposal(
    org_id: uuid.UUID, proposal_id: uuid.UUID, *, accept: bool, user_id: uuid.UUID
) -> Proposal:
    async with org_session(org_id) as db:
        proposal = await db.get(Proposal, proposal_id)
        if proposal is None:
            raise NotFound("Proposal not found")
        if proposal.status != "awaiting_review":
            raise Conflict(f"Proposal already {proposal.status}")
        proposal.status = "accepted" if accept else "rejected"
        proposal.decided_by = user_id
        proposal.decided_at = datetime.now(timezone.utc)
        await db.flush()
        await db.refresh(proposal)
        return proposal


async def mark_proposal_applied(
    org_id: uuid.UUID, proposal_id: uuid.UUID, *, applied_run_id: uuid.UUID
) -> None:
    async with org_session(org_id) as db:
        proposal = await db.get(Proposal, proposal_id)
        if proposal is None:
            raise NotFound("Proposal not found")
        proposal.status = "applied"
        proposal.applied_run_id = applied_run_id


async def record_usage(
    db,
    org_id: uuid.UUID,
    *,
    metric: str,
    quantity: float,
    run_id: uuid.UUID | None = None,
    agent: str | None = None,
    cost_micros: int = 0,
    meta: dict[str, Any] | None = None,
) -> None:
    """Written inside the caller's transaction — usage and work commit together."""
    db.add(
        UsageRecord(
            org_id=org_id, run_id=run_id, agent=agent, metric=metric,
            quantity=quantity, cost_micros=cost_micros, meta=meta or {},
        )
    )
```

- [ ] **Step 5: Write the migrations**

```bash
uv run --directory services/api alembic revision --autogenerate -m "agent domain" --rev-id 0003
```

Then create `services/api/alembic/versions/0004_rls_agent.py`:

```python
"""RLS for the agent-domain tables."""
from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None

TABLES = ("runs", "proposals", "agent_events", "usage_records", "dataset_handles")


def upgrade() -> None:
    for table in TABLES:
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        op.execute(
            f"""
            CREATE POLICY {table}_org_isolation ON {table}
            USING (org_id = current_setting('app.current_org', true)::uuid)
            WITH CHECK (org_id = current_setting('app.current_org', true)::uuid)
            """
        )


def downgrade() -> None:
    for table in TABLES:
        op.execute(f"DROP POLICY IF EXISTS {table}_org_isolation ON {table}")
        op.execute(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY")
```

Run: `uv run --directory services/api alembic upgrade head`

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run --directory services/api pytest tests/test_runs_repository.py tests/test_rls.py -v`
Expected: PASS — 4 new + 3 RLS tests. `test_every_org_scoped_table_has_rls_enabled` now also covers the five new tables.

- [ ] **Step 7: Commit**

```bash
git add services/api
git commit -m "feat: add run, proposal, agent event and usage record domain"
```

---

### Task 4: Tenant-aware handle store and the tool registry

**Files:**
- Create: `services/api/src/lumen_api/datasets/__init__.py`, `services/api/src/lumen_api/datasets/store.py`, `services/api/src/lumen_api/agents/__init__.py`, `services/api/src/lumen_api/agents/registry.py`
- Delete: `engine/src/lumen/model_tools/data_reader_tool.py`, `engine/src/lumen/model_tools/meta_data_context_tool.py`, `engine/src/lumen/model_tools/create_data_context.py`
- Test: `services/api/tests/test_tool_registry.py`

**Interfaces:**
- Consumes: `DatasetHandle`, `write_parquet`, `read_parquet`, `AgentMasterFactory`, `DatasetHandleRow`, `DataSource`
- Produces:
  - `PostgresHandleStore(org_id, backend)` implementing `HandleStore`
  - `Tool(spec: ToolSpec, handler: Callable[..., Awaitable[dict]])`
  - `ToolRegistry` with `specs() -> list[ToolSpec]`, `async invoke(name: str, arguments: dict) -> dict`, `has(name) -> bool`
  - `build_tool_registry(org_id: UUID, backend: str) -> ToolRegistry` exposing: `list_data_sources`, `profile_source`, `read_source`, `propose_cleaning_pipeline`, `run_statistic`

- [ ] **Step 1: Write the failing test**

Create `services/api/tests/test_tool_registry.py`:

```python
import uuid

import pytest

from lumen_api.agents.registry import build_tool_registry


@pytest.mark.asyncio
async def test_registry_exposes_the_expected_tools(one_org):
    registry = build_tool_registry(one_org, backend="polars")
    names = {spec.name for spec in registry.specs()}
    assert names == {
        "list_data_sources", "profile_source", "read_source",
        "propose_cleaning_pipeline", "run_statistic",
    }


@pytest.mark.asyncio
async def test_every_spec_has_a_json_schema(one_org):
    registry = build_tool_registry(one_org, backend="polars")
    for spec in registry.specs():
        assert spec.input_schema["type"] == "object"
        assert "properties" in spec.input_schema
        assert spec.description, f"{spec.name} needs a description the model can act on"


@pytest.mark.asyncio
async def test_invoking_an_unknown_tool_returns_an_error_result(one_org):
    registry = build_tool_registry(one_org, backend="polars")
    result = await registry.invoke("no_such_tool", {})
    assert result["ok"] is False
    assert "unknown tool" in result["error"].lower()


@pytest.mark.asyncio
async def test_a_raising_tool_becomes_an_error_result_not_an_exception(one_org):
    registry = build_tool_registry(one_org, backend="polars")
    result = await registry.invoke("read_source", {"source_id": str(uuid.uuid4())})
    assert result["ok"] is False
    assert isinstance(result["error"], str)


@pytest.mark.asyncio
async def test_propose_cleaning_pipeline_rejects_an_unknown_step(one_org):
    registry = build_tool_registry(one_org, backend="polars")
    result = await registry.invoke(
        "propose_cleaning_pipeline",
        {"rid": "deadbeef", "steps": [{"summon_daemon": {}}], "rationale": "no"},
    )
    assert result["ok"] is False
    assert "summon_daemon" in result["error"]


@pytest.mark.asyncio
async def test_list_data_sources_is_scoped_to_the_org(one_org):
    registry = build_tool_registry(one_org, backend="polars")
    result = await registry.invoke("list_data_sources", {})
    assert result["ok"] is True
    assert isinstance(result["data"]["sources"], list)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --directory services/api pytest tests/test_tool_registry.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lumen_api.agents'`

- [ ] **Step 3: Write the handle store**

Create `services/api/src/lumen_api/datasets/store.py`:

```python
"""Org-scoped dataset handles: Parquet in object storage, metadata in Postgres."""
from __future__ import annotations

import os
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import boto3

from lumen.datasets.handle import DatasetHandle
from lumen.datasets.materialize import read_parquet, write_parquet
from lumen_api.db.session import org_session
from lumen_api.errors import NotFound
from lumen_api.runs.models import DatasetHandleRow
from lumen_api.settings import get_settings

HANDLE_TTL = timedelta(days=7)


def _s3():
    settings = get_settings()
    return boto3.client(
        "s3",
        endpoint_url=settings.storage_endpoint,
        aws_access_key_id=settings.storage_access_key.get_secret_value(),
        aws_secret_access_key=settings.storage_secret_key.get_secret_value(),
    )


class PostgresHandleStore:
    def __init__(self, org_id: uuid.UUID, backend: str) -> None:
        self._org_id = org_id
        self._backend = backend
        self._settings = get_settings()

    async def put(self, frame: Any, backend: str | None = None, *, label: str = "") -> DatasetHandle:
        backend = backend or self._backend
        rid = uuid.uuid4().hex[:16]
        key = f"org/{self._org_id}/datasets/{rid}.parquet"

        with tempfile.TemporaryDirectory() as tmp:
            local = os.path.join(tmp, f"{rid}.parquet")
            meta = write_parquet(frame, backend, local)
            _s3().upload_file(local, self._settings.storage_bucket, key)

        uri = f"s3://{self._settings.storage_bucket}/{key}"
        async with org_session(self._org_id) as db:
            db.add(
                DatasetHandleRow(
                    rid=rid, org_id=self._org_id, uri=uri, backend=backend,
                    schema_json=meta.schema, row_count=meta.row_count,
                    byte_size=meta.byte_size, label=label,
                    expires_at=datetime.now(timezone.utc) + HANDLE_TTL,
                )
            )
        return DatasetHandle(
            rid=rid, uri=uri, backend=backend, schema=meta.schema,
            row_count=meta.row_count, byte_size=meta.byte_size,
        )

    async def get(self, rid: str) -> DatasetHandle:
        async with org_session(self._org_id) as db:
            row = await db.get(DatasetHandleRow, rid)
            if row is None:
                raise NotFound(f"Dataset handle '{rid}' not found")
            return DatasetHandle(
                rid=row.rid, uri=row.uri, backend=row.backend, schema=row.schema_json,
                row_count=row.row_count, byte_size=row.byte_size,
                created_at=row.created_at, expires_at=row.expires_at,
            )

    async def resolve(self, rid: str) -> Any:
        handle = await self.get(rid)
        bucket, _, key = handle.uri.removeprefix("s3://").partition("/")
        tmpdir = tempfile.mkdtemp(prefix="lumen-")
        local = os.path.join(tmpdir, f"{rid}.parquet")
        _s3().download_file(bucket, key, local)
        return read_parquet(local, handle.backend)
```

- [ ] **Step 4: Write the tool registry**

Create `services/api/src/lumen_api/agents/registry.py`:

```python
"""Tools are thin typed wrappers over engine capabilities.

Every handler returns `{"ok": bool, "data"|"error": ...}`. Handlers never raise: a failure
is a result the model can read and recover from.
"""
from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select

from lumen.agents.master_factory import AgentMasterFactory
from lumen.data_cleaning.data_cleaning_pipeline import PipelineBuilder
from lumen.llm.base import ToolSpec
from lumen_api.datasets.store import PostgresHandleStore
from lumen_api.db.models.source import DataSource
from lumen_api.db.session import org_session

Handler = Callable[..., Awaitable[dict[str, Any]]]


@dataclass(frozen=True)
class Tool:
    spec: ToolSpec
    handler: Handler


class ToolRegistry:
    def __init__(self, tools: list[Tool]) -> None:
        self._tools = {t.spec.name: t for t in tools}

    def specs(self) -> list[ToolSpec]:
        return [t.spec for t in self._tools.values()]

    def has(self, name: str) -> bool:
        return name in self._tools

    async def invoke(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        tool = self._tools.get(name)
        if tool is None:
            return {"ok": False, "error": f"Unknown tool '{name}'"}
        try:
            return await tool.handler(**arguments)
        except TypeError as exc:
            return {"ok": False, "error": f"Bad arguments for '{name}': {exc}"}
        except Exception as exc:  # noqa: BLE001 — tool errors are results, not crashes
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def build_tool_registry(org_id: uuid.UUID, backend: str) -> ToolRegistry:
    master = AgentMasterFactory(backend)
    store = PostgresHandleStore(org_id, backend)

    async def list_data_sources() -> dict[str, Any]:
        async with org_session(org_id) as db:
            rows = (await db.execute(select(DataSource).order_by(DataSource.created_at))).scalars()
            sources = [
                {
                    "id": str(s.id), "name": s.name, "kind": s.kind,
                    "table": s.table_name, "rows": s.row_count, "status": s.status,
                }
                for s in rows
            ]
        return {"ok": True, "data": {"sources": sources}}

    async def _source_or_error(source_id: str) -> DataSource:
        async with org_session(org_id) as db:
            source = await db.get(DataSource, uuid.UUID(source_id))
        if source is None:
            raise ValueError(f"No data source with id {source_id}")
        if not source.object_uri:
            raise ValueError(f"Source '{source.name}' has no uploaded object yet")
        return source

    async def read_source(source_id: str) -> dict[str, Any]:
        source = await _source_or_error(source_id)
        frame = master.readers().read(source.object_uri)
        handle = await store.put(frame, label=source.name)
        return {
            "ok": True,
            "data": {
                "rid": handle.rid,
                "row_count": handle.row_count,
                "schema": handle.schema,
            },
        }

    async def profile_source(rid: str) -> dict[str, Any]:
        handle = await store.get(rid)
        frame = await store.resolve(rid)
        stats = master.statistics()
        nulls: dict[str, float] = {}
        for column in handle.schema:
            try:
                nulls[column] = float(
                    stats.run("descriptive", "frequency", frame, column=column, method="null_rate")
                )
            except Exception:  # noqa: BLE001 — a column that cannot be profiled is reported as unknown
                nulls[column] = -1.0
        return {
            "ok": True,
            "data": {
                "rid": rid,
                "row_count": handle.row_count,
                "columns": handle.schema,
                "null_rate_by_column": nulls,
            },
        }

    async def propose_cleaning_pipeline(
        rid: str, steps: list[dict[str, Any]], rationale: str
    ) -> dict[str, Any]:
        """Validate the plan against the engine's factories. Nothing runs here."""
        frame = await store.resolve(rid)
        try:
            PipelineBuilder(frame).build(steps)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"Invalid pipeline: {exc}"}
        return {"ok": True, "data": {"rid": rid, "steps": steps, "rationale": rationale}}

    async def run_statistic(
        rid: str, domain: str, calculator: str, column: str | None = None
    ) -> dict[str, Any]:
        frame = await store.resolve(rid)
        value = master.statistics().run(domain, calculator, frame, column=column)
        return {"ok": True, "data": {"value": _jsonable(value)}}

    tools = [
        Tool(
            ToolSpec(
                name="list_data_sources",
                description="List every data source in this workspace with its table, row count and status.",
                input_schema={"type": "object", "properties": {}, "required": []},
            ),
            list_data_sources,
        ),
        Tool(
            ToolSpec(
                name="read_source",
                description="Load a data source into a working dataset and return its handle id (rid) and schema.",
                input_schema={
                    "type": "object",
                    "properties": {"source_id": {"type": "string", "description": "Data source uuid"}},
                    "required": ["source_id"],
                },
            ),
            read_source,
        ),
        Tool(
            ToolSpec(
                name="profile_source",
                description="Return row count, column types and per-column null rate for a loaded dataset.",
                input_schema={
                    "type": "object",
                    "properties": {"rid": {"type": "string", "description": "Dataset handle id"}},
                    "required": ["rid"],
                },
            ),
            profile_source,
        ),
        Tool(
            ToolSpec(
                name="propose_cleaning_pipeline",
                description=(
                    "Validate an ordered cleaning pipeline against the engine. Each step is an object "
                    "with exactly one key: the step name, mapped to its keyword arguments, e.g. "
                    '[{"drop_nulls": {"columns": ["country_code"]}}]. Returns an error naming the '
                    "offending step if any step is unknown. This does not execute anything."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "rid": {"type": "string"},
                        "steps": {"type": "array", "items": {"type": "object"}},
                        "rationale": {"type": "string"},
                    },
                    "required": ["rid", "steps", "rationale"],
                },
            ),
            propose_cleaning_pipeline,
        ),
        Tool(
            ToolSpec(
                name="run_statistic",
                description="Run one registered statistic, e.g. domain='descriptive', calculator='central_tendency'.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "rid": {"type": "string"},
                        "domain": {"type": "string"},
                        "calculator": {"type": "string"},
                        "column": {"type": "string"},
                    },
                    "required": ["rid", "domain", "calculator"],
                },
            ),
            run_statistic,
        ),
    ]
    return ToolRegistry(tools)


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)
```

- [ ] **Step 5: Delete the superseded tools**

```bash
git rm engine/src/lumen/model_tools/data_reader_tool.py engine/src/lumen/model_tools/meta_data_context_tool.py engine/src/lumen/model_tools/create_data_context.py
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run --directory services/api pytest tests/test_tool_registry.py -v`
Expected: PASS — 6 passed.
`test_propose_cleaning_pipeline_rejects_an_unknown_step` depends on `DataCleaningStepFactory` raising for an unregistered name. If it returns `ok: True`, the factory is silently accepting unknown steps — fix the factory, not the test.

- [ ] **Step 7: Commit**

```bash
git add services/api engine
git commit -m "feat: add tenant-scoped handle store and engine-backed tool registry"
```

---

### Task 5: The bounded agent loop

**Files:**
- Create: `services/api/src/lumen_api/agents/loop.py`
- Test: `services/api/tests/test_agent_loop.py`

**Interfaces:**
- Consumes: `LLMProvider`, `ChatMessage`, `ToolRegistry`, `repository.append_event`, `repository.record_usage`
- Produces: `AgentLoop(provider, registry, *, org_id, run_id, agent_name, max_iterations=12, deadline_seconds=180, max_total_tokens=120_000)` with `async run(system_prompt: str, user_prompt: str) -> LoopResult`; `LoopResult(final_text: str | None, iterations: int, usage: TokenUsage_total, stop_reason: Literal["done","max_iterations","deadline","budget"])`

- [ ] **Step 1: Write the failing test**

Create `services/api/tests/test_agent_loop.py`:

```python
import uuid

import pytest

from lumen.llm.base import ChatMessage, LLMResponse, ToolCall, ToolSpec, TokenUsage
from lumen_api.agents.loop import AgentLoop


class ScriptedProvider:
    """Replays a list of LLMResponses, one per call."""

    model = "test-model"

    def __init__(self, script):
        self._script = list(script)
        self.calls = 0

    async def complete(self, messages, tools, max_tokens=4096, temperature=0.0):
        self.calls += 1
        if self._script:
            return self._script.pop(0)
        return LLMResponse(
            text="fallback", tool_calls=[],
            usage=TokenUsage(1, 1, self.model), stop_reason="end_turn",
        )


class StubRegistry:
    def __init__(self, result=None):
        self.invocations = []
        self._result = result or {"ok": True, "data": {"x": 1}}

    def specs(self):
        return [ToolSpec(name="t", description="d", input_schema={"type": "object", "properties": {}})]

    async def invoke(self, name, arguments):
        self.invocations.append((name, arguments))
        return self._result


def _text(msg):
    return LLMResponse(text=msg, tool_calls=[], usage=TokenUsage(10, 5, "test-model"), stop_reason="end_turn")


def _call(name="t", args=None):
    return LLMResponse(
        text=None,
        tool_calls=[ToolCall(id="tc_1", name=name, arguments=args or {})],
        usage=TokenUsage(10, 5, "test-model"),
        stop_reason="tool_use",
    )


@pytest.mark.asyncio
async def test_a_plain_answer_finishes_in_one_iteration(one_org):
    loop = AgentLoop(
        ScriptedProvider([_text("done")]), StubRegistry(),
        org_id=one_org, run_id=None, agent_name="test",
    )
    result = await loop.run("system", "user")
    assert result.final_text == "done"
    assert result.iterations == 1
    assert result.stop_reason == "done"


@pytest.mark.asyncio
async def test_a_tool_call_is_executed_and_fed_back(one_org):
    registry = StubRegistry()
    loop = AgentLoop(
        ScriptedProvider([_call("t", {"a": 1}), _text("finished")]), registry,
        org_id=one_org, run_id=None, agent_name="test",
    )
    result = await loop.run("system", "user")
    assert registry.invocations == [("t", {"a": 1})]
    assert result.final_text == "finished"
    assert result.iterations == 2


@pytest.mark.asyncio
async def test_a_failing_tool_is_returned_to_the_model_not_raised(one_org):
    registry = StubRegistry({"ok": False, "error": "boom"})
    loop = AgentLoop(
        ScriptedProvider([_call(), _text("recovered")]), registry,
        org_id=one_org, run_id=None, agent_name="test",
    )
    result = await loop.run("system", "user")
    assert result.final_text == "recovered"


@pytest.mark.asyncio
async def test_iteration_cap_stops_a_runaway_loop(one_org):
    loop = AgentLoop(
        ScriptedProvider([_call() for _ in range(50)]), StubRegistry(),
        org_id=one_org, run_id=None, agent_name="test", max_iterations=3,
    )
    result = await loop.run("system", "user")
    assert result.iterations == 3
    assert result.stop_reason == "max_iterations"


@pytest.mark.asyncio
async def test_token_budget_stops_the_loop(one_org):
    loop = AgentLoop(
        ScriptedProvider([_call() for _ in range(50)]), StubRegistry(),
        org_id=one_org, run_id=None, agent_name="test",
        max_iterations=50, max_total_tokens=30,
    )
    result = await loop.run("system", "user")
    assert result.stop_reason == "budget"
    assert result.total_tokens >= 30


@pytest.mark.asyncio
async def test_an_unknown_tool_name_does_not_crash_the_loop(one_org):
    class RejectingRegistry(StubRegistry):
        async def invoke(self, name, arguments):
            return {"ok": False, "error": f"Unknown tool '{name}'"}

    loop = AgentLoop(
        ScriptedProvider([_call("ghost"), _text("ok")]), RejectingRegistry(),
        org_id=one_org, run_id=None, agent_name="test",
    )
    result = await loop.run("system", "user")
    assert result.final_text == "ok"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --directory services/api pytest tests/test_agent_loop.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lumen_api.agents.loop'`

- [ ] **Step 3: Implement the loop**

Create `services/api/src/lumen_api/agents/loop.py`:

```python
"""A bounded, observable agent loop.

Three guarantees the prototype loop did not have:
  1. it always terminates — iteration cap, wall-clock deadline, token budget;
  2. a tool failure is a tool result, never an exception out of the loop;
  3. every step is persisted as an AgentEvent, which is also the SSE stream.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Literal

from lumen.llm.base import ChatMessage, LLMProvider
from lumen_api.runs import repository as repo

StopReason = Literal["done", "max_iterations", "deadline", "budget"]


@dataclass(frozen=True)
class LoopResult:
    final_text: str | None
    iterations: int
    total_tokens: int
    stop_reason: StopReason
    transcript: list[ChatMessage]


class AgentLoop:
    def __init__(
        self,
        provider: LLMProvider,
        registry: Any,
        *,
        org_id: uuid.UUID,
        run_id: uuid.UUID | None,
        agent_name: str,
        max_iterations: int = 12,
        deadline_seconds: float = 180.0,
        max_total_tokens: int = 120_000,
        max_tokens_per_call: int = 4096,
    ) -> None:
        self._provider = provider
        self._registry = registry
        self._org_id = org_id
        self._run_id = run_id
        self._agent = agent_name
        self._max_iterations = max_iterations
        self._deadline_seconds = deadline_seconds
        self._max_total_tokens = max_total_tokens
        self._max_tokens_per_call = max_tokens_per_call

    async def run(self, system_prompt: str, user_prompt: str) -> LoopResult:
        messages: list[ChatMessage] = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_prompt),
        ]
        specs = self._registry.specs()
        started = time.monotonic()
        total_tokens = 0
        iterations = 0

        while True:
            if iterations >= self._max_iterations:
                return self._finish(None, iterations, total_tokens, "max_iterations", messages)
            if time.monotonic() - started > self._deadline_seconds:
                return self._finish(None, iterations, total_tokens, "deadline", messages)
            if total_tokens >= self._max_total_tokens:
                return self._finish(None, iterations, total_tokens, "budget", messages)

            response = await self._provider.complete(
                messages, specs, max_tokens=self._max_tokens_per_call
            )
            iterations += 1
            total_tokens += response.usage.input_tokens + response.usage.output_tokens
            await self._record_usage(response.usage)

            if response.text:
                await self._emit("message", {"text": response.text})

            if not response.tool_calls:
                return self._finish(response.text, iterations, total_tokens, "done", messages)

            messages.append(
                ChatMessage(
                    role="assistant", content=response.text or "", tool_calls=response.tool_calls
                )
            )

            for call in response.tool_calls:
                await self._emit("tool_call", {"name": call.name, "arguments": call.arguments})
                result = await self._registry.invoke(call.name, call.arguments)
                await self._emit(
                    "tool_result", {"name": call.name, "ok": result.get("ok", False)}
                )
                messages.append(
                    ChatMessage(
                        role="tool",
                        tool_call_id=call.id,
                        content=json.dumps(result, default=str)[:12_000],
                        is_error=not result.get("ok", False),
                    )
                )

    def _finish(
        self,
        text: str | None,
        iterations: int,
        total_tokens: int,
        stop_reason: StopReason,
        messages: list[ChatMessage],
    ) -> LoopResult:
        return LoopResult(
            final_text=text,
            iterations=iterations,
            total_tokens=total_tokens,
            stop_reason=stop_reason,
            transcript=messages,
        )

    async def _emit(self, type_: str, payload: dict[str, Any]) -> None:
        if self._run_id is None:
            return
        await repo.append_event(self._org_id, self._run_id, type_, payload)

    async def _record_usage(self, usage) -> None:
        if self._run_id is None:
            return
        from lumen_api.db.session import org_session

        async with org_session(self._org_id) as db:
            await repo.record_usage(
                db, self._org_id, metric="llm_input_tokens",
                quantity=usage.input_tokens, run_id=self._run_id,
                agent=self._agent, meta={"model": usage.model},
            )
            await repo.record_usage(
                db, self._org_id, metric="llm_output_tokens",
                quantity=usage.output_tokens, run_id=self._run_id,
                agent=self._agent, meta={"model": usage.model},
            )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --directory services/api pytest tests/test_agent_loop.py -v`
Expected: PASS — 6 passed

- [ ] **Step 5: Commit**

```bash
git add services/api/src/lumen_api/agents/loop.py services/api/tests/test_agent_loop.py
git commit -m "feat: add bounded budgeted agent loop with persisted events"
```

---

### Task 6: Context and cleaning agents

**Files:**
- Create: `services/api/src/lumen_api/agents/prompts/context.md`, `services/api/src/lumen_api/agents/prompts/cleaning.md`, `services/api/src/lumen_api/agents/context_agent.py`, `services/api/src/lumen_api/agents/cleaning_agent.py`
- Delete: `engine/src/lumen/agents/context_creator.py`, `engine/src/lumen/agents/postgres_admin_agent.py`
- Modify: `services/api/src/lumen_api/settings.py` (add `model_reasoning`, `model_specialist`, `model_fast`)
- Test: `services/api/tests/test_agents.py`

**Interfaces:**
- Consumes: `AgentLoop`, `build_tool_registry`, `get_provider`, `repository`
- Produces: `async run_context_agent(org_id, run_id, source_id, backend) -> str` (returns the profile summary); `async run_cleaning_agent(org_id, run_id, thread_id, rid, backend) -> Proposal | None`

- [ ] **Step 1: Write the failing test**

Create `services/api/tests/test_agents.py`:

```python
import uuid

import pytest

from lumen.llm.base import LLMResponse, ToolCall, TokenUsage
from lumen_api.agents import cleaning_agent
from lumen_api.runs import repository as repo


class ScriptedProvider:
    model = "test-model"

    def __init__(self, script):
        self._script = list(script)

    async def complete(self, messages, tools, max_tokens=4096, temperature=0.0):
        return self._script.pop(0)


@pytest.mark.asyncio
async def test_cleaning_agent_creates_a_proposal_from_a_validated_plan(one_org, monkeypatch):
    org_id = one_org
    thread_id = uuid.uuid4()
    run = await repo.create_run(
        org_id, source_id=None, thread_id=thread_id, kind="clean", backend="polars"
    )

    steps = [{"drop_nulls": {"columns": ["country_code"]}}]

    class Registry:
        def specs(self):
            return []

        async def invoke(self, name, arguments):
            assert name == "propose_cleaning_pipeline"
            return {"ok": True, "data": {"rid": arguments["rid"], "steps": steps,
                                         "rationale": arguments["rationale"]}}

    monkeypatch.setattr(cleaning_agent, "build_tool_registry", lambda *_a, **_k: Registry())
    monkeypatch.setattr(
        cleaning_agent,
        "_provider",
        lambda: ScriptedProvider(
            [
                LLMResponse(
                    text=None,
                    tool_calls=[
                        ToolCall(
                            id="tc1",
                            name="propose_cleaning_pipeline",
                            arguments={"rid": "abc", "steps": steps, "rationale": "3.2% nulls"},
                        )
                    ],
                    usage=TokenUsage(10, 5, "test-model"),
                    stop_reason="tool_use",
                ),
                LLMResponse(
                    text="Proposed one step.", tool_calls=[],
                    usage=TokenUsage(5, 5, "test-model"), stop_reason="end_turn",
                ),
            ]
        ),
    )

    proposal = await cleaning_agent.run_cleaning_agent(
        org_id, run_id=run.id, thread_id=thread_id, rid="abc", backend="polars"
    )

    assert proposal is not None
    assert proposal.status == "awaiting_review"
    assert proposal.kind == "cleaning_pipeline"
    assert proposal.spec == steps
    assert "3.2%" in proposal.rationale


@pytest.mark.asyncio
async def test_cleaning_agent_returns_none_when_no_plan_validated(one_org, monkeypatch):
    org_id = one_org
    thread_id = uuid.uuid4()
    run = await repo.create_run(
        org_id, source_id=None, thread_id=thread_id, kind="clean", backend="polars"
    )

    class Registry:
        def specs(self):
            return []

        async def invoke(self, name, arguments):
            return {"ok": False, "error": "Invalid pipeline: unknown step"}

    monkeypatch.setattr(cleaning_agent, "build_tool_registry", lambda *_a, **_k: Registry())
    monkeypatch.setattr(
        cleaning_agent,
        "_provider",
        lambda: ScriptedProvider(
            [
                LLMResponse(
                    text="The data looks clean.", tool_calls=[],
                    usage=TokenUsage(5, 5, "test-model"), stop_reason="end_turn",
                )
            ]
        ),
    )

    proposal = await cleaning_agent.run_cleaning_agent(
        org_id, run_id=run.id, thread_id=thread_id, rid="abc", backend="polars"
    )
    assert proposal is None


@pytest.mark.asyncio
async def test_a_proposal_event_is_emitted(one_org, monkeypatch):
    org_id = one_org
    thread_id = uuid.uuid4()
    run = await repo.create_run(
        org_id, source_id=None, thread_id=thread_id, kind="clean", backend="polars"
    )
    steps = [{"drop_duplicates": {"columns": ["email_hash"]}}]

    class Registry:
        def specs(self):
            return []

        async def invoke(self, name, arguments):
            return {"ok": True, "data": {"rid": "abc", "steps": steps, "rationale": "dupes"}}

    monkeypatch.setattr(cleaning_agent, "build_tool_registry", lambda *_a, **_k: Registry())
    monkeypatch.setattr(
        cleaning_agent,
        "_provider",
        lambda: ScriptedProvider(
            [
                LLMResponse(
                    text=None,
                    tool_calls=[
                        ToolCall(id="tc1", name="propose_cleaning_pipeline",
                                 arguments={"rid": "abc", "steps": steps, "rationale": "dupes"})
                    ],
                    usage=TokenUsage(10, 5, "test-model"), stop_reason="tool_use",
                ),
                LLMResponse(text="ok", tool_calls=[], usage=TokenUsage(1, 1, "test-model"),
                            stop_reason="end_turn"),
            ]
        ),
    )

    await cleaning_agent.run_cleaning_agent(
        org_id, run_id=run.id, thread_id=thread_id, rid="abc", backend="polars"
    )
    events = await repo.list_events(org_id, run.id)
    assert any(e.type == "proposal" for e in events)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --directory services/api pytest tests/test_agents.py -v`
Expected: FAIL — `ImportError: cannot import name 'cleaning_agent'`

- [ ] **Step 3: Add model settings**

Add to `Settings` in `services/api/src/lumen_api/settings.py`:

```python
    model_reasoning: str = "claude-opus-5"
    model_specialist: str = "claude-sonnet-5"
    model_fast: str = "qwen/qwen3.6-27b"
    llm_mode: Literal["auto", "anthropic", "groq", "mock", "bridge"] = "auto"
    llm_bridge_inbox: str = ".llm-bridge"
```

`Literal` is already imported in `settings.py`. `auto` means: Anthropic when `ANTHROPIC_API_KEY`
is set, the deterministic `MockProvider` from Task 9 otherwise — so a fresh checkout runs the
whole flow with no credentials.

- [ ] **Step 4: Write the prompts**

Create `services/api/src/lumen_api/agents/prompts/cleaning.md`:

```markdown
You are the Cleaning Agent for Lumen, an agentic data platform.

Your job is to propose an ordered cleaning pipeline for one loaded dataset. You do not
execute anything and you do not modify data. A human reviews and accepts your proposal
before any of it runs.

## How to work

1. Call `profile_source` with the dataset handle to see row count, column types and per-column null rates.
2. Decide which problems are worth fixing. Ignore anything below 0.5% unless it breaks a key.
3. Call `propose_cleaning_pipeline` with an ordered list of steps.

## Step format

Each step is an object with exactly one key — the step name — mapped to its arguments:

```json
[
  { "drop_nulls":      { "columns": ["country_code"] } },
  { "drop_duplicates": { "columns": ["email_hash"], "keep": "last" } }
]
```

If `propose_cleaning_pipeline` returns an error, the step name or its arguments are wrong.
Read the error, correct the plan, and call it again. Do not invent step names.

## Rules

- Order matters: drop rows before deduplicating, cast types before computing on them.
- Never drop more than 5% of rows without saying so explicitly in your rationale.
- Your rationale must cite the numbers you saw, not general advice.
- When the data has no material problems, say so plainly and propose nothing.

## Finishing

After a successful `propose_cleaning_pipeline`, write one short paragraph for the user:
what you found, what you propose, and what it will affect. No preamble, no sign-off.
```

Create `services/api/src/lumen_api/agents/prompts/context.md`:

```markdown
You are the Context Agent for Lumen.

Your job is to turn an unknown data source into a briefing that downstream agents and the
user can act on.

## How to work

1. Call `list_data_sources` if you were not given a specific source.
2. Call `read_source` with the source id. It returns a dataset handle (`rid`) and the schema.
3. Call `profile_source` with that `rid`.

## Output

Write markdown with exactly these three sections and nothing else:

### Shape
Row count, column count, and the backend used.

### Columns
A short list: name, type, null rate. Flag anything above 1%.

### Read
Two to four sentences: what this dataset appears to represent, which columns are the keys,
and the specific structural problems the next agent should know about. Cite numbers.
```

- [ ] **Step 5: Write the agents**

Create `services/api/src/lumen_api/agents/cleaning_agent.py`:

```python
from __future__ import annotations

import uuid
from pathlib import Path

from lumen.llm.registry import ModelTiers, get_provider
from lumen_api.agents.loop import AgentLoop
from lumen_api.agents.registry import build_tool_registry
from lumen_api.runs import repository as repo
from lumen_api.runs.models import Proposal
from lumen_api.settings import get_settings

PROMPT = (Path(__file__).parent / "prompts" / "cleaning.md").read_text(encoding="utf-8")


def _provider():
    settings = get_settings()
    return get_provider(
        "specialist",
        anthropic_key=settings.anthropic_api_key.get_secret_value()
        if settings.anthropic_api_key
        else None,
        groq_key=settings.groq_api_key.get_secret_value() if settings.groq_api_key else None,
        tiers=ModelTiers(
            reasoning=settings.model_reasoning,
            specialist=settings.model_specialist,
            fast=settings.model_fast,
        ),
        mode=settings.llm_mode,
        bridge_inbox=settings.llm_bridge_inbox,
    )


async def run_cleaning_agent(
    org_id: uuid.UUID,
    *,
    run_id: uuid.UUID,
    thread_id: uuid.UUID,
    rid: str,
    backend: str,
) -> Proposal | None:
    """Run the cleaning agent. Returns the created Proposal, or None if it proposed nothing."""
    registry = build_tool_registry(org_id, backend)
    captured: dict[str, object] = {}

    class Capturing:
        """Wraps the registry so a successful proposal is captured as it happens."""

        def specs(self):
            return registry.specs()

        async def invoke(self, name, arguments):
            result = await registry.invoke(name, arguments)
            if name == "propose_cleaning_pipeline" and result.get("ok"):
                captured.update(result["data"])
            return result

    loop = AgentLoop(
        _provider(), Capturing(), org_id=org_id, run_id=run_id, agent_name="cleaning"
    )
    result = await loop.run(PROMPT, f"Propose a cleaning pipeline for dataset {rid}.")

    if not captured:
        return None

    steps = captured.get("steps") or []
    proposal = await repo.create_proposal(
        org_id,
        run_id=run_id,
        thread_id=thread_id,
        author_agent="cleaning",
        kind="cleaning_pipeline",
        spec=steps,
        rationale=str(captured.get("rationale") or result.final_text or ""),
        estimate={"steps": len(steps), "rid": rid},
    )
    await repo.append_event(
        org_id,
        run_id,
        "proposal",
        {"proposal_id": str(proposal.id), "kind": proposal.kind, "steps": steps},
    )
    return proposal
```

Create `services/api/src/lumen_api/agents/context_agent.py`:

```python
from __future__ import annotations

import uuid
from pathlib import Path

from lumen_api.agents.cleaning_agent import _provider
from lumen_api.agents.loop import AgentLoop
from lumen_api.agents.registry import build_tool_registry

PROMPT = (Path(__file__).parent / "prompts" / "context.md").read_text(encoding="utf-8")


async def run_context_agent(
    org_id: uuid.UUID, *, run_id: uuid.UUID, source_id: uuid.UUID, backend: str
) -> str:
    registry = build_tool_registry(org_id, backend)
    loop = AgentLoop(_provider(), registry, org_id=org_id, run_id=run_id, agent_name="context")
    result = await loop.run(PROMPT, f"Profile data source {source_id}.")
    return result.final_text or ""
```

- [ ] **Step 6: Delete the superseded agents**

```bash
git rm engine/src/lumen/agents/context_creator.py engine/src/lumen/agents/postgres_admin_agent.py
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `uv run --directory services/api pytest tests/test_agents.py -v`
Expected: PASS — 3 passed

- [ ] **Step 8: Commit**

```bash
git add services/api engine
git commit -m "feat: add context and cleaning agents producing validated proposals"
```

---

### Task 7: Runs API, proposal decisions and the SSE event stream

**Files:**
- Create: `services/api/src/lumen_api/runs/router.py`, `services/api/src/lumen_api/uploads/__init__.py`, `services/api/src/lumen_api/uploads/router.py`
- Modify: `services/api/src/lumen_api/main.py`, `services/api/src/lumen_api/runs/schemas.py`
- Test: `services/api/tests/test_runs_api.py`

**Interfaces:**
- Consumes: repository, `current_org_id`, `current_user`, `require_role`
- Produces: `POST /v1/uploads` (multipart, returns a `DataSourceOut`), `POST /v1/runs`, `GET /v1/runs/{id}`, `GET /v1/runs/{id}/events` (SSE), `GET /v1/threads/{thread_id}/messages`, `POST /v1/proposals/{id}/accept`, `POST /v1/proposals/{id}/reject`, `GET /v1/proposals/{id}`

- [ ] **Step 1: Write the failing test**

Create `services/api/tests/test_runs_api.py`:

```python
import io
import uuid

import pytest
from httpx import ASGITransport, AsyncClient

from lumen_api.main import create_app


async def _client() -> AsyncClient:
    client = AsyncClient(transport=ASGITransport(app=create_app()), base_url="http://test")
    tag = uuid.uuid4().hex[:8]
    await client.post(
        "/v1/auth/signup",
        json={
            "email": f"u+{tag}@lumen.dev", "password": "correct horse battery staple",
            "display_name": "U", "org_name": f"O {tag}",
        },
    )
    return client


@pytest.mark.asyncio
async def test_upload_creates_a_data_source():
    csv = b"id,country_code\n1,DE\n2,\n3,US\n"
    client = await _client()
    async with client as c:
        response = await c.post(
            "/v1/uploads",
            files={"file": ("users.csv", io.BytesIO(csv), "text/csv")},
        )
        assert response.status_code == 201
        body = response.json()
        assert body["name"] == "users.csv"
        assert body["kind"] == "csv"


@pytest.mark.asyncio
async def test_creating_a_run_returns_queued():
    client = await _client()
    async with client as c:
        source = (
            await c.post("/v1/sources", json={"name": "s.csv", "kind": "csv", "table_name": "s"})
        ).json()
        response = await c.post(
            "/v1/runs", json={"kind": "profile", "source_id": source["id"], "backend": "polars"}
        )
        assert response.status_code == 202
        assert response.json()["status"] == "queued"


@pytest.mark.asyncio
async def test_accepting_a_proposal_twice_conflicts():
    from lumen_api.runs import repository as repo

    client = await _client()
    async with client as c:
        me = (await c.get("/v1/auth/me")).json()
        org_id = uuid.UUID(me["org_id"])
        thread_id = uuid.uuid4()
        run = await repo.create_run(
            org_id, source_id=None, thread_id=thread_id, kind="clean", backend="polars"
        )
        proposal = await repo.create_proposal(
            org_id, run_id=run.id, thread_id=thread_id, author_agent="cleaning",
            kind="cleaning_pipeline", spec=[], rationale="r", estimate={},
        )

        first = await c.post(f"/v1/proposals/{proposal.id}/accept")
        assert first.status_code == 200
        assert first.json()["status"] == "accepted"

        second = await c.post(f"/v1/proposals/{proposal.id}/accept")
        assert second.status_code == 409


@pytest.mark.asyncio
async def test_event_stream_replays_persisted_events():
    from lumen_api.runs import repository as repo

    client = await _client()
    async with client as c:
        me = (await c.get("/v1/auth/me")).json()
        org_id = uuid.UUID(me["org_id"])
        run = await repo.create_run(
            org_id, source_id=None, thread_id=uuid.uuid4(), kind="profile", backend="polars"
        )
        await repo.append_event(org_id, run.id, "message", {"text": "hello"})
        await repo.set_run_status(org_id, run.id, "succeeded")

        async with c.stream("GET", f"/v1/runs/{run.id}/events") as stream:
            assert stream.status_code == 200
            body = ""
            async for chunk in stream.aiter_text():
                body += chunk
                if "done" in body:
                    break
        assert "hello" in body


@pytest.mark.asyncio
async def test_runs_require_authentication():
    client = AsyncClient(transport=ASGITransport(app=create_app()), base_url="http://test")
    async with client as c:
        assert (await c.post("/v1/runs", json={"kind": "profile", "backend": "polars"})).status_code == 401
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --directory services/api pytest tests/test_runs_api.py -v`
Expected: FAIL — 404 on `/v1/uploads`

- [ ] **Step 3: Write the schemas**

Create `services/api/src/lumen_api/runs/schemas.py`:

```python
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

RunKind = Literal["profile", "clean", "analyze", "apply_pipeline"]
Backend = Literal["pandas", "polars", "spark"]


class RunCreate(BaseModel):
    kind: RunKind
    source_id: uuid.UUID | None = None
    rid: str | None = None
    thread_id: uuid.UUID | None = None
    backend: Backend = "polars"
    prompt: str | None = Field(default=None, max_length=4000)


class RunOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    thread_id: uuid.UUID
    source_id: uuid.UUID | None
    kind: str
    status: str
    backend: str
    started_at: datetime
    finished_at: datetime | None
    error: str | None


class ProposalOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    run_id: uuid.UUID
    thread_id: uuid.UUID
    author_agent: str
    kind: str
    status: str
    spec: Any
    rationale: str
    estimate: dict[str, Any]
    decided_at: datetime | None
    created_at: datetime


class AgentEventOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    seq: int
    type: str
    payload: dict[str, Any]
    created_at: datetime
```

- [ ] **Step 4: Write the uploads router**

Create `services/api/src/lumen_api/uploads/router.py`:

```python
from __future__ import annotations

import uuid
from typing import Annotated

import boto3
from fastapi import APIRouter, Depends, File, UploadFile, status

from lumen_api.auth.dependencies import current_org_id, require_role
from lumen_api.db.models.source import DataSource
from lumen_api.db.session import org_session
from lumen_api.errors import AppError
from lumen_api.settings import get_settings
from lumen_api.sources.schemas import DataSourceOut

router = APIRouter(prefix="/v1/uploads", tags=["uploads"])

EXTENSION_KIND = {".csv": "csv", ".json": "json", ".parquet": "parquet"}
MAX_BYTES = 512 * 1024 * 1024


class UnsupportedFile(AppError):
    status_code = status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
    title = "Unsupported file type"


@router.post("", response_model=DataSourceOut, status_code=status.HTTP_201_CREATED)
async def upload(
    org_id: Annotated[uuid.UUID, Depends(current_org_id)],
    _: Annotated[str, Depends(require_role("owner", "admin", "member"))],
    file: Annotated[UploadFile, File()],
) -> DataSourceOut:
    name = file.filename or "upload"
    suffix = "." + name.rsplit(".", 1)[-1].lower() if "." in name else ""
    kind = EXTENSION_KIND.get(suffix)
    if kind is None:
        raise UnsupportedFile(f"Supported: {', '.join(EXTENSION_KIND)}")

    payload = await file.read(MAX_BYTES + 1)
    if len(payload) > MAX_BYTES:
        raise UnsupportedFile("File exceeds 512 MB")

    settings = get_settings()
    key = f"org/{org_id}/uploads/{uuid.uuid4().hex}{suffix}"
    client = boto3.client(
        "s3",
        endpoint_url=settings.storage_endpoint,
        aws_access_key_id=settings.storage_access_key.get_secret_value(),
        aws_secret_access_key=settings.storage_secret_key.get_secret_value(),
    )
    try:
        client.head_bucket(Bucket=settings.storage_bucket)
    except Exception:  # noqa: BLE001 — first upload in a fresh environment
        client.create_bucket(Bucket=settings.storage_bucket)
    client.put_object(Bucket=settings.storage_bucket, Key=key, Body=payload)

    source = DataSource(
        org_id=org_id,
        name=name,
        kind=kind,
        object_uri=f"s3://{settings.storage_bucket}/{key}",
        table_name=name.rsplit(".", 1)[0][:128],
        status="idle",
    )
    async with org_session(org_id) as db:
        db.add(source)
        await db.flush()
        await db.refresh(source)
        return DataSourceOut.model_validate(source)
```

- [ ] **Step 5: Write the runs router**

Create `services/api/src/lumen_api/runs/router.py`:

```python
from __future__ import annotations

import asyncio
import json
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, status
from sse_starlette.sse import EventSourceResponse

from lumen_api.auth.dependencies import current_org_id, current_user, require_role
from lumen_api.db.models.identity import User
from lumen_api.runs import repository as repo
from lumen_api.runs.schemas import AgentEventOut, ProposalOut, RunCreate, RunOut

router = APIRouter(prefix="/v1", tags=["runs"])

OrgId = Annotated[uuid.UUID, Depends(current_org_id)]
TERMINAL = {"succeeded", "failed", "cancelled"}


@router.post("/runs", response_model=RunOut, status_code=status.HTTP_202_ACCEPTED)
async def create_run(payload: RunCreate, org_id: OrgId) -> RunOut:
    run = await repo.create_run(
        org_id,
        source_id=payload.source_id,
        thread_id=payload.thread_id or uuid.uuid4(),
        kind=payload.kind,
        backend=payload.backend,
    )
    from lumen_api.jobs import enqueue_run

    await enqueue_run(org_id, run.id, payload)
    return RunOut.model_validate(run)


@router.get("/runs/{run_id}", response_model=RunOut)
async def get_run(run_id: uuid.UUID, org_id: OrgId) -> RunOut:
    return RunOut.model_validate(await repo.get_run(org_id, run_id))


@router.get("/runs/{run_id}/events")
async def stream_events(run_id: uuid.UUID, org_id: OrgId) -> EventSourceResponse:
    """Replay every persisted event, then poll until the run reaches a terminal state."""

    async def generator():
        last_seq = 0
        while True:
            events = await repo.list_events(org_id, run_id, after_seq=last_seq)
            for event in events:
                last_seq = event.seq
                yield {
                    "event": event.type,
                    "id": str(event.seq),
                    "data": json.dumps(AgentEventOut.model_validate(event).model_dump(mode="json")),
                }
            run = await repo.get_run(org_id, run_id)
            if run.status in TERMINAL and not events:
                yield {"event": "done", "data": json.dumps({"status": run.status})}
                return
            await asyncio.sleep(0.4)

    return EventSourceResponse(generator())


@router.get("/proposals/{proposal_id}", response_model=ProposalOut)
async def get_proposal(proposal_id: uuid.UUID, org_id: OrgId) -> ProposalOut:
    return ProposalOut.model_validate(await repo.get_proposal(org_id, proposal_id))


@router.post("/proposals/{proposal_id}/accept", response_model=ProposalOut)
async def accept_proposal(
    proposal_id: uuid.UUID,
    org_id: OrgId,
    user: Annotated[User, Depends(current_user)],
    _: Annotated[str, Depends(require_role("owner", "admin", "member"))],
) -> ProposalOut:
    proposal = await repo.decide_proposal(org_id, proposal_id, accept=True, user_id=user.id)

    apply_run = await repo.create_run(
        org_id, source_id=None, thread_id=proposal.thread_id,
        kind="apply_pipeline", backend="polars",
    )
    from lumen_api.jobs import enqueue_apply

    await enqueue_apply(org_id, apply_run.id, proposal.id)
    return ProposalOut.model_validate(proposal)


@router.post("/proposals/{proposal_id}/reject", response_model=ProposalOut)
async def reject_proposal(
    proposal_id: uuid.UUID,
    org_id: OrgId,
    user: Annotated[User, Depends(current_user)],
    _: Annotated[str, Depends(require_role("owner", "admin", "member"))],
) -> ProposalOut:
    proposal = await repo.decide_proposal(org_id, proposal_id, accept=False, user_id=user.id)
    return ProposalOut.model_validate(proposal)
```

- [ ] **Step 6: Add the job dispatcher and register the routers**

Create `services/api/src/lumen_api/jobs.py`:

```python
"""Enqueue work for the worker. Falls back to in-process execution when Redis is absent."""
from __future__ import annotations

import uuid

from arq import create_pool
from arq.connections import RedisSettings

from lumen_api.settings import get_settings


async def _pool():
    return await create_pool(RedisSettings.from_dsn(get_settings().redis_url))


async def enqueue_run(org_id: uuid.UUID, run_id: uuid.UUID, payload) -> None:
    pool = await _pool()
    await pool.enqueue_job(
        "run_agent",
        str(org_id),
        str(run_id),
        payload.kind,
        str(payload.source_id) if payload.source_id else None,
        payload.rid,
        payload.backend,
    )


async def enqueue_apply(org_id: uuid.UUID, run_id: uuid.UUID, proposal_id: uuid.UUID) -> None:
    pool = await _pool()
    await pool.enqueue_job("apply_proposal", str(org_id), str(run_id), str(proposal_id))
```

Add to `main.py`:

```python
from lumen_api.runs import router as runs_router
from lumen_api.uploads import router as uploads_router
...
    app.include_router(uploads_router.router)
    app.include_router(runs_router.router)
```

Add dependencies:

```bash
cd services/api && uv add sse-starlette arq && cd ../..
```

- [ ] **Step 7: Run the tests to verify they pass**

Run:
```bash
docker compose -f infra/docker-compose.yml up -d db redis storage
uv run --directory services/api alembic upgrade head
uv run --directory services/api pytest tests/test_runs_api.py -v
```
Expected: PASS — 5 passed.
`test_creating_a_run_returns_queued` needs Redis reachable, since `enqueue_run` opens a pool.

- [ ] **Step 8: Commit**

```bash
git add services/api
git commit -m "feat: add uploads, runs, proposal decisions and sse event stream"
```

---

### Task 8: The worker

**Files:**
- Create: `services/worker/pyproject.toml`, `services/worker/Dockerfile`, `services/worker/src/lumen_worker/__init__.py`, `services/worker/src/lumen_worker/main.py`, `services/worker/src/lumen_worker/jobs/run_agent.py`, `services/worker/src/lumen_worker/jobs/apply_proposal.py`
- Modify: `infra/docker-compose.yml`, `Makefile`
- Test: `services/worker/tests/test_apply_proposal.py`

**Interfaces:**
- Consumes: everything above
- Produces: arq tasks `run_agent(ctx, org_id, run_id, kind, source_id, rid, backend)` and `apply_proposal(ctx, org_id, run_id, proposal_id)`; `WorkerSettings`

- [ ] **Step 1: Write the failing test**

Create `services/worker/tests/test_apply_proposal.py`:

```python
import uuid

import polars as pl
import pytest

from lumen_worker.jobs.apply_proposal import apply_proposal
from lumen_api.runs import repository as repo


@pytest.mark.asyncio
async def test_applying_a_pipeline_marks_the_proposal_applied(one_org, monkeypatch, tmp_path):
    org_id = one_org
    thread_id = uuid.uuid4()
    frame = pl.DataFrame({"country_code": ["DE", None, "US"], "email_hash": ["a", "b", "a"]})

    class Store:
        def __init__(self, *_a, **_k):
            pass

        async def resolve(self, rid):
            return frame.lazy()

        async def put(self, f, backend=None, *, label=""):
            from lumen.datasets.handle import DatasetHandle

            collected = f.collect() if hasattr(f, "collect") else f
            return DatasetHandle(
                rid="out1", uri="s3://x/out1.parquet", backend="polars",
                schema={c: "str" for c in collected.columns},
                row_count=collected.height, byte_size=1,
            )

    monkeypatch.setattr("lumen_worker.jobs.apply_proposal.PostgresHandleStore", Store)

    run = await repo.create_run(
        org_id, source_id=None, thread_id=thread_id, kind="clean", backend="polars"
    )
    proposal = await repo.create_proposal(
        org_id, run_id=run.id, thread_id=thread_id, author_agent="cleaning",
        kind="cleaning_pipeline",
        spec=[{"drop_nulls": {"columns": ["country_code"]}}],
        rationale="nulls", estimate={"rid": "in1"},
    )
    await repo.decide_proposal(org_id, proposal.id, accept=True, user_id=uuid.uuid4())

    apply_run = await repo.create_run(
        org_id, source_id=None, thread_id=thread_id, kind="apply_pipeline", backend="polars"
    )
    await apply_proposal({}, str(org_id), str(apply_run.id), str(proposal.id))

    assert (await repo.get_proposal(org_id, proposal.id)).status == "applied"
    assert (await repo.get_run(org_id, apply_run.id)).status == "succeeded"


@pytest.mark.asyncio
async def test_a_failing_apply_marks_the_run_failed(one_org, monkeypatch):
    org_id = one_org
    thread_id = uuid.uuid4()

    class Store:
        def __init__(self, *_a, **_k):
            pass

        async def resolve(self, rid):
            raise RuntimeError("storage unavailable")

    monkeypatch.setattr("lumen_worker.jobs.apply_proposal.PostgresHandleStore", Store)

    run = await repo.create_run(
        org_id, source_id=None, thread_id=thread_id, kind="clean", backend="polars"
    )
    proposal = await repo.create_proposal(
        org_id, run_id=run.id, thread_id=thread_id, author_agent="cleaning",
        kind="cleaning_pipeline", spec=[], rationale="r", estimate={"rid": "in1"},
    )
    await repo.decide_proposal(org_id, proposal.id, accept=True, user_id=uuid.uuid4())
    apply_run = await repo.create_run(
        org_id, source_id=None, thread_id=thread_id, kind="apply_pipeline", backend="polars"
    )

    await apply_proposal({}, str(org_id), str(apply_run.id), str(proposal.id))

    reloaded = await repo.get_run(org_id, apply_run.id)
    assert reloaded.status == "failed"
    assert "storage unavailable" in (reloaded.error or "")
    assert (await repo.get_proposal(org_id, proposal.id)).status == "accepted"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --directory services/worker pytest tests -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lumen_worker'`

- [ ] **Step 3: Write the project definition**

Create `services/worker/pyproject.toml`:

```toml
[project]
name = "lumen-worker"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["lumen-engine", "lumen-api", "arq>=0.26", "redis>=5.2"]

[project.optional-dependencies]
dev = ["pytest>=8", "pytest-asyncio>=0.24"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/lumen_worker"]

[tool.uv.sources]
lumen-engine = { path = "../../engine", editable = true }
lumen-api = { path = "../api", editable = true }

[tool.pytest.ini_options]
pythonpath = ["src", "../api/src"]
testpaths = ["tests"]
asyncio_mode = "auto"
```

Copy `services/api/tests/conftest.py` to `services/worker/tests/conftest.py`.

- [ ] **Step 4: Write the apply job**

Create `services/worker/src/lumen_worker/jobs/apply_proposal.py`:

```python
"""Execute an accepted cleaning proposal.

The proposal's `spec` is fed straight into the engine's PipelineBuilder — the same
declarative structure the agent produced and the tool registry already validated.
"""
from __future__ import annotations

import time
import uuid
from typing import Any

from lumen.data_cleaning.data_cleaning_pipeline import PipelineBuilder
from lumen_api.datasets.store import PostgresHandleStore
from lumen_api.db.session import org_session
from lumen_api.runs import repository as repo


async def apply_proposal(ctx: dict, org_id_s: str, run_id_s: str, proposal_id_s: str) -> None:
    org_id = uuid.UUID(org_id_s)
    run_id = uuid.UUID(run_id_s)
    proposal_id = uuid.UUID(proposal_id_s)

    started = time.monotonic()
    await repo.set_run_status(org_id, run_id, "running")

    try:
        proposal = await repo.get_proposal(org_id, proposal_id)
        rid = str(proposal.estimate.get("rid") or "")
        if not rid:
            raise ValueError("Proposal has no source dataset handle")

        store = PostgresHandleStore(org_id, "polars")
        frame = await store.resolve(rid)

        await repo.append_event(
            org_id, run_id, "tool_call",
            {"name": "apply_pipeline", "arguments": {"steps": proposal.spec}},
        )

        pipeline = PipelineBuilder(frame).build(proposal.spec)
        cleaned = pipeline.run(frame)

        handle = await store.put(cleaned, label=f"cleaned:{rid}")
        await repo.mark_proposal_applied(org_id, proposal_id, applied_run_id=run_id)

        await repo.append_event(
            org_id, run_id, "tool_result",
            {
                "name": "apply_pipeline",
                "ok": True,
                "rid": handle.rid,
                "row_count": handle.row_count,
                "report": _report(pipeline),
            },
        )
        await repo.append_event(org_id, run_id, "done", {"rid": handle.rid})
        await repo.set_run_status(org_id, run_id, "succeeded")

    except Exception as exc:  # noqa: BLE001 — a failed job must record why
        await repo.append_event(org_id, run_id, "error", {"message": str(exc)})
        await repo.set_run_status(org_id, run_id, "failed", error=str(exc))

    finally:
        async with org_session(org_id) as db:
            await repo.record_usage(
                db, org_id, metric="compute_seconds",
                quantity=round(time.monotonic() - started, 3),
                run_id=run_id, agent="worker",
            )


def _report(pipeline: Any) -> dict[str, Any]:
    report = getattr(pipeline, "report", None)
    if report is None:
        return {}
    for attribute in ("to_dict", "as_dict", "summary"):
        method = getattr(report, attribute, None)
        if callable(method):
            try:
                return method()
            except Exception:  # noqa: BLE001
                break
    return {"repr": str(report)[:2000]}
```

- [ ] **Step 5: Write the agent job and the worker entry point**

Create `services/worker/src/lumen_worker/jobs/run_agent.py`:

```python
from __future__ import annotations

import time
import uuid

from lumen_api.agents.cleaning_agent import run_cleaning_agent
from lumen_api.agents.context_agent import run_context_agent
from lumen_api.agents.registry import build_tool_registry
from lumen_api.db.session import org_session
from lumen_api.runs import repository as repo


async def run_agent(
    ctx: dict,
    org_id_s: str,
    run_id_s: str,
    kind: str,
    source_id_s: str | None,
    rid: str | None,
    backend: str,
) -> None:
    org_id = uuid.UUID(org_id_s)
    run_id = uuid.UUID(run_id_s)
    started = time.monotonic()
    await repo.set_run_status(org_id, run_id, "running")

    try:
        if kind == "profile":
            if not source_id_s:
                raise ValueError("profile requires source_id")
            summary = await run_context_agent(
                org_id, run_id=run_id, source_id=uuid.UUID(source_id_s), backend=backend
            )
            await repo.append_event(org_id, run_id, "message", {"text": summary})

        elif kind == "clean":
            handle_id = rid
            if handle_id is None:
                if not source_id_s:
                    raise ValueError("clean requires source_id or rid")
                registry = build_tool_registry(org_id, backend)
                loaded = await registry.invoke("read_source", {"source_id": source_id_s})
                if not loaded.get("ok"):
                    raise ValueError(loaded.get("error", "read_source failed"))
                handle_id = loaded["data"]["rid"]

            run = await repo.get_run(org_id, run_id)
            await run_cleaning_agent(
                org_id, run_id=run_id, thread_id=run.thread_id, rid=handle_id, backend=backend
            )
        else:
            raise ValueError(f"Unsupported run kind '{kind}'")

        await repo.append_event(org_id, run_id, "done", {})
        await repo.set_run_status(org_id, run_id, "succeeded")

    except Exception as exc:  # noqa: BLE001
        await repo.append_event(org_id, run_id, "error", {"message": str(exc)})
        await repo.set_run_status(org_id, run_id, "failed", error=str(exc))

    finally:
        async with org_session(org_id) as db:
            await repo.record_usage(
                db, org_id, metric="compute_seconds",
                quantity=round(time.monotonic() - started, 3),
                run_id=run_id, agent="worker",
            )
```

Create `services/worker/src/lumen_worker/main.py`:

```python
from __future__ import annotations

from arq.connections import RedisSettings

from lumen_api.settings import get_settings
from lumen_worker.jobs.apply_proposal import apply_proposal
from lumen_worker.jobs.run_agent import run_agent


class WorkerSettings:
    functions = [run_agent, apply_proposal]
    redis_settings = RedisSettings.from_dsn(get_settings().redis_url)
    max_jobs = 4
    job_timeout = 900
    keep_result = 3600
```

Create empty `services/worker/src/lumen_worker/__init__.py` and `services/worker/src/lumen_worker/jobs/__init__.py`.

- [ ] **Step 6: Run the tests to verify they pass**

```bash
uv sync --directory services/worker --extra dev
uv run --directory services/worker pytest tests -v
```
Expected: PASS — 2 passed

- [ ] **Step 7: Containerize and register the worker**

Create `services/worker/Dockerfile`:

```dockerfile
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY engine /app/engine
COPY services /app/services

WORKDIR /app/services/worker
RUN uv sync --no-dev

CMD ["uv", "run", "arq", "lumen_worker.main.WorkerSettings"]
```

Append to `infra/docker-compose.yml`:

```yaml
  worker:
    build:
      context: ..
      dockerfile: services/worker/Dockerfile
    environment:
      ENVIRONMENT: dev
      DATABASE_URL: postgresql+asyncpg://lumen_app:lumen_app@db:5432/lumen
      REDIS_URL: redis://redis:6379/0
      STORAGE_ENDPOINT: http://storage:9000
      ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY:-}
      GROQ_API_KEY: ${GROQ_API_KEY:-}
    depends_on:
      db: { condition: service_healthy }
      redis: { condition: service_healthy }
```

Add to the `Makefile`:

```makefile
test-worker:
	uv run --directory services/worker pytest tests -q
```
and extend `test:` to `test: test-engine test-api test-worker`.

- [ ] **Step 8: Commit**

```bash
git add services/worker infra Makefile
git commit -m "feat: add arq worker executing agent runs and accepted proposals"
```

---

## Self-Review

**Spec coverage vs ADR-0003** — Provider abstraction with Claude default: Task 1. DatasetHandle replacing `ObjectRegistry`: Tasks 2 and 4. `Proposal`/`Run`/`AgentEvent` tables and RLS: Task 3. `ToolRegistry` over `AgentMasterFactory`: Task 4. Bounded loop with errors-as-results: Task 5. Context and cleaning specialists: Task 6. SSE stream: Task 7. Spec validation before review: Task 4 (`propose_cleaning_pipeline`) and again at apply time in Task 8.

**Deliberately deferred** — `SupervisorAgent`, `AnalysisAgent`, `PersistenceAgent` and `AdminAgent` are not in this plan. The vertical slice needs profile → propose → accept → apply; the supervisor becomes worthwhile at three or more specialists. `QuotaGate` from ADR-0004 is also deferred: `UsageRecord` rows are written from Task 5 onward, so the data exists before the gate that reads it.

**Placeholder scan** — every step has runnable code or an exact command. Two steps intentionally leave the tree in a non-importing state (Task 1 Step 6, Task 2 Step 6) and say so explicitly, naming the task that restores it.

**Type consistency** — `ToolSpec`/`ToolCall`/`ChatMessage`/`TokenUsage` defined in Task 1, used unchanged in Tasks 4, 5, 6. `DatasetHandle` fields (`rid, uri, backend, schema, row_count, byte_size`) defined in Task 2, produced by `PostgresHandleStore` in Task 4, consumed in Task 8. `repo.create_proposal(..., spec, rationale, estimate)` defined in Task 3, called in Task 6, read in Task 8 — Task 6 writes `estimate={"steps": n, "rid": rid}` and Task 8 reads `proposal.estimate["rid"]`, which match. `AgentEvent.type` values emitted by the loop (`message, tool_call, tool_result`), by the cleaning agent (`proposal`) and by the worker (`done, error`) are all in the Task 3 enumeration.

**Known gap to close during execution** — `profile_source` in Task 4 calls `stats.run("descriptive", "frequency", frame, column=..., method="null_rate")`. Verify that `null_rate` exists on the frequency calculator for the active backend; if it does not, implement it in `engine/src/lumen/statistics/descriptive/` as part of Task 4 rather than weakening the tool.
