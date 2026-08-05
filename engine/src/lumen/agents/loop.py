"""A bounded, observable agent loop.

Three guarantees the prototype loop did not have:

1. **It always terminates.** Iteration cap, wall-clock deadline and token budget.
   The original `while True:` had none of them, so a model that kept calling
   tools looped forever on the tenant's money.
2. **A tool failure is a result, not an exception.** The original raised out of
   the loop on the first bad path, which meant one unreadable file killed the
   run instead of letting the model try something else.
3. **Every step is observable.** Steps go to an injected `EventSink`, so the
   loop itself has no idea whether events are being persisted, streamed or
   discarded — which is what makes it testable without a database.

The sink and the tool registry are protocols rather than concrete types on
purpose: this module lives in the engine and must stay free of web and tenant
concerns.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from lumen.llm.base import ChatMessage, LLMProvider, TokenUsage

StopReason = Literal["done", "max_iterations", "deadline", "budget"]

# A tool result is truncated before it goes back to the model. 12k characters is
# roughly 3k tokens — enough for a schema or a profile, small enough that one
# chatty tool cannot consume the whole budget.
MAX_TOOL_RESULT_CHARS = 12_000


class ToolRegistry(Protocol):
    def specs(self) -> list[Any]: ...

    async def invoke(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]: ...


class EventSink(Protocol):
    async def emit(self, type_: str, payload: dict[str, Any]) -> None: ...

    async def record_usage(self, usage: TokenUsage) -> None: ...


class NullSink:
    """Discards everything. The default, so the loop runs anywhere."""

    async def emit(self, type_: str, payload: dict[str, Any]) -> None:
        return None

    async def record_usage(self, usage: TokenUsage) -> None:
        return None


@dataclass
class CollectingSink:
    """Keeps events in memory. Useful in tests and for a dry run."""

    events: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    usage: list[TokenUsage] = field(default_factory=list)

    async def emit(self, type_: str, payload: dict[str, Any]) -> None:
        self.events.append((type_, payload))

    async def record_usage(self, usage: TokenUsage) -> None:
        self.usage.append(usage)


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
        registry: ToolRegistry,
        *,
        agent_name: str = "agent",
        sink: EventSink | None = None,
        max_iterations: int = 12,
        deadline_seconds: float = 180.0,
        max_total_tokens: int = 120_000,
        max_tokens_per_call: int = 4096,
    ) -> None:
        self._provider = provider
        self._registry = registry
        self._agent = agent_name
        self._sink = sink or NullSink()
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
                return self._stop(None, iterations, total_tokens, "max_iterations", messages)
            if time.monotonic() - started > self._deadline_seconds:
                return self._stop(None, iterations, total_tokens, "deadline", messages)
            if total_tokens >= self._max_total_tokens:
                return self._stop(None, iterations, total_tokens, "budget", messages)

            response = await self._provider.complete(
                messages, specs, max_tokens=self._max_tokens_per_call
            )
            iterations += 1
            total_tokens += response.usage.total
            await self._sink.record_usage(response.usage)

            if response.text:
                await self._sink.emit("message", {"text": response.text})

            if not response.tool_calls:
                return self._stop(response.text, iterations, total_tokens, "done", messages)

            messages.append(
                ChatMessage(
                    role="assistant",
                    content=response.text or "",
                    tool_calls=response.tool_calls,
                )
            )

            for call in response.tool_calls:
                await self._sink.emit(
                    "tool_call", {"name": call.name, "arguments": call.arguments}
                )
                result = await self._invoke(call.name, call.arguments)
                ok = bool(result.get("ok"))
                await self._sink.emit("tool_result", {"name": call.name, "ok": ok})
                messages.append(
                    ChatMessage(
                        role="tool",
                        tool_call_id=call.id,
                        content=json.dumps(result, default=str)[:MAX_TOOL_RESULT_CHARS],
                        is_error=not ok,
                    )
                )

    async def _invoke(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Never let a tool raise out of the loop — the model can recover, a crash cannot."""
        try:
            return await self._registry.invoke(name, arguments)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def _stop(
        self,
        text: str | None,
        iterations: int,
        total_tokens: int,
        reason: StopReason,
        messages: list[ChatMessage],
    ) -> LoopResult:
        return LoopResult(
            final_text=text,
            iterations=iterations,
            total_tokens=total_tokens,
            stop_reason=reason,
            transcript=messages,
        )
