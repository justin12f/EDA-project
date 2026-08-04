"""A deterministic, network-free LLMProvider.

This is not a stub that replays canned text. It inspects the tool results
already in the transcript and derives its next move from them, so an end-to-end
run over a real dataset produces a proposal built from that dataset's real null
rates and duplicate counts.

Swapping in a real model changes the wording of the rationale, not the shape of
the flow — which is what makes this safe as the default when no API key is
configured, and what lets the same end-to-end assertions hold either way.
"""

from __future__ import annotations

import json
import re
from typing import Any

from lumen.llm.base import (
    ChatMessage,
    LLMProvider,
    LLMResponse,
    TokenUsage,
    ToolCall,
    ToolSpec,
)

UUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE
)
RID_RE = re.compile(r"\bdataset ([0-9a-f]{8,32})\b", re.IGNORECASE)
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
            last_error = results[-1].get("error", "unknown") if results else "unknown"
            return self._answer(
                messages,
                f"I could not complete this: every tool call failed. Last error: {last_error}",
            )

        # A proposal already validated — summarise and stop.
        proposal = _latest_ok(results, lambda data: "steps" in data)
        if proposal is not None:
            steps = proposal.get("steps") or []
            rationale = str(proposal.get("rationale", "")).strip()
            return self._answer(
                messages,
                f"Proposed {len(steps)} cleaning step(s). {rationale}".strip(),
            )

        # A profile in hand — turn it into a plan.
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

        # A dataset loaded — profile it.
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

    # ── response builders ───────────────────────────────────────────────────

    def _call(
        self, messages: list[ChatMessage], name: str, arguments: dict[str, Any]
    ) -> LLMResponse:
        return LLMResponse(
            text=None,
            tool_calls=[
                ToolCall(id=f"mock_{name}_{len(messages)}", name=name, arguments=arguments)
            ],
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


# ── transcript inspection ───────────────────────────────────────────────────


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


# ── planning ────────────────────────────────────────────────────────────────


def _plan(profile: dict[str, Any], threshold: float) -> tuple[list[dict[str, Any]], str]:
    """Derive cleaning steps from an observed profile.

    The threshold exists so a column with a handful of nulls in a million rows
    does not cost the tenant a row-dropping pipeline. Anything above it is worth
    a human's attention, which is the bar for proposing — not for acting.
    """
    null_rates: dict[str, float] = profile.get("null_rate_by_column") or {}
    duplicates: dict[str, int] = profile.get("duplicate_counts") or {}

    offenders = sorted(
        (column, rate)
        for column, rate in null_rates.items()
        if isinstance(rate, (int, float)) and rate > threshold
    )

    steps: list[dict[str, Any]] = []
    notes: list[str] = []

    # Step names come from the engine's registry, not from what a name *ought* to
    # be called. `drop_nulls` reads naturally and has never been registered.
    if offenders:
        steps.append(
            {
                "impute_categorical": {
                    "columns": [column for column, _ in offenders],
                    "strategy": "mode",
                }
            }
        )
        notes.append(
            ", ".join(f"{rate * 100:.1f}% nulls in {column}" for column, rate in offenders)
        )

    if any(count and count > 0 for count in duplicates.values()):
        steps.append({"remove_duplicates_rows": {}})
        notes.append(
            ", ".join(
                f"{count} duplicate {column} values"
                for column, count in sorted(duplicates.items())
                if count
            )
        )

    if not steps:
        rows = profile.get("row_count", "an unknown number of")
        return [], (
            f"every column is below the {threshold * 100:.1f}% null threshold across "
            f"{rows} rows, and no duplicate keys were reported."
        )

    return steps, "Found " + "; ".join(notes) + "."


def _usage(messages: list[ChatMessage], output: str, model: str) -> TokenUsage:
    prompt_chars = sum(len(m.content or "") for m in messages)
    return TokenUsage(
        input_tokens=max(1, prompt_chars // 4),
        output_tokens=max(1, len(output) // 4),
        model=model,
    )
