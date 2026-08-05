"""Resolve a provider from configuration.

`auto` is what makes the product runnable with no credentials: Anthropic when a
key is configured, Groq when only that is, and the deterministic MockProvider
otherwise. An *explicit* mode never silently falls back — asking for `anthropic`
without a key is a configuration error and is raised as one, because a
deployment that quietly degrades to a mock is worse than one that refuses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from lumen.llm.base import LLMProvider

Tier = Literal["reasoning", "specialist", "fast"]
Mode = Literal["auto", "anthropic", "groq", "mock", "bridge"]


@dataclass(frozen=True)
class ModelTiers:
    reasoning: str = "claude-opus-5"
    specialist: str = "claude-sonnet-5"
    fast: str = "qwen/qwen3.6-27b"


def resolve_mode(mode: Mode, *, anthropic_key: str | None, groq_key: str | None) -> Mode:
    if mode != "auto":
        return mode
    if anthropic_key:
        return "anthropic"
    if groq_key:
        return "groq"
    return "mock"


def get_provider(
    tier: Tier,
    *,
    anthropic_key: str | None,
    groq_key: str | None,
    tiers: ModelTiers | None = None,
    mode: Mode = "auto",
    bridge_inbox: str | None = None,
) -> LLMProvider:
    tiers = tiers or ModelTiers()
    resolved = resolve_mode(mode, anthropic_key=anthropic_key, groq_key=groq_key)

    if resolved == "mock":
        from lumen.llm.mock_provider import MockProvider

        return MockProvider()

    if resolved == "bridge":
        from lumen.llm.bridge_provider import BridgeProvider

        return BridgeProvider(inbox=bridge_inbox or ".llm-bridge")

    if resolved == "anthropic":
        if not anthropic_key:
            raise ValueError(
                f"ANTHROPIC_API_KEY is required for the '{tier}' tier under LLM_MODE=anthropic"
            )
        from anthropic import AsyncAnthropic

        from lumen.llm.anthropic_provider import AnthropicProvider

        model = tiers.reasoning if tier == "reasoning" else tiers.specialist
        return AnthropicProvider(client=AsyncAnthropic(api_key=anthropic_key), model=model)

    if not groq_key:
        raise ValueError("GROQ_API_KEY is required under LLM_MODE=groq")

    from openai import AsyncOpenAI

    from lumen.llm.groq_provider import GroqProvider

    client = AsyncOpenAI(api_key=groq_key, base_url="https://api.groq.com/openai/v1")
    return GroqProvider(client=client, model=tiers.fast)
