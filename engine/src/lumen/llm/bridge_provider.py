"""An LLMProvider whose replies come from a file, not a network call.

Each `complete()` writes `NNN.request.json` into the inbox and polls for
`NNN.response.json`. A person — or an agent driving the session — reads the
request and writes the reply.

Two uses: exercising a path `MockProvider` does not cover without spending a
token, and demonstrating the agent flow with a human in the model's seat.

Response file format:

    {"text": "…" | null,
     "tool_calls": [{"name": "…", "arguments": {…}}, …]}
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from pathlib import Path

from lumen.llm.base import (
    ChatMessage,
    LLMProvider,
    LLMResponse,
    TokenUsage,
    ToolCall,
    ToolSpec,
)


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
                return LLMResponse(
                    text=payload.get("text"),
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
