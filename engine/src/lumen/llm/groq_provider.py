"""OpenAI-compatible chat completions — the cheap tier, served by Groq."""

from __future__ import annotations

import json
from typing import Any

from lumen.llm.base import (
    ChatMessage,
    LLMProvider,
    LLMResponse,
    TokenUsage,
    ToolCall,
    ToolSpec,
)


class GroqProvider(LLMProvider):
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
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [self._encode(m) for m in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.input_schema,
                    },
                }
                for tool in tools
            ]
            kwargs["tool_choice"] = "auto"

        reply = await self._client.chat.completions.create(**kwargs)
        choice = reply.choices[0]
        message = choice.message

        calls = [
            ToolCall(
                id=call.id,
                name=call.function.name,
                arguments=json.loads(call.function.arguments or "{}"),
            )
            for call in (message.tool_calls or [])
        ]

        return LLMResponse(
            text=message.content,
            tool_calls=calls,
            usage=TokenUsage(
                input_tokens=reply.usage.prompt_tokens,
                output_tokens=reply.usage.completion_tokens,
                model=self.model,
            ),
            stop_reason=choice.finish_reason,
        )

    @staticmethod
    def _encode(message: ChatMessage) -> dict[str, Any]:
        if message.role == "tool":
            return {
                "role": "tool",
                "tool_call_id": message.tool_call_id,
                "content": message.content,
            }

        if message.role == "assistant" and message.tool_calls:
            return {
                "role": "assistant",
                "content": message.content or None,
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(call.arguments),
                        },
                    }
                    for call in message.tool_calls
                ],
            }

        return {"role": message.role, "content": message.content}
