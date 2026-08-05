"""Anthropic Messages API with native tool use."""

from __future__ import annotations

from typing import Any

from lumen.llm.base import (
    ChatMessage,
    LLMProvider,
    LLMResponse,
    TokenUsage,
    ToolCall,
    ToolSpec,
)


class AnthropicProvider(LLMProvider):
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
        # Anthropic takes the system prompt as a top-level parameter, not as a
        # message with role="system".
        system_parts = [m.content for m in messages if m.role == "system" and m.content]
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
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.input_schema,
                }
                for tool in tools
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
                {"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments}
                for call in message.tool_calls
            )
            return {"role": "assistant", "content": blocks}

        return {"role": message.role, "content": message.content}
