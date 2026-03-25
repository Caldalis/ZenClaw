"""
Anthropic 提供商

将 Anthropic API (Claude) 适配到统一的 AIProvider 接口。
Anthropic 的 API 格式与 OpenAI 不同，这个适配器封装了差异:
  - system prompt 是独立参数，不在 messages 中
  - 工具调用格式不同（content blocks vs function calling）
  - 流式格式不同（server-sent events）
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

from anthropic import AsyncAnthropic

from miniclaw.types.enums import Role
from miniclaw.types.messages import Message, ToolCall
from miniclaw.utils.logging import get_logger

from .base import AIProvider

logger = get_logger(__name__)


class AnthropicProvider(AIProvider):
    """Anthropic Claude API 适配器"""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514",
                 max_tokens: int = 4096, temperature: float = 0.7):
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._client = AsyncAnthropic(api_key=api_key)

    @property
    def name(self) -> str:
        return "anthropic"

    def _convert_messages(self, messages: list[Message]) -> tuple[str, list[dict[str, Any]]]:
        """转换消息格式

        Anthropic 的特殊之处:
          - system prompt 是独立的顶级参数
          - messages 只包含 user 和 assistant 消息
          - tool_result 是 user 消息中的 content block

        Returns:
            (system_prompt, messages_list)
        """
        system_prompt = ""
        result = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                system_prompt = msg.content
                continue

            if msg.role == Role.TOOL and msg.tool_result:
                # Anthropic 格式: 工具结果作为 user 消息的 content block
                result.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.tool_result.tool_call_id,
                        "content": msg.tool_result.content,
                        "is_error": msg.tool_result.is_error,
                    }],
                })
            elif msg.role == Role.ASSISTANT and msg.tool_calls:
                # Assistant 回复含工具调用 → content blocks
                content_blocks = []
                if msg.content:
                    content_blocks.append({"type": "text", "text": msg.content})
                for tc in msg.tool_calls:
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.arguments,
                    })
                result.append({"role": "assistant", "content": content_blocks})
            elif msg.role == Role.ASSISTANT:
                result.append({"role": "assistant", "content": msg.content})
            elif msg.role == Role.USER:
                result.append({"role": "user", "content": msg.content})

        return system_prompt, result

    def _convert_tools(self, tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        """将 OpenAI 格式的 tools 转为 Anthropic 格式

        OpenAI:    {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}
        Anthropic: {"name": ..., "description": ..., "input_schema": ...}
        """
        if not tools:
            return None
        result = []
        for tool in tools:
            func = tool.get("function", {})
            result.append({
                "name": func["name"],
                "description": func.get("description", ""),
                "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
            })
        return result

    async def chat(self, messages: list[Message],
                   tools: list[dict[str, Any]] | None = None) -> Message:
        """调用 Anthropic Messages API"""
        system_prompt, api_messages = self._convert_messages(messages)

        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": api_messages,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        anthropic_tools = self._convert_tools(tools)
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools

        response = await self._client.messages.create(**kwargs)

        # 解析回复
        content = ""
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input if isinstance(block.input, dict) else {},
                ))

        return Message(
            role=Role.ASSISTANT,
            content=content,
            tool_calls=tool_calls if tool_calls else None,
        )

    async def chat_stream(self, messages: list[Message],
                          tools: list[dict[str, Any]] | None = None) -> AsyncIterator[dict[str, Any]]:
        """流式调用 Anthropic API

        Anthropic 使用 Server-Sent Events 格式:
          content_block_start → content_block_delta → content_block_stop
        """
        system_prompt, api_messages = self._convert_messages(messages)

        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": api_messages,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        anthropic_tools = self._convert_tools(tools)
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools

        full_content = ""
        tool_calls: list[ToolCall] = []
        current_tool: dict[str, Any] | None = None

        async with self._client.messages.stream(**kwargs) as stream:
            async for event in stream:
                # 文本片段
                if event.type == "content_block_delta":
                    if hasattr(event.delta, "text"):
                        full_content += event.delta.text
                        yield {"type": "text_delta", "text": event.delta.text}
                    elif hasattr(event.delta, "partial_json"):
                        # 工具调用参数片段
                        if current_tool is not None:
                            current_tool["arguments_str"] += event.delta.partial_json

                elif event.type == "content_block_start":
                    if hasattr(event.content_block, "type") and event.content_block.type == "tool_use":
                        current_tool = {
                            "id": event.content_block.id,
                            "name": event.content_block.name,
                            "arguments_str": "",
                        }

                elif event.type == "content_block_stop":
                    if current_tool is not None:
                        try:
                            arguments = json.loads(current_tool["arguments_str"]) if current_tool["arguments_str"] else {}
                        except json.JSONDecodeError:
                            arguments = {}
                        tc = ToolCall(
                            id=current_tool["id"],
                            name=current_tool["name"],
                            arguments=arguments,
                        )
                        tool_calls.append(tc)
                        yield {"type": "tool_call", "id": tc.id, "name": tc.name, "arguments": arguments}
                        current_tool = None

        yield {
            "type": "done",
            "message": Message(
                role=Role.ASSISTANT,
                content=full_content,
                tool_calls=tool_calls if tool_calls else None,
            ),
        }
