"""
OpenAI 提供商

将 OpenAI API (GPT-4, GPT-4o 等) 适配到统一的 AIProvider 接口。
支持:
  - 普通聊天完成 (chat completions)
  - 流式输出 (streaming)
  - 工具调用 (function calling / tool use)
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

from openai import AsyncOpenAI

from miniclaw.types.enums import Role
from miniclaw.types.messages import Message, ToolCall
from miniclaw.utils.logging import get_logger

from .base import AIProvider

logger = get_logger(__name__)


class OpenAIProvider(AIProvider):
    """OpenAI API 适配器"""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini",
                 base_url: str | None = None, max_tokens: int = 4096,
                 temperature: float = 0.7):
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    @property
    def name(self) -> str:
        return "openai"

    def _convert_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """将 MiniClaw 消息格式转换为 OpenAI API 格式

        这是适配器模式的核心: 屏蔽不同 API 的消息格式差异。
        """
        result = []
        for msg in messages:
            if msg.role == Role.TOOL and msg.tool_result:
                # 工具结果 → OpenAI 的 tool message
                result.append({
                    "role": "tool",
                    "tool_call_id": msg.tool_result.tool_call_id,
                    "content": msg.tool_result.content,
                })
            elif msg.role == Role.ASSISTANT and msg.tool_calls:
                # AI 回复含工具调用
                tool_calls_data = []
                for tc in msg.tool_calls:
                    tool_calls_data.append({
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                        },
                    })
                entry = {"role": "assistant", "content": msg.content or None, "tool_calls": tool_calls_data}
                result.append(entry)
            else:
                result.append({
                    "role": msg.role.value,
                    "content": msg.content,
                })
        return result

    async def chat(self, messages: list[Message],
                   tools: list[dict[str, Any]] | None = None) -> Message:
        """调用 OpenAI Chat Completions API"""
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": self._convert_messages(messages),
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = await self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]

        # 解析工具调用
        tool_calls = None
        if choice.message.tool_calls:
            tool_calls = []
            for tc in choice.message.tool_calls:
                try:
                    arguments = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    arguments = {}
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=arguments,
                ))

        return Message(
            role=Role.ASSISTANT,
            content=choice.message.content or "",
            tool_calls=tool_calls,
        )

    async def chat_stream(self, messages: list[Message],
                          tools: list[dict[str, Any]] | None = None) -> AsyncIterator[dict[str, Any]]:
        """流式调用 OpenAI API

        OpenAI 的流式 API 返回一系列 chunk，每个 chunk 包含:
          - 文本片段 (delta.content)
          - 工具调用片段 (delta.tool_calls)

        我们将这些 chunk 转换为统一的事件流。
        """
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": self._convert_messages(messages),
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        stream = await self._client.chat.completions.create(**kwargs)

        full_content = ""
        # 累积工具调用（流式中工具调用参数是分片到达的）
        tool_call_accum: dict[int, dict] = {}  # index → {id, name, arguments_str}

        async for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if not delta:
                continue

            # 文本片段
            if delta.content:
                full_content += delta.content
                yield {"type": "text_delta", "text": delta.content}

            # 工具调用片段
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_call_accum:
                        tool_call_accum[idx] = {
                            "id": tc_delta.id or "",
                            "name": "",
                            "arguments_str": "",
                        }
                    acc = tool_call_accum[idx]
                    if tc_delta.id:
                        acc["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            acc["name"] = tc_delta.function.name
                        if tc_delta.function.arguments:
                            acc["arguments_str"] += tc_delta.function.arguments

        # 组装完整的工具调用
        tool_calls = None
        if tool_call_accum:
            tool_calls = []
            for idx in sorted(tool_call_accum.keys()):
                acc = tool_call_accum[idx]
                try:
                    arguments = json.loads(acc["arguments_str"])
                except json.JSONDecodeError:
                    arguments = {}
                tc = ToolCall(id=acc["id"], name=acc["name"], arguments=arguments)
                tool_calls.append(tc)
                yield {"type": "tool_call", "id": tc.id, "name": tc.name, "arguments": arguments}

        # 最终事件: 包含完整消息
        yield {
            "type": "done",
            "message": Message(
                role=Role.ASSISTANT,
                content=full_content,
                tool_calls=tool_calls,
            ),
        }
