"""
AI 提供商抽象基类
test
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator

from miniclaw.types.messages import Message


class AIProvider(ABC):
    """AI 提供商抽象基类

    子类需要实现:
      - chat(): 发送消息并获取完整回复
      - chat_stream(): 发送消息并以流式返回回复
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """提供商名称"""
        ...

    @abstractmethod
    async def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> Message:
        """发送消息并获取完整回复

        Args:
            messages: 对话消息列表（含 system prompt）
            tools: 可用工具的 schema 列表（OpenAI function calling 格式）

        Returns:
            AI 的回复消息（可能包含 tool_calls）
        """
        ...
