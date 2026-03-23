"""
技能抽象基类

每个 Skill 封装一个独立功能（如天气查询）
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Skill(ABC):
    """
        技能抽象基类 — 所有技能必须继承此类
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """技能唯一名称 — AI 通过此名称调用技能"""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """技能描述 — 告诉 AI 这个技能能做什么、何时使用"""
        ...

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """参数 JSON Schema — 告诉 AI 调用时需要传什么参数

        格式遵循 OpenAI function calling 规范:
        {
            "type": "object",
            "properties": { ... },
            "required": [...]
        }
        """
        ...

    @abstractmethod
    async def execute(self, **kwargs: Any) -> str:
        """执行技能

        Args:
            **kwargs: AI 传入的参数（已经过 JSON 解析）

        Returns:
            执行结果的字符串表示（会作为 tool_result 返回给 AI）
        """
        ...

    def to_tool_schema(self) -> dict[str, Any]:
        """生成 OpenAI function calling 格式的工具定义

        返回格式:
        {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "执行数学计算",
                "parameters": { ... }
            }
        }

        这个 schema 会被传给 AI API 的 tools 参数，
        让 AI 知道有哪些工具可用。
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
