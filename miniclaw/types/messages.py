"""
消息数据模型
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from .enums import Role


class ToolCall(BaseModel):
    """工具调用请求 — AI 要求执行某个工具

    对标 OpenAI 的 tool_calls 结构:
      { id, type: "function", function: { name, arguments } }
    """
    id: str = Field(default_factory=lambda: f"call_{uuid.uuid4().hex[:12]}")
    name: str                        # 技能名称，如 "calculator"
    arguments: dict[str, Any] = {}   # 调用参数


class ToolResult(BaseModel):
    """工具调用结果 — 技能执行后的返回值"""
    tool_call_id: str       # 关联的 ToolCall.id
    content: str            # 执行结果（字符串化）
    is_error: bool = False  # 是否为错误结果


class Message(BaseModel):
    """消息模型
    """
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    role: Role
    content: str = ""
    # AI 回复时可能携带工具调用请求
    tool_calls: list[ToolCall] | None = None
    # 工具执行结果（role=tool 时使用）
    tool_result: ToolResult | None = None
    # 元数据
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    session_id: str | None = None

    def is_tool_call(self) -> bool:
        """是否包含工具调用请求"""
        return bool(self.tool_calls)

    def token_estimate(self) -> int:
        """粗略估算 token 数（1 token ≈ 4 字符，中文 ≈ 2 字符）

        用于上下文窗口管理，不需要精确值。
        """
        text = self.content
        if self.tool_calls:
            text += str(self.tool_calls)
        if self.tool_result:
            text += self.tool_result.content
        # 简单启发式: ASCII 字符 / 4, 非 ASCII / 2
        ascii_chars = sum(1 for c in text if ord(c) < 128)
        non_ascii = len(text) - ascii_chars
        return ascii_chars // 4 + non_ascii // 2 + 1
