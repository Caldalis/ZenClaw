"""
消息数据模型
test
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

