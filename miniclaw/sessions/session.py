"""
会话数据模型

Session 代表一次完整的对话会话，包含:
  - 唯一 ID
  - 会话内的消息历史
  - 元数据（创建时间、最后活跃时间等）

OpenClaw 中 Session 是连接用户和 Agent 的桥梁:
  用户通过 Channel 发送消息 → 关联到某个 Session → Agent 在该 Session 上下文中处理
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from pydantic import BaseModel, Field

from miniclaw.types.enums import SessionStatus
from miniclaw.types.messages import Message


class Session(BaseModel):
    """会话模型

    一个 Session 对应一段连续的对话。
    消息历史按时间顺序存储，Agent 处理时会截取最近的消息作为上下文。
    """
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    title: str = ""                # 会话标题（可自动生成）
    status: SessionStatus = SessionStatus.ACTIVE
    messages: list[Message] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def add_message(self, message: Message) -> None:
        """添加消息并更新时间戳"""
        message.session_id = self.id
        self.messages.append(message)
        self.updated_at = datetime.now(timezone.utc)

    def get_recent_messages(self, n: int = 50) -> list[Message]:
        """获取最近 N 条消息"""
        return self.messages[-n:]

    @property
    def message_count(self) -> int:
        return len(self.messages)
