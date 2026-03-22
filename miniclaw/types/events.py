"""
事件数据模型
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from .enums import EventType


class Event(BaseModel):
    """事件基类

    示例事件流:
      Event(THINKING)
      Event(TEXT_DELTA, data={"text": "让我"})
      Event(TEXT_DELTA, data={"text": "查一下"})
      Event(TOOL_CALL_START, data={"name": "weather", ...})
      Event(TOOL_CALL_RESULT, data={"result": "北京 25°C"})
      Event(TEXT_DELTA, data={"text": "北京今天25度"})
      Event(TEXT_DONE)
      Event(DONE)
    """
    event_type: EventType
    data: dict[str, Any] = {}
    session_id: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def text_delta(cls, text: str, session_id: str | None = None) -> Event:
        """创建文本片段事件"""
        return cls(event_type=EventType.TEXT_DELTA, data={"text": text}, session_id=session_id)

    @classmethod
    def text_done(cls, full_text: str = "", session_id: str | None = None) -> Event:
        """创建文本完成事件"""
        return cls(event_type=EventType.TEXT_DONE, data={"text": full_text}, session_id=session_id)

    @classmethod
    def tool_call_start(cls, name: str, arguments: dict, session_id: str | None = None) -> Event:
        """创建工具调用开始事件"""
        return cls(
            event_type=EventType.TOOL_CALL_START,
            data={"name": name, "arguments": arguments},
            session_id=session_id,
        )

    @classmethod
    def tool_call_result(cls, name: str, result: str, session_id: str | None = None) -> Event:
        """创建工具调用结果事件"""
        return cls(
            event_type=EventType.TOOL_CALL_RESULT,
            data={"name": name, "result": result},
            session_id=session_id,
        )

    @classmethod
    def thinking(cls, session_id: str | None = None) -> Event:
        return cls(event_type=EventType.THINKING, session_id=session_id)

    @classmethod
    def error(cls, message: str, session_id: str | None = None) -> Event:
        return cls(event_type=EventType.ERROR, data={"message": message}, session_id=session_id)

    @classmethod
    def done(cls, session_id: str | None = None) -> Event:
        return cls(event_type=EventType.DONE, session_id=session_id)
