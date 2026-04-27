"""
流式响应处理

将 AI Provider 的原始流式输出转换为 MiniClaw 的 Event 事件流。
这是 Agent 和 Channel 之间的桥梁:
  Provider 输出 → StreamProcessor → Event 事件流 → Channel 展示

事件驱动是 OpenClaw 的核心设计:
  不直接返回字符串，而是产出一系列事件，让上层决定如何呈现。
"""

from __future__ import annotations

from typing import Any, AsyncIterator

from miniclaw.types.events import Event
from miniclaw.types.messages import Message
from miniclaw.utils.logging import get_logger

logger = get_logger(__name__)


async def process_stream(
    raw_stream: AsyncIterator[dict[str, Any]],
    session_id: str | None = None,
    sink: list | None = None,
) -> AsyncIterator[Event]:
    """将 Provider 的原始流转换为 Event 事件流

    Args:
        raw_stream: Provider.chat_stream() 产出的原始字典流
        session_id: 关联的会话 ID
        sink: 可选的可变列表；若传入，会在收到 done 事件时把完整 Message
            追加到 sink[0]，便于调用方拿到最终消息而不必二次请求模型。

    Yields:
        Event 事件流

    Provider 原始流格式:
      {"type": "text_delta", "text": "..."}
      {"type": "tool_call", "id": "...", "name": "...", "arguments": {...}}
      {"type": "done", "message": Message}
    """
    async for chunk in raw_stream:
        chunk_type = chunk.get("type")

        if chunk_type == "text_delta":
            yield Event.text_delta(chunk["text"], session_id)

        elif chunk_type == "tool_call":
            yield Event.tool_call_start(
                name=chunk["name"],
                arguments=chunk.get("arguments", {}),
                session_id=session_id,
            )

        elif chunk_type == "done":
            # done 事件不需要转发，由上层处理
            # 但我们可以发出 text_done 事件
            message: Message = chunk.get("message")
            if sink is not None and message is not None:
                sink.append(message)
            if message and message.content:
                yield Event.text_done(message.content, session_id)

        else:
            logger.debug("未知的流式事件类型: %s", chunk_type)
