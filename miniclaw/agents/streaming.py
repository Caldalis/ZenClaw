"""
流式响应处理

将 AI Provider 的原始流式输出转换为 MiniClaw 的 Event 事件流。
这是 Agent 和 Channel 之间的桥梁:
  Provider 输出 → StreamProcessor → Event 事件流 → Channel 展示

事件驱动是 OpenClaw 的核心设计:
  不直接返回字符串，而是产出一系列事件，让上层决定如何呈现。

Thinking 提取（Phase 2）:
  Master Agent 的 prompt 鼓励它在 <thinking>...</thinking> 块里推理。
  这里的 ThinkingExtractor 在 token 流上做状态机解析，把 thinking 内容
  抽出为独立 THINKING 事件，让 Channel 可以差异化渲染（默认隐藏/浅色）。
  正常文本仍走 TEXT_DELTA。
"""

from __future__ import annotations

from typing import Any, AsyncIterator

from miniclaw.types.events import Event
from miniclaw.types.messages import Message
from miniclaw.utils.logging import get_logger

logger = get_logger(__name__)


_THINKING_OPEN = "<thinking>"
_THINKING_CLOSE = "</thinking>"
# 任何以 "<" 开头但还没构成完整开/闭标签的字符都需要 buffer，
# 防止 "<thi" 这种被当成普通文本提前 flush。
_MAX_PARTIAL_LEN = max(len(_THINKING_OPEN), len(_THINKING_CLOSE))


class ThinkingExtractor:
    """流式状态机：从 token 流中提取 <thinking>...</thinking> 块。

    输入：增量文本片段 (delta)
    输出：[(kind, text), ...]，kind ∈ {"text", "thinking"}

    状态：
      - NORMAL：在 <thinking> 之外，输出 ("text", ...)
      - INSIDE：在 <thinking> 之内，输出 ("thinking", ...)
    Buffer：跨片段保存"看上去像标签前缀"的尾巴，避免误切。

    标签匹配是大小写敏感的（与 prompt 写法一致）。如果 LLM 输出了
    `<Thinking>` 这种大小写错误，会被当作普通文本——这是有意的，
    避免做模糊匹配带来的误识别。
    """

    def __init__(self) -> None:
        self._inside = False
        self._buffer = ""

    def feed(self, delta: str) -> list[tuple[str, str]]:
        """喂入一段增量，返回零或多个 (kind, text) 输出片段。"""
        if not delta:
            return []
        text = self._buffer + delta
        self._buffer = ""
        out: list[tuple[str, str]] = []
        i = 0
        n = len(text)

        while i < n:
            if not self._inside:
                # 在 NORMAL 状态找下一个 <thinking>
                idx = text.find(_THINKING_OPEN, i)
                if idx == -1:
                    # 没找到，但要看尾部有没有 <thinking 的部分前缀
                    safe_end = self._safe_emit_end(text, i, _THINKING_OPEN)
                    if safe_end > i:
                        out.append(("text", text[i:safe_end]))
                    self._buffer = text[safe_end:]
                    return out
                # 找到了 open tag
                if idx > i:
                    out.append(("text", text[i:idx]))
                i = idx + len(_THINKING_OPEN)
                self._inside = True
            else:
                # 在 INSIDE 状态找 </thinking>
                idx = text.find(_THINKING_CLOSE, i)
                if idx == -1:
                    safe_end = self._safe_emit_end(text, i, _THINKING_CLOSE)
                    if safe_end > i:
                        out.append(("thinking", text[i:safe_end]))
                    self._buffer = text[safe_end:]
                    return out
                if idx > i:
                    out.append(("thinking", text[i:idx]))
                i = idx + len(_THINKING_CLOSE)
                self._inside = False

        return out

    def flush(self) -> list[tuple[str, str]]:
        """流结束时把 buffer 里残留的内容刷出。"""
        if not self._buffer:
            return []
        kind = "thinking" if self._inside else "text"
        # 残留缓冲：在结束处必然不是部分标签了，直接当文本/思考输出
        out = [(kind, self._buffer)]
        self._buffer = ""
        return out

    @staticmethod
    def _safe_emit_end(text: str, start: int, target: str) -> int:
        """计算从 start 到 len(text) 这段中，"安全可发送"的结尾下标。

        规则：尾部最多保留 len(target)-1 个字符，且必须是 target 的某个前缀
        才需要保留——否则全部可发。这避免了把 "<th" 当成普通文本提前发出。
        """
        n = len(text)
        # 最多回退 max_partial-1 个字符
        max_back = min(len(target) - 1, n - start)
        for back in range(max_back, 0, -1):
            tail = text[n - back:]
            if target.startswith(tail):
                return n - back
        return n


async def process_stream(
    raw_stream: AsyncIterator[dict[str, Any]],
    session_id: str | None = None,
    sink: list | None = None,
    extract_thinking: bool = True,
) -> AsyncIterator[Event]:
    """将 Provider 的原始流转换为 Event 事件流

    Args:
        raw_stream: Provider.chat_stream() 产出的原始字典流
        session_id: 关联的会话 ID
        sink: 可选的可变列表；若传入，会在收到 done 事件时把完整 Message
            追加到 sink[0]，便于调用方拿到最终消息而不必二次请求模型。
        extract_thinking: 是否从文本流中提取 <thinking>...</thinking>
            块为独立 THINKING 事件。默认 True；关闭时全部走 TEXT_DELTA。

    Yields:
        Event 事件流

    Provider 原始流格式:
      {"type": "text_delta", "text": "..."}
      {"type": "tool_call", "id": "...", "name": "...", "arguments": {...}}
      {"type": "done", "message": Message}
    """
    extractor = ThinkingExtractor() if extract_thinking else None

    async for chunk in raw_stream:
        chunk_type = chunk.get("type")

        if chunk_type == "text_delta":
            text = chunk["text"]
            if extractor is None:
                yield Event.text_delta(text, session_id)
                continue
            for kind, segment in extractor.feed(text):
                if not segment:
                    continue
                if kind == "thinking":
                    yield Event.thinking(session_id=session_id, text=segment)
                else:
                    yield Event.text_delta(segment, session_id)

        elif chunk_type == "tool_call":
            yield Event.tool_call_start(
                name=chunk["name"],
                arguments=chunk.get("arguments", {}),
                session_id=session_id,
            )

        elif chunk_type == "done":
            # 流结束：把 thinking buffer 残留刷出，再发 text_done
            if extractor is not None:
                for kind, segment in extractor.flush():
                    if not segment:
                        continue
                    if kind == "thinking":
                        yield Event.thinking(session_id=session_id, text=segment)
                    else:
                        yield Event.text_delta(segment, session_id)

            message: Message = chunk.get("message")
            if sink is not None and message is not None:
                sink.append(message)
            if message and message.content:
                yield Event.text_done(message.content, session_id)

        else:
            logger.debug("未知的流式事件类型: %s", chunk_type)


__all__ = ["process_stream", "ThinkingExtractor"]
