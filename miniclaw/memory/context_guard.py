"""
上下文守卫 (Context Guard)

职责：
1. Token 记账 — 估算当前消息列表的 token 总量
2. 压缩触发 — 判断是否需要触发上下文压缩
3. 压缩分割 — 将消息分为「待压缩」和「保留」两部分
4. 工具输出截断 — 防止超大工具输出导致 Context Overflow
"""

from __future__ import annotations

from miniclaw.types.enums import Role
from miniclaw.types.messages import Message


COMPACTION_PROMPT_TEMPLATE = """\
以下是 AI 助手与用户的对话历史记录。请将其压缩为一段简洁的摘要，\
保留所有关键信息、决策、工具调用结果和重要结论。\
摘要应足够详细，让 AI 助手能够继续进行对话，同时比原始消息更紧凑。

对话历史:
{conversation_text}

请提供简洁的摘要（用中文）:"""


class ContextGuard:
    """上下文守卫 — 无状态辅助类，提供 token 预算和压缩决策

    设计为无状态：不持有消息列表，方便测试和复用。
    Agent 持有状态，调用 ContextGuard 方法获取决策。
    """

    def __init__(self, max_context_tokens: int, compaction_threshold: float = 0.85):
        self._max_tokens = max_context_tokens
        self._threshold = compaction_threshold

    def estimate_tokens(self, messages: list[Message]) -> int:
        """估算消息列表的总 token 数"""
        return sum(m.token_estimate() for m in messages)

    def should_compact(self, messages: list[Message]) -> bool:
        """判断是否需要触发上下文压缩"""
        used = self.estimate_tokens(messages)
        threshold_tokens = int(self._max_tokens * self._threshold)
        return used >= threshold_tokens

    def select_messages_for_compaction(
        self,
        messages: list[Message],
        keep_recent: int = 6,
    ) -> tuple[list[Message], list[Message]]:
        """将消息分割为「待压缩」和「保留」两部分

        保留策略：
        - 所有 SYSTEM 消息始终保留
        - 最近 keep_recent 条非 SYSTEM 消息始终保留
        - 其余旧消息进入压缩队列

        Returns:
            (to_compact, to_keep)
        """
        system_msgs = [m for m in messages if m.role == Role.SYSTEM]
        non_system = [m for m in messages if m.role != Role.SYSTEM]

        if len(non_system) <= keep_recent:
            return [], list(messages)

        to_compact = non_system[:-keep_recent]
        to_keep_non_system = non_system[-keep_recent:]

        return to_compact, system_msgs + to_keep_non_system

    @staticmethod
    def truncate_tool_result(content: str, max_bytes: int = 102400) -> str:
        """截断超大工具输出，防止 Context Overflow

        如果输出超过 max_bytes（UTF-8），截断并追加提示信息，
        让 AI 知道输出被截断、可以换用其他工具（如 grep）。

        Args:
            content: 工具输出文本
            max_bytes: 最大允许字节数（默认 100KB）

        Returns:
            可能被截断的输出文本
        """
        encoded = content.encode("utf-8")
        if len(encoded) <= max_bytes:
            return content

        actual_mb = len(encoded) / (1024 * 1024)
        truncated = encoded[:max_bytes].decode("utf-8", errors="ignore")
        notice = f"\n\n[truncated: tool output exceeded 100KB (actual: {actual_mb:.2f}MB)]"
        return truncated + notice
