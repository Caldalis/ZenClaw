"""
对话历史 / 上下文窗口管理

AI 模型有上下文长度限制（如 GPT-4 = 128K tokens, Claude = 200K tokens）。
上下文窗口管理器负责:
  1. 从完整历史中截取适合的消息作为 AI 输入
  2. 确保 system prompt 始终在最前面
  3. 优先保留最近的消息（滑动窗口策略）
  4. 可选：加入语义搜索检索的相关旧消息

"""

from __future__ import annotations

from miniclaw.types.enums import Role
from miniclaw.types.messages import Message
from miniclaw.utils.logging import get_logger

logger = get_logger(__name__)


class ContextWindow:
    """上下文窗口管理器

    策略: 滑动窗口
      1. system prompt (固定)
      2. 最近的消息（从新到旧，直到 token 用尽）

    这是 OpenClaw 中最关键的模块之一:
    好的上下文管理 = AI 回复质量 + 不浪费 token
    """

    def __init__(self, max_tokens: int = 8000):
        self._max_tokens = max_tokens

    def build_context(
        self,
        system_prompt: str,
        messages: list[Message],
        relevant_memories: list[Message] | None = None,
    ) -> list[Message]:
        """构建发送给 AI 的上下文消息列表

        Args:
            system_prompt: 系统提示词
            messages: 完整的对话历史（时间正序）
            relevant_memories: 语义搜索召回的相关旧消息（可选）

        Returns:
            截取后的消息列表（包含 system prompt）

        构建逻辑:
          [system_prompt]           ← 固定
          [relevant_memories...]    ← 可选，语义召回的旧消息
          [recent_messages...]      ← 最近的消息（滑动窗口）
        """
        context: list[Message] = []
        used_tokens = 0

        # 1. System prompt 始终在最前
        sys_msg = Message(role=Role.SYSTEM, content=system_prompt)
        sys_tokens = sys_msg.token_estimate()
        used_tokens += sys_tokens
        context.append(sys_msg)

        remaining = self._max_tokens - used_tokens

        # 2. 预留 token 给语义召回的旧消息
        memory_messages: list[Message] = []
        if relevant_memories:
            memory_budget = min(remaining // 4, 1000)  # 最多用 1/4 或 1000 tokens
            for mem in relevant_memories:
                cost = mem.token_estimate()
                if memory_budget >= cost:
                    memory_messages.append(mem)
                    memory_budget -= cost
                    remaining -= cost

        # 3. 滑动窗口: 从最新消息往前取，直到 token 用尽
        recent: list[Message] = []
        for msg in reversed(messages):
            cost = msg.token_estimate()
            if remaining < cost:
                break
            recent.append(msg)
            remaining -= cost
        recent.reverse()  # 恢复时间正序

        # 4. 组装: system → memories → recent
        context.extend(memory_messages)
        context.extend(recent)

        logger.debug(
            "上下文构建: system=%d tokens, memories=%d条, recent=%d条, total≈%d tokens",
            sys_tokens, len(memory_messages), len(recent), self._max_tokens - remaining,
        )
        return context

    def estimate_total_tokens(self, messages: list[Message]) -> int:
        """估算消息列表的总 token 数（不含 system prompt）"""
        return sum(m.token_estimate() for m in messages)
