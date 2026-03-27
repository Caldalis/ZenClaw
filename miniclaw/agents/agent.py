"""
核心代理

Agent 是整个系统的核心，实现了 AI 对话 + 工具调用循环:

  ┌─────────────────────────────────────────────┐
  │              Agent.process_message()         │
  │                                             │
  │  1. 构建上下文 (ContextWindow)               │
  │  2. 调用 AI (Provider)                       │
  │  3. AI 回复是否包含工具调用?                   │
  │     ├─ 是 → 执行工具 (ToolExecutor)           │
  │     │       将结果加入上下文                   │
  │     │       回到步骤 2 (循环)                  │
  │     └─ 否 → 返回最终文本回复                   │
  │                                             │
  │  产出事件流: AsyncIterator[Event]             │
  └─────────────────────────────────────────────┘

关键设计:
  - 事件驱动: 不返回字符串，产出 Event 流
  - 工具调用循环: 多轮调用直到 AI 不再请求工具
  - max_iterations: 防止死循环（AI 可能无限循环调用工具）
"""

from __future__ import annotations

from typing import AsyncIterator

from miniclaw.agents.providers.registry import ProviderRegistry
from miniclaw.agents.streaming import process_stream
from miniclaw.agents.tool_executor import ToolExecutor
from miniclaw.config.settings import AgentConfig
from miniclaw.memory.base import MemoryStore
from miniclaw.sessions.history import ContextWindow
from miniclaw.sessions.manager import SessionManager
from miniclaw.tools.registry import ToolRegistry
from miniclaw.types.enums import Role
from miniclaw.types.events import Event
from miniclaw.types.messages import Message
from miniclaw.utils.errors import ProviderError
from miniclaw.utils.logging import get_logger

logger = get_logger(__name__)


class Agent:
    """核心代理 — 处理用户消息，调用 AI 和工具

    这是 MiniClaw 最核心的类，对标 OpenClaw 的 Agent / AgentRunner。

    协作关系:
      SessionManager → 提供对话历史
      ContextWindow  → 截取合适的上下文
      ProviderRegistry → 调用 AI
      ToolExecutor   → 执行工具
      ToolRegistry  → 提供可用工具列表
    """

    def __init__(
        self,
        config: AgentConfig,
        provider_registry: ProviderRegistry,
        tool_registry: ToolRegistry,
        session_manager: SessionManager,
        memory_store: MemoryStore,
    ):
        self._config = config
        self._providers = provider_registry
        self._tools = tool_registry
        self._session_mgr = session_manager
        self._memory = memory_store
        self._tool_executor = ToolExecutor(tool_registry)
        self._context_window = ContextWindow(config.max_context_tokens)

    async def process_message(self, user_message: Message, session_id: str) -> AsyncIterator[Event]:
        """处理用户消息 — Agent 的核心入口

        这是一个 async generator，产出 Event 事件流。
        上层（Gateway/Channel）通过 async for 消费事件。

        流程:
          1. 保存用户消息
          2. 构建上下文
          3. 调用 AI（可能多轮工具调用循环）
          4. 保存 AI 回复
          5. 产出事件流
        """
        # Step 1: 保存用户消息
        await self._session_mgr.add_message(session_id, user_message)
        yield Event.thinking(session_id)

        # Step 2: 获取工具 schema
        tool_schemas = self._tools.get_tool_schemas() if self._tools.tool_count > 0 else None

        # Step 3: 工具调用循环
        iteration = 0
        while iteration < self._config.max_iterations:
            iteration += 1
            logger.debug("Agent 循环 #%d", iteration)

            # 构建上下文
            session = await self._session_mgr.get_session(session_id)
            if session is None:
                yield Event.error("会话不存在", session_id)
                yield Event.done(session_id)
                return

            # 可选: 语义搜索相关记忆
            relevant = []
            if user_message.content:
                try:
                    relevant = await self._memory.search(session_id, user_message.content, top_k=3)
                except Exception:
                    pass

            context = self._context_window.build_context(
                system_prompt=self._config.system_prompt,
                messages=session.messages,
                relevant_memories=relevant,
            )

            # 调用 AI
            try:
                ai_message = await self._call_ai_with_failover(context, tool_schemas)
            except ProviderError as e:
                yield Event.error(f"AI 调用失败: {e}", session_id)
                yield Event.done(session_id)
                return

            # 保存 AI 回复
            await self._session_mgr.add_message(session_id, ai_message)

            # 检查是否有工具调用
            if ai_message.is_tool_call():
                # 执行工具
                tool_messages, tool_events = await self._tool_executor.execute(
                    ai_message.tool_calls, session_id
                )
                # 产出工具调用事件
                for event in tool_events:
                    yield event
                # 保存工具结果
                for msg in tool_messages:
                    await self._session_mgr.add_message(session_id, msg)
                # 继续循环: AI 需要看到工具结果后再回复
                continue
            else:
                # 没有工具调用 → 最终回复
                # 流式输出最终文本
                if ai_message.content:
                    yield Event.text_delta(ai_message.content, session_id)
                    yield Event.text_done(ai_message.content, session_id)
                yield Event.done(session_id)
                return

        # 超过最大迭代次数
        logger.warning("Agent 达到最大迭代次数 %d", self._config.max_iterations)
        yield Event.error(f"处理轮数超过上限 ({self._config.max_iterations})", session_id)
        yield Event.done(session_id)

    async def process_message_stream(self, user_message: Message, session_id: str) -> AsyncIterator[Event]:
        """流式处理用户消息 — 使用 Provider 的流式 API

        与 process_message() 类似，但使用流式 API 逐字输出。
        适合 CLI 和 WebSocket 等需要实时显示的场景。
        """
        await self._session_mgr.add_message(session_id, user_message)
        yield Event.thinking(session_id)

        tool_schemas = self._tools.get_tool_schemas() if self._tools.tool_count > 0 else None

        iteration = 0
        while iteration < self._config.max_iterations:
            iteration += 1

            session = await self._session_mgr.get_session(session_id)
            if session is None:
                yield Event.error("会话不存在", session_id)
                yield Event.done(session_id)
                return

            relevant = []
            if user_message.content:
                try:
                    relevant = await self._memory.search(session_id, user_message.content, top_k=3)
                except Exception:
                    pass

            context = self._context_window.build_context(
                system_prompt=self._config.system_prompt,
                messages=session.messages,
                relevant_memories=relevant,
            )

            # 流式调用 AI
            try:
                ai_message = None
                provider = self._providers.get_provider()
                raw_stream = provider.chat_stream(context, tool_schemas)

                async for event in process_stream(raw_stream, session_id):
                    yield event

                # 获取完整消息（从 process_stream 的 done 事件）
                # 需要重新获取，因为 process_stream 是 generator
                # 改为直接调用 provider 获取完整消息
                ai_message = await self._call_ai_with_failover(context, tool_schemas)

            except ProviderError as e:
                yield Event.error(f"AI 调用失败: {e}", session_id)
                yield Event.done(session_id)
                return

            await self._session_mgr.add_message(session_id, ai_message)

            if ai_message.is_tool_call():
                tool_messages, tool_events = await self._tool_executor.execute(
                    ai_message.tool_calls, session_id
                )
                for event in tool_events:
                    yield event
                for msg in tool_messages:
                    await self._session_mgr.add_message(session_id, msg)
                continue
            else:
                yield Event.done(session_id)
                return

        yield Event.error(f"处理轮数超过上限 ({self._config.max_iterations})", session_id)
        yield Event.done(session_id)

    async def _call_ai_with_failover(
        self,
        context: list[Message],
        tools: list[dict] | None,
    ) -> Message:
        """调用 AI，支持自动故障转移

        如果当前 Provider 失败，自动切换到下一个。
        所有 Provider 都失败则抛出 ProviderError。
        """
        last_error = None

        while True:
            provider = self._providers.get_provider()
            try:
                return await provider.chat(context, tools)
            except Exception as e:
                last_error = e
                logger.warning("提供商 %s 调用失败: %s", provider.name, e)

                # 尝试故障转移
                next_provider = self._providers.failover()
                if next_provider is None:
                    raise ProviderError(f"所有 AI 提供商都已失败。最后错误: {last_error}") from last_error
                logger.info("故障转移到: %s", next_provider.name)
