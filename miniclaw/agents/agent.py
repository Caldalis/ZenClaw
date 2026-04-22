"""
核心代理

Agent 是整个系统的核心，实现了 AI 对话 + 工具调用循环:

关键设计:
  - 事件驱动: 不返回字符串，产出 Event 流
  - 工具调用循环: 多轮调用直到 AI 不再请求工具
  - max_iterations: 防止死循环（AI 可能无限循环调用工具）
  - ContextGuard: Token 记账 + 自动压缩 + 工具输出截断
"""

from __future__ import annotations

from typing import AsyncIterator, Any

from miniclaw.agents.providers.registry import ProviderRegistry
from miniclaw.agents.streaming import process_stream
from miniclaw.agents.tool_executor import ToolExecutor
from miniclaw.config.settings import AgentConfig, MemoryConfig
from miniclaw.memory.base import MemoryStore
from miniclaw.memory.context_guard import ContextGuard, COMPACTION_PROMPT_TEMPLATE
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
      ContextGuard   → Token 记账 + 压缩触发 + 工具输出截断
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
        memory_config: MemoryConfig | None = None,
        validation_gatekeeper: Any | None = None,
    ):
        self._config = config
        self._providers = provider_registry
        self._tools = tool_registry
        self._session_mgr = session_manager
        self._memory = memory_store

        mem_cfg = memory_config or MemoryConfig()
        self._tool_executor = ToolExecutor(
            tool_registry,
            tool_result_max_bytes=mem_cfg.tool_result_max_bytes,
            validation_gatekeeper=validation_gatekeeper,
        )
        self._context_window = ContextWindow(config.max_context_tokens)
        self._context_guard = ContextGuard(
            max_context_tokens=config.max_context_tokens,
            compaction_threshold=config.compaction_threshold,
        )
        self._compaction_keep_recent = config.compaction_keep_recent

    async def process_message(self, user_message: Message, session_id: str) -> AsyncIterator[Event]:
        """处理用户消息 — Agent 的核心入口

        这是一个 async generator，产出 Event 事件流。
        上层（Gateway/Channel）通过 async for 消费事件。

        流程:
          1. 保存用户消息
          2. 构建上下文
          3. 调用 AI（可能多轮工具调用循环）
          4. 保存 AI 回复
          5. 检查是否需要上下文压缩
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

            # 检查是否需要上下文压缩
            session = await self._session_mgr.get_session(session_id)
            if session and self._context_guard.should_compact(session.messages):
                logger.info("上下文接近上限 (%d messages)，开始自动压缩...", len(session.messages))
                await self._compact_messages(session_id, session.messages)

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

            # 检查是否需要上下文压缩
            session = await self._session_mgr.get_session(session_id)
            if session and self._context_guard.should_compact(session.messages):
                logger.info("上下文接近上限，开始自动压缩...")
                await self._compact_messages(session_id, session.messages)

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

    async def _compact_messages(self, session_id: str, messages: list[Message]) -> None:
        """将旧消息压缩为摘要，节省上下文 token

        流程:
          1. 分割消息为「待压缩」和「保留」
          2. 构建摘要 prompt，调用 LLM 生成摘要
          3. 将待压缩消息替换为一条摘要消息
          4. 持久化变更（删除旧消息，保存摘要）
        """
        to_compact, to_keep = self._context_guard.select_messages_for_compaction(
            messages,
            keep_recent=self._compaction_keep_recent,
        )
        if not to_compact:
            logger.debug("压缩: 没有可压缩的消息")
            return

        # 构建对话文本用于摘要
        lines = []
        for msg in to_compact:
            role_label = msg.role.value
            content = msg.content or ""
            if msg.tool_calls:
                content += f" [调用工具: {[tc.name for tc in msg.tool_calls]}]"
            if msg.tool_result:
                content += f" [工具结果: {msg.tool_result.content[:200]}...]"
            lines.append(f"[{role_label}]: {content}")
        conversation_text = "\n".join(lines)

        summary_prompt = COMPACTION_PROMPT_TEMPLATE.format(conversation_text=conversation_text)
        summary_request = [
            Message(role=Role.SYSTEM, content="你是一个对话摘要助手，擅长将长对话压缩为简洁摘要。"),
            Message(role=Role.USER, content=summary_prompt),
        ]

        try:
            summary_msg = await self._call_ai_with_failover(summary_request, tools=None)
        except Exception as e:
            logger.warning("压缩摘要生成失败: %s，跳过本次压缩", e)
            return

        summary_message = Message(
            role=Role.SYSTEM,
            content=f"[对话历史摘要]\n{summary_msg.content}",
            session_id=session_id,
        )

        # 更新内存中的会话（session 是缓存对象的引用）
        session = await self._session_mgr.get_session(session_id)
        if session is None:
            return

        ids_to_delete = {m.id for m in to_compact}
        system_msgs = [m for m in to_keep if m.role == Role.SYSTEM]
        non_system_keep = [m for m in to_keep if m.role != Role.SYSTEM]
        session.messages = system_msgs + [summary_message] + non_system_keep

        # 持久化：删除旧消息，保存摘要
        for msg_id in ids_to_delete:
            try:
                await self._memory.delete_message(msg_id)
            except Exception as e:
                logger.warning("删除消息 %s 失败: %s", msg_id[:8], e)
        await self._memory.save_message(session_id, summary_message)

        logger.info(
            "压缩完成: 压缩 %d 条 → 1 条摘要，保留 %d 条",
            len(to_compact),
            len(non_system_keep),
        )

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
