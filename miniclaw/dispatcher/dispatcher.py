"""
Dispatcher - 多 Agent 架构的调度中心

核心职责:
  1. 任务调度与 DAG 执行管理
  2. 超时监控与状态异常检测
  3. 异常捕获与错误移交（返回 isError: true 的精简错误给父 Agent）
  4. 断点恢复支持

Dispatcher 不主动 Kill Subagent，而是自动监控并标记异常状态。

设计原则:
  - Master Agent 不亲自执行脏活，只做调度
  - Dispatcher 是底层基础设施，不直接暴露给上层 Agent
  - 错误信息精简传递，避免上下文污染
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncIterator

from miniclaw.dispatcher.event_bus import EventBus
from miniclaw.dispatcher.subagent_registry import (
    GuardrailConfig,
    GuardrailViolationError,
    SubagentRegistry,
    SubagentSpec,
)
from miniclaw.types.enums import AgentRole, TurnStatus
from miniclaw.types.turn_snapshot import AgentNode, TaskDAG, TurnSnapshot
from miniclaw.utils.logging import get_logger

logger = get_logger(__name__)


class DispatcherError(Exception):
    """Dispatcher 错误 — 用于向上层传递精简错误信息"""

    def __init__(self, message: str, is_timeout: bool = False, details: dict[str, Any] = {}):
        self.is_timeout = is_timeout
        self.details = details
        super().__init__(message)


class TimeoutMonitor:
    """超时监控器 — 自动监控正在执行的 Turn"""

    def __init__(self, event_bus: EventBus, check_interval_ms: int = 1000):
        self._event_bus = event_bus
        self._check_interval = check_interval_ms / 1000  # 转换为秒
        self._running_tasks: dict[str, asyncio.Task] = {}  # turn_id -> Task
        self._timeout_at: dict[str, datetime] = {}  # turn_id -> 超时时间
        self._monitor_task: asyncio.Task | None = None
        self._stop_event: asyncio.Event | None = None

    async def start(self) -> None:
        """启动监控循环"""
        self._stop_event = asyncio.Event()
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("超时监控器已启动 (interval=%.1fs)", self._check_interval)

    async def stop(self) -> None:
        """停止监控"""
        if self._stop_event:
            self._stop_event.set()
        if self._monitor_task:
            try:
                await asyncio.wait_for(self._monitor_task, timeout=5)
            except asyncio.TimeoutError:
                self._monitor_task.cancel()
        logger.info("超时监控器已停止")

    def register_turn(self, turn_id: str, task: asyncio.Task, timeout_ms: int) -> None:
        """注册一个 Turn 到监控"""
        self._running_tasks[turn_id] = task
        timeout_seconds = timeout_ms / 1000
        self._timeout_at[turn_id] = datetime.now(timezone.utc) + timedelta(seconds=timeout_seconds)
        logger.debug("Turn %s 注册到监控，超时时间: %sms", turn_id, timeout_ms)

    def unregister_turn(self, turn_id: str) -> None:
        """从监控移除 Turn"""
        self._running_tasks.pop(turn_id, None)
        self._timeout_at.pop(turn_id, None)
        logger.debug("Turn %s 已从监控移除", turn_id)

    async def _monitor_loop(self) -> None:
        """监控循环 — 检查超时的 Turn"""
        while self._stop_event and not self._stop_event.is_set():
            await asyncio.sleep(self._check_interval)

            now = datetime.now(timezone.utc)
            timed_out = []

            for turn_id, timeout_at in list(self._timeout_at.items()):
                if now >= timeout_at:
                    timed_out.append(turn_id)

            for turn_id in timed_out:
                task = self._running_tasks.get(turn_id)
                if task and not task.done():
                    logger.warning("Turn %s 超时，准备终止", turn_id)
                    # 取消任务（不主动 Kill，而是让任务自然终止）
                    task.cancel()
                    # 记录超时状态
                    interrupted = await self._event_bus.get_all_interrupted()
                    for snap in interrupted:
                        if snap.turn_id == turn_id:
                            await self._event_bus.record_turn_error(
                                snap, "执行超时", is_timeout=True
                            )
                    self.unregister_turn(turn_id)

    @property
    def active_count(self) -> int:
        """当前监控中的 Turn 数量"""
        return len(self._running_tasks)


class Dispatcher:
    """调度中心 — 多 Agent 架构的核心

    负责:
      1. DAG 构建与执行调度
      2. Subagent Spawn 管理
      3. 超时监控与异常捕获
      4. 断点恢复支持

    不负责:
      - 实际执行 AI 对话（由 Agent 执行）
      - 主动 Kill Subagent（由超时监控自动处理）
    """

    def __init__(
        self,
        registry: SubagentRegistry,
        event_bus: EventBus,
    ):
        self._registry = registry
        self._event_bus = event_bus
        self._timeout_monitor = TimeoutMonitor(event_bus)
        self._active_dags: dict[str, TaskDAG] = {}  # dag_id -> DAG
        self._agent_count: int = 0  # 全局 Agent 计数（用于护栏检查）

    async def initialize(self) -> None:
        """初始化 Dispatcher"""
        await self._event_bus.initialize()
        await self._timeout_monitor.start()
        logger.info("Dispatcher 已初始化")

    async def shutdown(self) -> None:
        """关闭 Dispatcher"""
        await self._timeout_monitor.stop()
        await self._event_bus.close()
        logger.info("Dispatcher 已关闭")

    async def create_dag(self, root_agent_id: str, session_id: str) -> TaskDAG:
        """创建新的任务 DAG

        Args:
            root_agent_id: Master Agent 的 ID
            session_id: 会话 ID（上下文隔离）

        Returns:
            新创建的 TaskDAG
        """
        root_node = AgentNode(
            agent_id=root_agent_id,
            role=AgentRole.MASTER,
            depth=0,
            parent_id=None,
            session_id=session_id,
        )

        dag = TaskDAG(
            root_node_id=root_agent_id,
            nodes={root_agent_id: root_node},
            pending_nodes=[root_agent_id],
        )

        self._active_dags[dag.dag_id] = dag
        self._agent_count += 1

        await self._event_bus.save_dag_state(dag)
        logger.info("DAG 已创建: %s, root=%s", dag.dag_id, root_agent_id)

        return dag

    async def spawn_agent(
        self,
        dag: TaskDAG,
        parent_id: str,
        role: AgentRole,
        input_message: str,
        session_id: str | None = None,
    ) -> AgentNode:
        """Spawn 新 Subagent

        执行护栏检查，创建新 AgentNode 并加入 DAG。

        Args:
            dag: 目标 DAG
            parent_id: 父 Agent ID
            role: 新 Agent 的角色
            input_message: 输入消息
            session_id: 可选，指定会话 ID（不指定则继承父节点）

        Returns:
            新创建的 AgentNode

        Raises:
            GuardrailViolationError: 护栏检查失败
        """
        parent_node = dag.nodes.get(parent_id)
        if parent_node is None:
            raise DispatcherError(f"父节点不存在: {parent_id}")

        # 护栏检查
        self._registry.check_can_spawn(
            caller_depth=parent_node.depth,
            caller_children_count=len(parent_node.children_ids),
            caller_id=parent_id,
            total_agent_count=self._agent_count,
        )

        # 获取角色规格
        spec = self._registry.get_spec(role)
        if spec is None:
            spec = self._registry.get_spec(AgentRole.GENERIC)

        # 创建新节点
        new_depth = parent_node.depth + 1
        new_session = session_id or f"{parent_node.session_id}:{role.value}"

        new_node = AgentNode(
            role=role,
            depth=new_depth,
            parent_id=parent_id,
            session_id=new_session,
            max_children=spec.max_steps,  # 使用规格中的最大步数
            max_steps=spec.max_steps,
            timeout_ms=spec.timeout_ms,
        )

        # 更新父节点
        parent_node.children_ids.append(new_node.agent_id)

        # 更新 DAG
        dag.nodes[new_node.agent_id] = new_node
        dag.edges.append((parent_id, new_node.agent_id))
        dag.pending_nodes.append(new_node.agent_id)

        self._agent_count += 1

        await self._event_bus.save_dag_state(dag)
        logger.info(
            "Agent 已 Spawn: %s (role=%s, depth=%d, parent=%s)",
            new_node.agent_id, role.value, new_depth, parent_id
        )

        return new_node

    async def start_turn(
        self,
        node: AgentNode,
        input_message: str,
    ) -> TurnSnapshot:
        """开始一个 Turn

        创建 TurnSnapshot，记录开始状态，注册到超时监控。

        Args:
            node: 执行该 Turn 的 AgentNode
            input_message: 输入消息

        Returns:
            新创建的 TurnSnapshot
        """
        snapshot = TurnSnapshot(
            node=node,
            input_message=input_message,
            status=TurnStatus.PENDING,
        )

        await self._event_bus.record_turn_start(snapshot)
        logger.debug("Turn 已开始: %s (agent=%s)", snapshot.turn_id, node.agent_id)

        return snapshot

    async def execute_turn(
        self,
        snapshot: TurnSnapshot,
        executor,  # Agent 实例或执行函数
    ) -> AsyncIterator[dict]:
        """执行一个 Turn

        包装执行过程，自动处理:
          1. 超时监控
          2. 步数限制
          3. 异常捕获
          4. 结果记录

        Args:
            snapshot: Turn 状态快照
            executor: 执行器（Agent.process_message 或自定义函数）

        Yields:
            执行过程中的事件流
        """
        node = snapshot.node
        spec = self._registry.get_spec(node.role)

        # 创建执行任务
        async def _execute():
            results = []
            step = 0
            try:
                # 调用执行器
                async for event in executor:
                    step += 1
                    # 更新步数
                    await self._event_bus.record_turn_step(snapshot, step)

                    # 步数限制检查
                    if step > node.max_steps:
                        raise DispatcherError(
                            f"步数超限: {step} > {node.max_steps}",
                            is_timeout=False,
                            details={"step": step, "max_steps": node.max_steps},
                        )

                    results.append(event)

                return results

            except asyncio.CancelledError:
                # 被取消（通常是超时）
                await self._event_bus.record_turn_error(snapshot, "执行超时", is_timeout=True)
                raise DispatcherError(
                    "执行超时",
                    is_timeout=True,
                    details={"turn_id": snapshot.turn_id, "timeout_ms": node.timeout_ms},
                )

        # 包装为 asyncio Task
        task = asyncio.create_task(_execute())

        # 注册到超时监控
        self._timeout_monitor.register_turn(snapshot.turn_id, task, node.timeout_ms)

        try:
            # 执行并收集结果
            result = await task

            # 记录完成
            output = self._format_output(result)
            await self._event_bus.record_turn_complete(snapshot, output)

            # 返回结果流
            for event in result:
                yield event

        except DispatcherError as e:
            # Dispatcher 内部错误
            await self._event_bus.record_turn_error(
                snapshot, str(e), is_timeout=e.is_timeout
            )
            yield {"type": "error", "message": str(e), "is_error": True, "is_timeout": e.is_timeout}

        except Exception as e:
            # 其他异常
            error_msg = self._simplify_error(e)
            await self._event_bus.record_turn_error(snapshot, error_msg, is_timeout=False)
            yield {"type": "error", "message": error_msg, "is_error": True}

        finally:
            self._timeout_monitor.unregister_turn(snapshot.turn_id)

    async def complete_agent(self, dag: TaskDAG, agent_id: str) -> None:
        """标记 Agent 完成

        更新 DAG 状态，触发父 Agent 的后续处理。
        """
        if agent_id in dag.running_nodes:
            dag.running_nodes.remove(agent_id)
        if agent_id in dag.pending_nodes:
            dag.pending_nodes.remove(agent_id)
        if agent_id not in dag.completed_nodes:
            dag.completed_nodes.append(agent_id)

        await self._event_bus.save_dag_state(dag)
        logger.info("Agent 已完成: %s (dag=%s)", agent_id, dag.dag_id)

    async def fail_agent(self, dag: TaskDAG, agent_id: str, error: str) -> None:
        """标记 Agent 失败

        更新 DAG 状态，将错误信息传递给父 Agent。
        """
        if agent_id in dag.running_nodes:
            dag.running_nodes.remove(agent_id)
        if agent_id in dag.pending_nodes:
            dag.pending_nodes.remove(agent_id)
        if agent_id not in dag.failed_nodes:
            dag.failed_nodes.append(agent_id)

        await self._event_bus.save_dag_state(dag)

        # 触发父 Agent 的错误处理
        node = dag.nodes.get(agent_id)
        if node and node.parent_id:
            logger.warning(
                "Agent 失败: %s (error=%s), 将错误传递给父 Agent %s",
                agent_id, error[:100], node.parent_id
            )

        logger.info("Agent 已标记失败: %s (dag=%s)", agent_id, dag.dag_id)

    async def recover_dag(self, dag_id: str) -> TaskDAG | None:
        """恢复 DAG 执行

        从最后一个未完成的叶子节点继续执行。
        """
        dag = await self._event_bus.restore_dag(dag_id)
        if dag is None:
            logger.warning("DAG 不存在: %s", dag_id)
            return None

        recovery_point = await self._event_bus.get_recovery_point(dag_id)
        if recovery_point is None:
            logger.info("DAG %s 无需恢复（已完成或无中断节点）", dag_id)
            return dag

        logger.info(
            "恢复 DAG %s 从节点 %s (status=%s)",
            dag_id, recovery_point.node.agent_id, recovery_point.status.value
        )

        self._active_dags[dag_id] = dag
        return dag

    async def get_status(self) -> dict[str, Any]:
        """获取 Dispatcher 状态"""
        return {
            "active_dags": len(self._active_dags),
            "agent_count": self._agent_count,
            "monitored_turns": self._timeout_monitor.active_count,
            "guardrails": {
                "max_spawn_depth": self._registry.guardrails.max_spawn_depth,
                "max_children_per_agent": self._registry.guardrails.max_children_per_agent,
                "max_total_agents": self._registry.guardrails.max_total_agents,
            },
        }

    def _format_output(self, result: list) -> str:
        """格式化输出结果"""
        # 提取文本内容
        texts = []
        for item in result:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    texts.append(item.get("content", ""))
        return "\n".join(texts) if texts else str(result)

    def _simplify_error(self, error: Exception) -> str:
        """精简错误信息

        避免: 将完整的堆栈追踪传递给上层 Agent（污染上下文）
        策略: 提取关键信息，格式化为结构化错误消息
        """
        error_type = type(error).__name__
        error_msg = str(error)

        # 限制长度
        if len(error_msg) > 200:
            error_msg = error_msg[:200] + "..."

        return f"[{error_type}] {error_msg}"

    @property
    def registry(self) -> SubagentRegistry:
        """获取 Subagent 注册表"""
        return self._registry

    @property
    def event_bus(self) -> EventBus:
        """获取事件总线"""
        return self._event_bus