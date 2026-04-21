"""
Subagent 编排器 — 整合所有多 Agent 架构组件

核心职责:
  1. 初始化和管理所有组件（Dispatcher, TaskScheduler, WorktreeManager, Critic）
  2. 协调 Master Agent 与 Subagent 的交互
  3. 处理任务图的执行生命周期
  4. 提供统一的接口给上层调用

这是 MASTER_SUBAGENT 模式的核心入口。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Callable

from miniclaw.agents.agent import Agent
from miniclaw.agents.critic.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerError,
)
from miniclaw.agents.critic.critic_injector import CriticInjector, CriticConfig
from miniclaw.agents.critic.validation_gatekeeper import (
    GatekeeperConfig,
    ValidationGatekeeper,
    ValidationRequirement,
)
from miniclaw.agents.critic.validation_tools import RunLinterTool, RunTestsTool
from miniclaw.agents.subagent_executor import SubagentExecutor, create_subagent_executor
from miniclaw.agents.subagent_factory import SubagentConfig, SubagentFactory
from miniclaw.config.settings import Settings, SubagentConfig as SubagentSettings
from miniclaw.dispatcher.dispatcher import Dispatcher, DispatcherError
from miniclaw.dispatcher.event_bus import EventBus, TurnLogStore
from miniclaw.dispatcher.subagent_registry import GuardrailConfig, SubagentRegistry
from miniclaw.dispatcher.task_scheduler import TaskScheduler
from miniclaw.memory.base import MemoryStore
from miniclaw.sessions.manager import SessionManager
from miniclaw.tools.builtin.create_task_graph import get_pending_graph, clear_pending_graph
from miniclaw.tools.builtin.submit_task_result import submit_task_result_tool
from miniclaw.types.enums import AgentRole
from miniclaw.types.events import Event
from miniclaw.types.messages import Message
from miniclaw.types.structured_result import StructuredResult, TaskStatus, validate_result
from miniclaw.types.task_graph import TaskGraphRequest, TaskGraphResult, TaskNode
from miniclaw.types.turn_snapshot import AgentNode
from miniclaw.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class OrchestratorState:
    """编排器状态"""
    is_running: bool = False
    current_graph_id: str | None = None
    active_subagents: int = 0
    total_tasks_completed: int = 0
    total_tasks_failed: int = 0


class SubagentOrchestrator:
    """Subagent 编排器 — 多 Agent 架构的核心协调器

    负责:
      1. 组件初始化和生命周期管理
      2. 任务图解析和调度
      3. Subagent 执行监控
      4. 结果收集和聚合
    """

    def __init__(
        self,
        settings: Settings,
        agent: Agent,
        session_manager: SessionManager,
        memory_store: MemoryStore,
        repo_root: Path,
    ):
        self._settings = settings
        self._agent = agent
        self._session_mgr = session_manager
        self._memory = memory_store
        self._repo_root = repo_root

        self._state = OrchestratorState()

        # 组件（延迟初始化）
        self._event_bus: EventBus | None = None
        self._registry: SubagentRegistry | None = None
        self._dispatcher: Dispatcher | None = None
        self._scheduler: TaskScheduler | None = None
        self._subagent_executor: SubagentExecutor | None = None
        self._factory: SubagentFactory | None = None

        # Critic 组件
        self._circuit_breaker: CircuitBreaker | None = None
        self._critic_injector: CriticInjector | None = None
        self._validation_gatekeeper: ValidationGatekeeper | None = None

    async def initialize(self) -> None:
        """初始化所有组件"""
        subagent_cfg = self._settings.subagent

        # 1. 初始化 EventBus
        turn_store = TurnLogStore(db_path=self._settings.memory.db_path.replace(".db", "_turns.db"))
        self._event_bus = EventBus(turn_store)
        await self._event_bus.initialize()
        logger.info("EventBus 已初始化")

        # 2. 初始化 SubagentRegistry
        guardrail_config = GuardrailConfig(
            max_spawn_depth=subagent_cfg.max_spawn_depth,
            max_children_per_agent=subagent_cfg.max_children_per_agent,
            default_max_steps=subagent_cfg.default_max_steps,
            default_timeout_ms=subagent_cfg.default_timeout_ms,
            max_total_agents=subagent_cfg.max_total_agents,
        )
        self._registry = SubagentRegistry(guardrail_config)
        logger.info("SubagentRegistry 已初始化")

        # 3. 初始化 Dispatcher
        self._dispatcher = Dispatcher(self._registry, self._event_bus)
        await self._dispatcher.initialize()
        logger.info("Dispatcher 已初始化")

        # 4. 初始化 TaskScheduler
        self._scheduler = TaskScheduler(
            self._dispatcher,
            self._registry,
            self._event_bus,
        )
        logger.info("TaskScheduler 已初始化")

        # 5. 初始化 SubagentExecutor
        self._subagent_executor = await create_subagent_executor(
            repo_root=self._repo_root,
            auto_merge=subagent_cfg.auto_merge,
            auto_cleanup=subagent_cfg.auto_cleanup,
        )
        logger.info("SubagentExecutor 已初始化")

        # 6. 初始化 SubagentFactory
        self._factory = SubagentFactory()
        logger.info("SubagentFactory 已初始化")

        # 7. 初始化 Critic 组件
        self._circuit_breaker = CircuitBreaker(CircuitBreakerConfig(
            failure_threshold=3,
            same_error_threshold=subagent_cfg.circuit_breaker_threshold,
        ))

        self._critic_injector = CriticInjector(CriticConfig())

        self._validation_gatekeeper = ValidationGatekeeper(GatekeeperConfig(
            requirement=ValidationRequirement.ANY,
            allow_skip_with_reason=True,
        ))
        logger.info("Critic 组件已初始化")

        self._state.is_running = True
        logger.info("SubagentOrchestrator 初始化完成")

    async def shutdown(self) -> None:
        """关闭所有组件"""
        self._state.is_running = False

        if self._subagent_executor:
            await self._subagent_executor.cleanup_all()

        if self._dispatcher:
            await self._dispatcher.shutdown()

        if self._event_bus:
            await self._event_bus.close()

        logger.info("SubagentOrchestrator 已关闭")

    async def process_message(
        self,
        user_message: Message,
        session_id: str,
    ) -> AsyncIterator[Event]:
        """处理用户消息（Master Agent 入口）

        流程:
          1. Master Agent 分析任务
          2. 如果需要分解，调用 create_task_graph
          3. 调度执行子任务
          4. 收集结果
          5. 返回最终响应
        """
        # 创建 Master Agent 节点
        master_node = AgentNode(
            agent_id="master-001",
            role=AgentRole.MASTER,
            depth=0,
            session_id=session_id,
        )

        # 创建 DAG
        dag = await self._dispatcher.create_dag(
            root_agent_id=master_node.agent_id,
            session_id=session_id,
        )

        self._state.current_graph_id = dag.dag_id

        # 调用 Master Agent 处理
        yield Event.thinking(session_id)

        try:
            # 运行 Master Agent
            accumulated_content = ""
            tool_call_detected = False
            graph_id = None

            async for event in self._agent.process_message(user_message, session_id):
                yield event

                # 检查是否有 create_task_graph 调用
                if event.event_type.value == "tool_call_start":
                    if event.data.get("name") == "create_task_graph":
                        tool_call_detected = True

                if event.event_type.value == "tool_call_result":
                    result_data = event.data.get("result", "")
                    # 尝试解析 graph_id
                    import json
                    try:
                        result_json = json.loads(result_data)
                        graph_id = result_json.get("graph_id")
                    except:
                        pass

                if event.event_type.value == "text_delta":
                    accumulated_content += event.data.get("text", "")

            # 如果检测到任务图，执行调度
            if tool_call_detected and graph_id:
                logger.info("检测到任务图: %s，开始调度执行", graph_id)

                # 获取任务图请求
                graph_request = get_pending_graph(graph_id)
                if graph_request:
                    # 执行任务图
                    graph_result = await self._execute_task_graph(
                        graph_request,
                        master_node,
                        session_id,
                    )

                    # 将结果注入到对话中
                    result_summary = self._format_graph_result(graph_result)

                    # 继续让 Master Agent 处理结果
                    result_message = Message(
                        role="user",
                        content=f"[任务图执行结果]\n{result_summary}",
                    )

                    async for event in self._agent.process_message(result_message, session_id):
                        yield event

                    # 清理
                    clear_pending_graph(graph_id)

        except CircuitBreakerError as e:
            logger.error("熔断触发: %s", e)
            yield Event.error(f"任务执行中断: {str(e)}", session_id)

        except Exception as e:
            logger.error("处理消息失败: %s", e)
            yield Event.error(f"处理失败: {str(e)}", session_id)

        finally:
            yield Event.done(session_id)

    async def _execute_task_graph(
        self,
        request: TaskGraphRequest,
        master_node: AgentNode,
        session_id: str,
    ) -> TaskGraphResult:
        """执行任务图（内部方法，由 process_message 调用）"""
        return await self.execute_task_graph(request, master_node)

    async def execute_task_graph(
        self,
        request: TaskGraphRequest,
        master_node: AgentNode,
        timeout_seconds: int = 300,
    ) -> TaskGraphResult:
        """执行任务图（外部入口，由 MasterAgent._wait_for_graph_result 调用）

        Args:
            request: 任务图请求
            master_node: Master Agent 节点
            timeout_seconds: 总超时时间

        Returns:
            TaskGraphResult: 执行结果
        """
        if self._scheduler is None:
            raise RuntimeError("TaskScheduler 未初始化")

        # 将 Orchestrator 自身注入到 scheduler，使 _execute_single_task 能调用真实 Agent
        self._scheduler.set_orchestrator(self)

        # 执行调度
        try:
            result = await asyncio.wait_for(
                self._scheduler.schedule(request, master_node),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError:
            logger.warning("任务图执行超时: %ds", timeout_seconds)
            # 从 scheduler 获取当前结果
            ctx = self._scheduler._running_graphs.get(request.graph_id if hasattr(request, 'graph_id') else "")
            if ctx:
                ctx.result.status = "timeout"
                return ctx.result
            return TaskGraphResult(
                graph_id="timeout",
                total_tasks=len(request.tasks),
                max_depth=0,
                execution_order=[],
                dynamic_roles=[],
                status="timeout",
            )

        return result

    def _format_graph_result(self, result: TaskGraphResult) -> str:
        """格式化任务图结果"""
        lines = [
            f"**任务图执行完成**",
            f"- 总任务数: {result.total_tasks}",
            f"- 成功: {len(result.completed_tasks)}",
            f"- 失败: {len(result.failed_tasks)}",
            "",
        ]

        if result.completed_tasks:
            lines.append("**已完成任务**:")
            for task_id in result.completed_tasks:
                lines.append(f"  ✓ {task_id}")

        if result.failed_tasks:
            lines.append("**失败任务**:")
            for task_id in result.failed_tasks:
                lines.append(f"  ✗ {task_id}")

        return "\n".join(lines)

    async def spawn_subagent(
        self,
        task: TaskNode,
        session_id: str,
    ) -> tuple[SubagentConfig, Any]:
        """Spawn 一个 Subagent

        Args:
            task: 任务节点
            session_id: 会话 ID

        Returns:
            (SubagentConfig, workspace_info)
        """
        if self._factory is None:
            raise RuntimeError("SubagentFactory 未初始化")

        if self._subagent_executor is None:
            raise RuntimeError("SubagentExecutor 未初始化")

        # 检查护栏 — caller_depth 从 DAG 推断
        # 每个 spawn 的 subagent 深度 = master(0) + 任务依赖层数
        task_depth = 1  # 基础深度：从 master spawn 第一层
        # 如果任务有依赖，深度可能更高（间接依赖链长度）
        # 但护栏检查只关心直接 spawn 深度
        self._registry.guardrails.check_can_spawn(
            caller_depth=task_depth - 1,  # 父节点深度 = 当前深度 - 1
            caller_children_count=self._state.active_subagents,
            caller_id="master",
            total_agent_count=self._state.active_subagents,
        )

        # 创建 Subagent 配置
        config = self._factory.create_config(
            task=task,
            parent_agent_id="master-001",
            session_id=session_id,
        )

        # 准备执行环境
        ctx = await self._subagent_executor.prepare_for_execution(
            task=task,
            config=config,
            agent_id=f"subagent-{task.id}",
            session_id=session_id,
        )

        self._state.active_subagents += 1

        return config, ctx

    async def collect_subagent_result(
        self,
        task_id: str,
        result_data: dict[str, Any],
    ) -> StructuredResult:
        """收集 Subagent 结果"""
        if self._subagent_executor is None:
            raise RuntimeError("SubagentExecutor 未初始化")

        result = await self._subagent_executor.collect_result(task_id, result_data)

        if result.is_success():
            self._state.total_tasks_completed += 1
        else:
            self._state.total_tasks_failed += 1

        return result

    async def finalize_subagent(
        self,
        task_id: str,
    ) -> dict[str, Any]:
        """完成 Subagent 执行"""
        if self._subagent_executor is None:
            raise RuntimeError("SubagentExecutor 未初始化")

        result = await self._subagent_executor.finalize_execution(task_id)
        self._state.active_subagents -= 1

        return result

    def get_circuit_breaker(self) -> CircuitBreaker:
        """获取熔断器"""
        if self._circuit_breaker is None:
            raise RuntimeError("CircuitBreaker 未初始化")
        return self._circuit_breaker

    def get_critic_injector(self) -> CriticInjector:
        """获取 Critic 注入器"""
        if self._critic_injector is None:
            raise RuntimeError("CriticInjector 未初始化")
        return self._critic_injector

    def get_validation_gatekeeper(self) -> ValidationGatekeeper:
        """获取验证门禁器"""
        if self._validation_gatekeeper is None:
            raise RuntimeError("ValidationGatekeeper 未初始化")
        return self._validation_gatekeeper

    async def get_status(self) -> dict[str, Any]:
        """获取编排器状态"""
        return {
            "is_running": self._state.is_running,
            "current_graph_id": self._state.current_graph_id,
            "active_subagents": self._state.active_subagents,
            "total_tasks_completed": self._state.total_tasks_completed,
            "total_tasks_failed": self._state.total_tasks_failed,
            "dispatcher_status": await self._dispatcher.get_status() if self._dispatcher else None,
            "circuit_breaker_status": self._circuit_breaker.get_status() if self._circuit_breaker else None,
        }


async def create_orchestrator(
    settings: Settings,
    agent: Agent,
    session_manager: SessionManager,
    memory_store: MemoryStore,
    repo_root: Path | None = None,
) -> SubagentOrchestrator:
    """创建并初始化编排器"""
    if repo_root is None:
        repo_root = Path.cwd()

    orchestrator = SubagentOrchestrator(
        settings=settings,
        agent=agent,
        session_manager=session_manager,
        memory_store=memory_store,
        repo_root=repo_root,
    )

    await orchestrator.initialize()

    return orchestrator


# 导出
__all__ = [
    "SubagentOrchestrator",
    "OrchestratorState",
    "create_orchestrator",
]