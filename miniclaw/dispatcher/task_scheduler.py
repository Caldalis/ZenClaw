"""
任务调度器 — 解析任务图并调度执行

核心职责:
  1. 解析 create_task_graph 工具的输出
  2. 按 depends_on 顺序执行任务（spawn_sequential）
  3. 并行执行无依赖的任务（spawn_parallel）
  4. 管理 Subagent 的生命周期
  5. 收集子任务结果并通知 Master Agent

调度策略:
  - 同层级的任务可并行执行
  - 有依赖的任务必须等待依赖完成
  - 支持最大并发数限制
  - 支持 fail_fast 模式

状态管理:
  - 通过 EventBus 记录任务执行状态
  - Master Agent 进入 Async Await 状态等待子任务结果
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Callable

from miniclaw.dispatcher.dispatcher import Dispatcher, DispatcherError
from miniclaw.dispatcher.event_bus import EventBus
from miniclaw.dispatcher.subagent_registry import SubagentRegistry, SubagentSpec
from miniclaw.types.enums import AgentRole, TurnStatus
from miniclaw.types.task_graph import (
    TaskGraphRequest,
    TaskGraphResult,
    TaskNode,
    analyze_execution_order,
    build_task_graph_result,
)
from miniclaw.types.turn_snapshot import AgentNode, TurnSnapshot
from miniclaw.utils.logging import get_logger

logger = get_logger(__name__)


class TaskScheduler:
    """任务调度器 — 管理 DAG 任务图的执行

    调度流程:
      1. 接收任务图请求
      2. 分析执行顺序（拓扑排序）
      3. 按层级调度任务
      4. 监控执行状态
      5. 收集结果并返回
    """

    def __init__(
        self,
        dispatcher: Dispatcher,
        registry: SubagentRegistry,
        event_bus: EventBus,
    ):
        self._dispatcher = dispatcher
        self._registry = registry
        self._event_bus = event_bus

        # 执行状态
        self._running_graphs: dict[str, GraphExecutionContext] = {}

        # 子任务结果回调
        self._result_callbacks: dict[str, Callable] = {}

    async def schedule(
        self,
        request: TaskGraphRequest,
        master_node: AgentNode,
    ) -> TaskGraphResult:
        """调度执行任务图

        Args:
            request: 任务图请求（从 create_task_graph 工具解析）
            master_node: Master Agent 的节点信息

        Returns:
            TaskGraphResult: 任务图执行结果
        """
        # 构建结果对象
        result = build_task_graph_result(request)

        # 创建执行上下文
        ctx = GraphExecutionContext(
            graph_id=result.graph_id,
            request=request,
            result=result,
            master_node=master_node,
        )

        self._running_graphs[result.graph_id] = ctx

        logger.info(
            "开始调度任务图 %s: %d 任务, %d 层级",
            result.graph_id, result.total_tasks, len(result.execution_order)
        )

        # 注册动态角色
        await self._register_dynamic_roles(request, result)

        # 按层级执行
        try:
            await self._execute_levels(ctx)
            result.status = "completed"
        except Exception as e:
            result.status = "failed"
            logger.error("任务图 %s 执行失败: %s", result.graph_id, e)

        # 清理执行上下文
        self._running_graphs.pop(result.graph_id, None)

        return result

    async def _register_dynamic_roles(
        self,
        request: TaskGraphRequest,
        result: TaskGraphResult,
    ) -> None:
        """注册动态角色（从任务的 custom_role_prompt）"""
        for role in result.dynamic_roles:
            spec = SubagentSpec(
                role=AgentRole.GENERIC,  # 动态角色使用 GENERIC 作为基础
                name=role.name,
                description=f"动态角色: {role.name}",
                system_prompt=role.custom_prompt,
                max_steps=role.max_steps,
                timeout_ms=role.timeout_ms,
                allowed_tools=role.allowed_tools,
                forbidden_tools=role.forbidden_tools,
                requires_worktree=role.requires_worktree,
            )
            self._registry.register(spec)
            logger.info("动态角色已注册: %s", role.name)

    async def _execute_levels(self, ctx: GraphExecutionContext) -> None:
        """按层级执行任务

        同层级的任务可并行执行，层级间必须顺序执行。
        """
        task_map = {t.id: t for t in ctx.request.tasks}

        for level_idx, level_tasks in enumerate(ctx.result.execution_order):
            logger.info(
                "执行层级 %d: %s (并行=%d)",
                level_idx, level_tasks, ctx.request.max_concurrent
            )

            # 获取当前层级可并行执行的任务数
            parallel_count = min(len(level_tasks), ctx.request.max_concurrent)

            # 并行执行当前层级任务
            tasks_to_run = level_tasks[:parallel_count]
            remaining = level_tasks[parallel_count:]

            # 执行第一批
            await self._execute_parallel(ctx, tasks_to_run, task_map)

            # 如果有剩余，继续执行（受并发限制）
            while remaining:
                batch_size = min(len(remaining), ctx.request.max_concurrent)
                batch = remaining[:batch_size]
                remaining = remaining[batch_size:]
                await self._execute_parallel(ctx, batch, task_map)

    async def _execute_parallel(
        self,
        ctx: GraphExecutionContext,
        task_ids: list[str],
        task_map: dict[str, TaskNode],
    ) -> None:
        """并行执行一组任务"""
        async_tasks = []

        for task_id in task_ids:
            task = task_map.get(task_id)
            if task is None:
                continue

            async_tasks.append(self._execute_single_task(ctx, task))

        # 并行执行
        results = await asyncio.gather(*async_tasks, return_exceptions=True)

        # 处理结果
        for task_id, result in zip(task_ids, results):
            if isinstance(result, Exception):
                ctx.result.failed_tasks.append(task_id)
                ctx.failed_tasks.add(task_id)
                if ctx.request.fail_fast:
                    raise DispatcherError(
                        f"任务 {task_id} 失败: {result}",
                        is_timeout=False,
                    )
            else:
                ctx.result.completed_tasks.append(task_id)
                ctx.completed_tasks.add(task_id)

    async def _execute_single_task(
        self,
        ctx: GraphExecutionContext,
        task: TaskNode,
    ) -> str:
        """执行单个任务

        流程:
          1. 创建 Subagent Node
          2. 创建 Turn Snapshot
          3. 调用 Dispatcher 执行
          4. 收集结果
        """
        # 解析角色
        role = self._parse_role(task.role)

        # Spawn Subagent Node
        subagent_node = await self._dispatcher.spawn_agent(
            dag=self._get_or_create_dag(ctx),
            parent_id=ctx.master_node.agent_id,
            role=role,
            input_message=task.instruction,
            session_id=f"{ctx.master_node.session_id}:{task.id}",
        )

        # 创建 Turn
        snapshot = await self._dispatcher.start_turn(
            node=subagent_node,
            input_message=task.instruction,
        )

        logger.info(
            "任务 %s 已启动: agent=%s, role=%s",
            task.id, subagent_node.agent_id, task.role
        )

        # 这里需要实际的 Agent 执行器
        # 目前返回模拟结果，后续集成真正的 Agent.process_message TODO
        result = await self._simulate_execution(snapshot, task)

        # 记录结果到任务节点
        task.result = result
        task.is_error = result.startswith("错误:")

        return result

    async def _simulate_execution(
        self,
        snapshot: TurnSnapshot,
        task: TaskNode,
    ) -> str:
        """模拟执行（用于测试）

        实际实现应调用 Agent.process_message
        """
        # 模拟执行延迟 T
        await asyncio.sleep(0.1)

        # 模拟结果
        if "错误" in task.instruction or "失败" in task.instruction:
            return f"错误: 任务 '{task.id}' 执行失败（模拟）"
        else:
            return f"任务 '{task.id}' 完成。结果: {task.instruction[:50]}..."

    def _parse_role(self, role_name: str) -> AgentRole:
        """解析角色名称为 AgentRole 枚举"""
        # 尝试匹配预定义角色
        role_mapping = {
            "CoderAgent": AgentRole.CODER,
            "Coder": AgentRole.CODER,
            "SearcherAgent": AgentRole.SEARCHER,
            "Searcher": AgentRole.SEARCHER,
            "ReviewerAgent": AgentRole.REVIEWER,
            "Reviewer": AgentRole.REVIEWER,
            "TesterAgent": AgentRole.TESTER,
            "Tester": AgentRole.TESTER,
            "PlannerAgent": AgentRole.PLANNER,
            "Planner": AgentRole.PLANNER,
            "MasterAgent": AgentRole.MASTER,
            "Master": AgentRole.MASTER,
        }

        return role_mapping.get(role_name, AgentRole.GENERIC)

    def _get_or_create_dag(self, ctx: GraphExecutionContext):
        """获取或创建 DAG 对象

        这里返回一个简化版本的 DAG 结构
        """
        from miniclaw.types.turn_snapshot import TaskDAG

        # 简化实现：直接使用上下文中的信息
        return TaskDAG(
            dag_id=ctx.graph_id,
            root_node_id=ctx.master_node.agent_id,
            nodes={ctx.master_node.agent_id: ctx.master_node},
        )

    async def wait_for_results(
        self,
        graph_id: str,
        timeout_ms: int = 600000,
    ) -> TaskGraphResult | None:
        """等待任务图执行完成

        Master Agent 进入 Async Await 状态，等待子任务结果。
        """
        ctx = self._running_graphs.get(graph_id)
        if ctx is None:
            return None

        # 等待所有任务完成
        timeout_seconds = timeout_ms / 1000
        try:
            await asyncio.wait_for(
                self._wait_for_completion(ctx),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError:
            ctx.result.status = "timeout"
            logger.warning("任务图 %s 执行超时", graph_id)

        return ctx.result

    async def _wait_for_completion(self, ctx: GraphExecutionContext) -> None:
        """等待任务图完成"""
        while len(ctx.completed_tasks) + len(ctx.failed_tasks) < ctx.request.total_tasks:
            await asyncio.sleep(0.5)

    def register_result_callback(
        self,
        graph_id: str,
        callback: Callable[[TaskGraphResult], None],
    ) -> None:
        """注册结果回调

        当任务图完成时，回调会被触发。
        """
        self._result_callbacks[graph_id] = callback

    async def get_status(self, graph_id: str) -> dict[str, Any] | None:
        """获取任务图执行状态"""
        ctx = self._running_graphs.get(graph_id)
        if ctx is None:
            return None

        return {
            "graph_id": graph_id,
            "status": ctx.result.status,
            "total_tasks": ctx.request.total_tasks,
            "completed": len(ctx.completed_tasks),
            "failed": len(ctx.failed_tasks),
            "running": ctx.request.total_tasks - len(ctx.completed_tasks) - len(ctx.failed_tasks),
        }


class GraphExecutionContext:
    """任务图执行上下文

    管理单个任务图的执行状态。
    """

    def __init__(
        self,
        graph_id: str,
        request: TaskGraphRequest,
        result: TaskGraphResult,
        master_node: AgentNode,
    ):
        self.graph_id = graph_id
        self.request = request
        self.result = result
        self.master_node = master_node

        # 任务执行状态
        self.completed_tasks: set[str] = set()
        self.failed_tasks: set[str] = set()
        self.running_tasks: set[str] = set()

        # Subagent 映射
        self.task_to_agent: dict[str, str] = {}  # task_id -> agent_id

        # 执行结果
        self.task_results: dict[str, str] = {}  # task_id -> result


# 导出
__all__ = ["TaskScheduler", "GraphExecutionContext"]