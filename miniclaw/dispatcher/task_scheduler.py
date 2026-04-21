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

        # Orchestrator 引用（用于真实 Agent 执行）
        self._orchestrator: Any = None

    def set_orchestrator(self, orchestrator: Any) -> None:
        """设置 Orchestrator 引用，用于 _execute_single_task 调用真实 Agent

        Args:
            orchestrator: SubagentOrchestrator 实例
        """
        self._orchestrator = orchestrator
        logger.info("TaskScheduler 已绑定 Orchestrator")

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

        # 获取 DAG 用于状态更新
        dag = self._get_or_create_dag(ctx)

        # 处理结果
        for task_id, result in zip(task_ids, results):
            agent_id = ctx.task_to_agent.get(task_id)

            if isinstance(result, Exception):
                ctx.result.failed_tasks.append(task_id)
                ctx.failed_tasks.add(task_id)

                # 更新 Dispatcher DAG 状态
                if agent_id and dag:
                    await self._dispatcher.fail_agent(dag, agent_id, str(result))

                if ctx.request.fail_fast:
                    raise DispatcherError(
                        f"任务 {task_id} 失败: {result}",
                        is_timeout=False,
                    )
            else:
                ctx.result.completed_tasks.append(task_id)
                ctx.completed_tasks.add(task_id)

                # 更新 Dispatcher DAG 状态
                if agent_id and dag:
                    await self._dispatcher.complete_agent(dag, agent_id)

    async def _execute_single_task(
        self,
        ctx: GraphExecutionContext,
        task: TaskNode,
    ) -> str:
        """执行单个任务

        流程:
          1. 创建 Subagent 配置
          2. 通过 Orchestrator 准备执行环境
          3. 使用真实 Agent 执行任务（ReAct 循环）
          4. 收集结构化结果
          5. 失败时触发 Critic 系统
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

        # 使用 Orchestrator 执行真实 Agent（如果可用）
        if self._orchestrator is not None:
            result = await self._execute_with_agent(ctx, task, snapshot)
        else:
            # 回退到模拟执行（用于测试场景）
            result = await self._simulate_execution(snapshot, task)

        # 记录结果到任务节点
        task.result = result
        task.is_error = result.startswith("错误:") or result.startswith("失败:")

        # 保存到上下文
        ctx.task_results[task.id] = result
        ctx.task_to_agent[task.id] = subagent_node.agent_id

        return result

    async def _execute_with_agent(
        self,
        ctx: GraphExecutionContext,
        task: TaskNode,
        snapshot: TurnSnapshot,
    ) -> str:
        """通过 Orchestrator 执行真实 Agent

        使用 SubagentExecutor 运行 Agent 的 ReAct 循环：
          1. 准备执行环境（Worktree + 隔离工具）
          2. 构建隔离上下文
          3. 调用 Agent.process_message 执行
          4. 收集结果并验证
          5. 失败时检查 Circuit Breaker
        """
        from miniclaw.agents.context_isolator import ContextIsolator
        from miniclaw.types.structured_result import validate_result
        from miniclaw.types.messages import Message
        from miniclaw.types.enums import Role as MsgRole

        orchestrator = self._orchestrator

        # 1. 创建 Subagent 配置
        subagent_config = orchestrator._factory.create_config(
            task=task,
            parent_agent_id=ctx.master_node.agent_id,
            session_id=snapshot.node.session_id,
        )

        # 2. 准备执行环境
        exec_ctx = await orchestrator._subagent_executor.prepare_for_execution(
            task=task,
            config=subagent_config,
            agent_id=snapshot.node.agent_id,
            session_id=snapshot.node.session_id,
        )

        # 3. 构建隔离上下文
        isolator = ContextIsolator()

        # 收集依赖任务的执行结果
        dep_results = {}
        for dep_id in task.depends_on:
            if dep_id in ctx.task_results:
                dep_results[dep_id] = ctx.task_results[dep_id]

        isolated_context = isolator.isolate(
            instruction=task.instruction,
            task_dependencies=dep_results,
        )

        # 合并系统提示词
        isolated_context.system_prompt = subagent_config.system_prompt

        # 4. 构建消息并执行
        messages = isolated_context.to_messages()
        user_message = Message(
            role=MsgRole.USER,
            content=isolated_context.instruction,
        )

        # 获取工作目录（用于隔离工具）
        worktree_path = await orchestrator._subagent_executor.get_workspace_path(task.id)

        try:
            # 使用基础 Agent 执行任务
            # 创建一个配置了角色特定提示词的临时 Agent
            from miniclaw.agents.agent import Agent
            from miniclaw.config.settings import AgentConfig

            task_agent_config = AgentConfig(
                system_prompt=subagent_config.system_prompt,
                max_iterations=subagent_config.max_steps,
                max_context_tokens=4000,  # Subagent 使用更小的上下文
                compaction_threshold=0.85,
            )

            task_agent = Agent(
                config=task_agent_config,
                memory_config=orchestrator._settings.memory,
                provider_registry=orchestrator._agent._provider_registry,
                tool_registry=orchestrator._agent._tool_registry,
                session_manager=orchestrator._session_mgr,
                memory_store=orchestrator._memory,
            )

            # 执行 ReAct 循环
            accumulated_result = ""
            async for event in task_agent.process_message(
                user_message, snapshot.node.session_id
            ):
                if event.event_type.value == "text_delta":
                    accumulated_result += event.data.get("text", "")
                elif event.event_type.value == "text_done":
                    pass  # 文本完成
                elif event.event_type.value == "tool_call_result":
                    # 检查是否是 submit_task_result 调用
                    tool_name = event.data.get("name", "")
                    if tool_name == "submit_task_result":
                        result_data = event.data.get("result", "")
                        # 尝试解析结构化结果
                        import json
                        try:
                            parsed = json.loads(result_data)
                            result = validate_result(parsed)
                            # 收集结果
                            await orchestrator._subagent_executor.collect_result(
                                task.id, parsed
                            )
                            return result.to_master_context()
                        except json.JSONDecodeError:
                            accumulated_result += result_data

                elif event.event_type.value == "error":
                    error_msg = event.data.get("message", "未知错误")

                    # 记录到 Circuit Breaker
                    if orchestrator._circuit_breaker:
                        pattern = orchestrator._circuit_breaker.record_failure(
                            Exception(error_msg)
                        )
                        # 注入 Critic 警示
                        if orchestrator._critic_injector:
                            orchestrator._critic_injector.record_failure(
                                tool_name="agent_execution",
                                tool_arguments={"task_id": task.id},
                                error=error_msg,
                                error_pattern=pattern,
                            )

                    return f"错误: {error_msg}"

            # 如果 Agent 没有调用 submit_task_result，用累积结果构建默认结果
            if accumulated_result:
                default_result = validate_result({
                    "status": "partial_success",
                    "summary": accumulated_result[:200],
                    "files_changed": [],
                    "unresolved_issues": "Agent 未调用 submit_task_result 提交结构化结果",
                })
                return default_result.to_master_context()
            else:
                return f"任务 '{task.id}' 执行完成，但未产生输出"

        except Exception as e:
            logger.error("Agent 执行失败: task=%s, error=%s", task.id, e)

            # 记录到 Circuit Breaker
            if orchestrator._circuit_breaker:
                orchestrator._circuit_breaker.record_failure(e)

            return f"错误: {str(e)}"
        finally:
            # 清理执行环境
            try:
                await orchestrator._subagent_executor.finalize_execution(task.id)
            except Exception as e:
                logger.warning("清理执行环境失败: %s", e)

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

        在执行期间维护一个持久的 DAG，而不是每次调用都重新创建。
        """
        from miniclaw.types.turn_snapshot import TaskDAG

        # 如果已经存在 DAG（Dispatcher 创建的），使用它
        existing_dag = self._dispatcher._active_dags.get(ctx.graph_id)
        if existing_dag:
            return existing_dag

        # 否则创建一个新的
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