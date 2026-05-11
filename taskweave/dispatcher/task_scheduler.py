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

from taskweave.dispatcher.dispatcher import Dispatcher, DispatcherError
from taskweave.dispatcher.event_bus import EventBus
from taskweave.dispatcher.subagent_registry import SubagentRegistry, SubagentSpec
from taskweave.types.enums import AgentRole, TurnStatus
from taskweave.types.task_graph import (
    FailureCategory,
    TaskGraphRequest,
    TaskGraphResult,
    TaskNode,
    analyze_execution_order,
    build_task_graph_result,
)
from taskweave.types.task_outcome import (
    AgentReport,
    SystemObservation,
    TaskOutcome,
    WorkspaceFinalState,
    build_outcome_for_no_worktree_task,
    build_outcome_for_workspace_failure,
)
from taskweave.types.turn_snapshot import AgentNode, TurnSnapshot
from taskweave.utils.logging import get_logger
from taskweave.worktree.lifecycle import (
    SandboxViolation,
    WorkspaceCorrupted,
    WorktreeCommitFailed,
    WorktreeCreateFailed,
    WorktreeError,
    WorktreeMergeConflict,
    WorktreeNotFound,
)

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
        graph_id: str | None = None,
    ) -> TaskGraphResult:
        """调度执行任务图

        Args:
            request: 任务图请求（从 create_task_graph 工具解析）
            master_node: Master Agent 的节点信息
            graph_id: 已有的任务图 ID（复用 create_task_graph 工具生成的 ID）

        Returns:
            TaskGraphResult: 任务图执行结果
        """
        # 构建结果对象 — 复用已有的 graph_id 而非重新生成
        result = build_task_graph_result(request)
        if graph_id:
            result.graph_id = graph_id

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

            # 系统观察重判：把"agent 报成功但系统判失败"的任务从 completed_tasks
            # 移到 failed_tasks。这是断掉 Master 撒谎的最后一道闸门 ——
            # 即便 agent 调了 submit_task_result(status=success)，只要 outcome
            # 的 system_observation 显示产物没落地，task 就算失败。
            self._reconcile_outcomes_into_result(ctx)

            # 状态分级，反映真实结果（基于系统观察后的最终 failed/completed）：
            #   - 全失败  → failed
            #   - 有失败  → partial
            #   - 全成功  → completed
            failed = len(result.failed_tasks)
            done = len(result.completed_tasks)
            if failed > 0 and done == 0:
                result.status = "failed"
            elif failed > 0:
                result.status = "partial"
            else:
                result.status = "completed"
        except Exception as e:
            result.status = "failed"
            logger.error("任务图 %s 执行失败: %s", result.graph_id, e)

        # DAG 执行完毕后归档 subagent session（避免 sessions 表长期堆积）
        if ctx.subagent_session_ids and self._orchestrator is not None:
            try:
                await self._orchestrator.session_manager.archive_subagent_sessions(
                    ctx.subagent_session_ids,
                )
            except Exception as archive_err:
                logger.warning("归档 subagent session 失败: %s", archive_err)

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
            # requires_validation 仅在 master 显式声明时透传到 SubagentSpec；
            # 未声明（None）保留 SubagentSpec 默认值 True，让 SubagentRegistry
            # 的 guardrail 视图保持"严格"语义（实际执行时仍由 factory 用同一份
            # custom_role_config 推断，两个判断口径不会冲突）。
            spec_kwargs: dict[str, Any] = {
                "role": AgentRole.GENERIC,
                "name": role.name,
                "description": f"动态角色: {role.name}",
                "system_prompt": role.custom_prompt,
                "max_steps": role.max_steps,
                "timeout_ms": role.timeout_ms,
                "allowed_tools": role.allowed_tools,
                "forbidden_tools": role.forbidden_tools,
                "requires_worktree": role.requires_worktree,
            }
            if role.requires_validation is not None:
                spec_kwargs["requires_validation"] = role.requires_validation

            spec = SubagentSpec(**spec_kwargs)
            self._registry.register(spec)
            logger.info(
                "动态角色已注册: %s (requires_worktree=%s, requires_validation=%s)",
                role.name, role.requires_worktree,
                role.requires_validation if role.requires_validation is not None else "auto",
            )

    async def _execute_levels(self, ctx: GraphExecutionContext) -> None:
        """按层级执行任务

        同层级的任务可并行执行，层级间必须顺序执行。
        使用 Semaphore 控制并发上限，使得同一层级内的任务可以在
        并发槽位释放后立即开始，而非等待整批完成后再启动下一批。
        DAG 完成后统一合并和清理所有 worktree。
        """
        task_map = {t.id: t for t in ctx.request.tasks}
        semaphore = asyncio.Semaphore(ctx.request.max_concurrent)

        for level_idx, level_tasks in enumerate(ctx.result.execution_order):
            logger.info(
                "执行层级 %d: %s (并发上限=%d)",
                level_idx, level_tasks, ctx.request.max_concurrent
            )

            await self._execute_level_with_semaphore(
                ctx, level_tasks, task_map, semaphore,
            )

        # DAG 完成后统一合并和清理所有延迟的 worktree
        await self._finalize_dag_worktrees(ctx)

    async def _execute_level_with_semaphore(
        self,
        ctx: GraphExecutionContext,
        level_tasks: list[str],
        task_map: dict[str, TaskNode],
        semaphore: asyncio.Semaphore,
    ) -> None:
        """使用 Semaphore 控制并发执行同一层级内的任务

        每个 task 独立获取 semaphore 槽位，完成后立即释放，
        使得后续 task 无需等待整批完成即可开始。
        """
        async def _run_task_with_semaphore(task_id: str) -> tuple[str, Any]:
            async with semaphore:
                task = task_map.get(task_id)
                if task is None:
                    return task_id, None

                # 短路：任一上游依赖失败 → 当前任务直接失败，不再启动 subagent。
                # 之前的 bug: 单依赖路径不检查 failed_tasks，导致下游在错乱的
                # 默认分支上重新创建 worktree，重做一遍上游的工作。
                failed_deps = [
                    dep for dep in task.depends_on if dep in ctx.failed_tasks
                ]
                if failed_deps:
                    msg = f"上游依赖失败，跳过执行: {', '.join(failed_deps)}"
                    logger.warning("任务 %s 短路: %s", task_id, msg)
                    return task_id, f"错误: {msg}"

                logger.info("任务 %s 获取并发槽位，开始执行", task_id)
                result = await self._execute_single_task(ctx, task)
                return task_id, result

        # 并行启动所有任务（semaphore 内部控制并发上限）
        coros = [_run_task_with_semaphore(tid) for tid in level_tasks]
        results = await asyncio.gather(*coros, return_exceptions=True)

        # 获取 DAG 用于状态更新
        dag = self._get_or_create_dag(ctx)

        # 处理结果
        for task_id, result in zip(level_tasks, results):
            agent_id = ctx.task_to_agent.get(task_id)

            if isinstance(result, Exception):
                error_msg = str(result)
                logger.error(
                    "任务 %s 失败: %s",
                    task_id, error_msg,
                    exc_info=result,
                )
                ctx.result.failed_tasks.append(task_id)
                ctx.result.task_errors[task_id] = error_msg
                # DispatcherError 自带 is_timeout 标记，优先用它做精准分类；
                # 其他异常退回 FailureCategory.from_error 启发式分类。
                if isinstance(result, DispatcherError) and getattr(result, "is_timeout", False):
                    ctx.result.task_failure_categories[task_id] = FailureCategory.TIMEOUT
                else:
                    ctx.result.task_failure_categories[task_id] = (
                        FailureCategory.from_error(error_msg)
                    )
                ctx.failed_tasks.add(task_id)

                # 更新 Dispatcher DAG 状态
                if agent_id and dag:
                    await self._dispatcher.fail_agent(dag, agent_id, error_msg)

                if ctx.request.fail_fast:
                    raise DispatcherError(
                        f"任务 {task_id} 失败: {result}",
                        is_timeout=False,
                    )
            elif isinstance(result, tuple) and len(result) == 2:
                # 正常返回 (task_id, actual_result)
                actual_result = result[1]
                if actual_result is None:
                    # task_map 中没有该任务（不应发生）
                    logger.warning("任务 %s 未在 task_map 中找到，跳过", task_id)
                    continue

                if isinstance(actual_result, str) and (actual_result.startswith("错误:") or actual_result.startswith("失败:")):
                    # 兜底: 某些路径仍可能返回错误字符串
                    error_msg = actual_result
                    ctx.result.failed_tasks.append(task_id)
                    ctx.result.task_errors[task_id] = error_msg
                    # 短路失败（"上游依赖失败"）单独标 DEPENDENCY_FAILED，
                    # 否则启发式从错误文本推断分类
                    if "上游依赖失败" in error_msg:
                        ctx.result.task_failure_categories[task_id] = (
                            FailureCategory.DEPENDENCY_FAILED
                        )
                    else:
                        ctx.result.task_failure_categories[task_id] = (
                            FailureCategory.from_error(error_msg)
                        )
                    ctx.failed_tasks.add(task_id)

                    if agent_id and dag:
                        await self._dispatcher.fail_agent(dag, agent_id, error_msg)

                    if ctx.request.fail_fast:
                        raise DispatcherError(
                            f"任务 {task_id} 失败: {error_msg}",
                            is_timeout=False,
                        )
                else:
                    ctx.result.completed_tasks.append(task_id)
                    ctx.completed_tasks.add(task_id)

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

        # Subagent session 必须包含 graph_id：每一轮 master 创建的 DAG 都是
        # 独立尝试，不能让上一轮 task 的失败消息历史污染本轮 ReAct 循环。
        # 之前是 "{master_session}:{task_id}"，跨轮同 task_id 会拉回 14 条
        # 失败消息，agent 一上来就重演死循环（log: "恢复会话: 14 条消息"）。
        sub_session_id = f"{ctx.master_node.session_id}:{ctx.graph_id}:{task.id}"

        # Spawn Subagent Node
        subagent_node = await self._dispatcher.spawn_agent(
            dag=self._get_or_create_dag(ctx),
            parent_id=ctx.master_node.agent_id,
            role=role,
            input_message=task.instruction,
            session_id=sub_session_id,
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
            try:
                result = await self._execute_with_agent(ctx, task, snapshot)
            except DispatcherError as e:
                # 任务失败 — 清理执行环境
                # 记录失败结果到上下文（供下游依赖任务通过文本通道感知）
                error_msg = str(e)
                ctx.task_results[task.id] = f"错误: {error_msg}"
                ctx.task_structured_results[task.id] = {
                    "status": "failed",
                    "summary": error_msg[:200],
                    "files_changed": [],
                    "unresolved_issues": "任务执行失败",
                }

                # 失败的任务无论是否是依赖任务，都需要清理 worktree：
                #   - 依赖任务失败 → 下游无法继承其分支，需强制清理防止残留
                #   - 非依赖任务失败 → 无下游，直接清理
                # 但如果有下游依赖的任务已经部分提交了变更，
                # 仍需合并（保留可用产出），仅当完全无提交时才强制清理
                try:
                    # 失败任务的 worktree 不保留（无论是否有下游依赖）：
                    # 下游会从 dep_structured_results 文本通道感知失败，
                    # 不会尝试继承失败任务的分支
                    await self._orchestrator.subagent_executor.finalize_execution(
                        task.id, merge_and_cleanup=False,
                    )
                except Exception as cleanup_err:
                    logger.warning("清理执行环境失败: %s", cleanup_err)

                # 确保失败任务的 worktree 被强制清理
                try:
                    executor = self._orchestrator.subagent_executor
                    if executor.has_context(task.id):
                        await executor.force_cleanup_workspace(task.id)
                except Exception as cleanup_err2:
                    logger.warning("强制清理 worktree 失败: %s", cleanup_err2)

                raise
        else:
            # 回退模拟
            result = await self._simulate_execution(snapshot, task)

        # 记录结果到任务节点（仅在成功路径到达这里时）
        task.result = result
        task.is_error = result.startswith("错误:") or result.startswith("失败:")

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
          1. 检查 CircuitBreaker 熔断状态
          2. 准备执行环境（Worktree + 隔离工具）
          3. 构建隔离上下文 + Critic 警示注入
          4. 构建 per-task ToolRegistry（隔离工具 + 验证工具 + 门禁）
          5. 调用 Agent.process_message 执行
          6. 收集结果并验证
          7. 成功时 record_success，失败时 record_failure + Critic 注入
        """
        from taskweave.agents.context_isolator import ContextIsolator
        from taskweave.agents.critic.validation_gatekeeper import ValidationAwareSubmitTool
        from taskweave.agents.critic.validation_tools import RunLinterTool, RunTestsTool
        from taskweave.tools.registry import ToolRegistry
        from taskweave.types.structured_result import validate_result
        from taskweave.types.messages import Message
        from taskweave.types.enums import Role as MsgRole

        orchestrator = self._orchestrator
        executor = orchestrator.subagent_executor

        # 0. 检查 CircuitBreaker — 如果已熔断，直接中断
        if orchestrator.circuit_breaker and orchestrator.circuit_breaker.is_open:
            logger.warning("CircuitBreaker 已熔断，中断任务: %s", task.id)
            raise orchestrator.circuit_breaker.get_breaker_error()

        # 1. 创建 Subagent 配置
        subagent_config = orchestrator.factory.create_config(
            task=task,
            parent_agent_id=ctx.master_node.agent_id,
            session_id=snapshot.node.session_id,
        )

        # 1b. 查找依赖任务的 worktree 分支，使依赖产物可见
        # 多依赖场景：将所有依赖分支合并到中间分支，基于中间分支创建 worktree
        parent_branch = None
        intermediate_branch_name = None

        if len(task.depends_on) == 0:
            parent_branch = None
        elif len(task.depends_on) == 1:
            dep_id = task.depends_on[0]
            # 防御层：理论上 _run_task_with_semaphore 已短路掉这种情况，
            # 这里再检查一次防止其他调用路径绕过短路。
            if dep_id in ctx.failed_tasks:
                raise DispatcherError(
                    f"上游依赖 {dep_id} 已失败，拒绝在错乱状态上启动任务 {task.id}",
                    is_timeout=False,
                )
            dep_branch = await executor.get_workspace_branch(dep_id)
            if dep_branch:
                parent_branch = dep_branch
                logger.info(
                    "任务 %s 基于依赖 %s 的分支 %s 创建 worktree",
                    task.id, dep_id, dep_branch,
                )
        else:
            # 多依赖：收集所有依赖分支，合并到中间分支
            # 跳过失败任务的分支（失败任务的 worktree 已被清理）
            dep_branches = []
            skipped_deps = []
            for dep_id in task.depends_on:
                # 检查依赖任务是否失败 — 失败任务的分支不可继承
                if dep_id in ctx.failed_tasks:
                    skipped_deps.append(dep_id)
                    logger.warning(
                        "任务 %s 的依赖 %s 已失败，跳过其分支",
                        task.id, dep_id,
                    )
                    continue
                dep_branch = await executor.get_workspace_branch(dep_id)
                if dep_branch:
                    dep_branches.append(dep_branch)
                else:
                    logger.warning(
                        "任务 %s 的依赖 %s 没有可用的 worktree 分支",
                        task.id, dep_id,
                    )

            if dep_branches:
                intermediate_branch_name = f"merge-deps-{task.id}"
                merge_result = await executor.merge_dependency_branches(
                    source_branches=dep_branches,
                    target_branch_name=intermediate_branch_name,
                )
                if merge_result.get("merged"):
                    parent_branch = intermediate_branch_name
                    logger.info(
                        "任务 %s 基于中间合并分支 %s 创建 worktree (依赖: %s)",
                        task.id, intermediate_branch_name,
                        ", ".join(dep_branches),
                    )
                else:
                    # 合并失败（冲突等），降级为使用第一个依赖分支
                    parent_branch = dep_branches[0]
                    logger.warning(
                        "多依赖分支合并失败，降级使用第一个依赖分支: %s (冲突: %s)",
                        parent_branch, merge_result.get("conflict_files", []),
                    )

        # 记录中间分支名，用于 DAG 完成后清理
        if intermediate_branch_name:
            ctx.intermediate_branches.add(intermediate_branch_name)

        # 2. 准备执行环境（Worktree + 隔离工具）
        # WorktreeError 现在是类化错误，包装成 TaskOutcome(NEVER_CREATED) 让
        # master 拿到结构化失败信号，不进入 ReAct
        try:
            exec_ctx = await executor.prepare_for_execution(
                task=task,
                config=subagent_config,
                agent_id=snapshot.node.agent_id,
                session_id=snapshot.node.session_id,
                parent_branch=parent_branch,
            )
        except WorktreeError as we:
            logger.error(
                "Worktree 创建失败 (task=%s) [%s]: %s",
                task.id, we.category.value, we,
            )
            ctx.task_outcomes[task.id] = build_outcome_for_workspace_failure(
                task_id=task.id,
                failure_category=we.category,
                failure_reason=str(we)[:200],
                branch_name=f"subagent-{task.id}",
            )
            raise DispatcherError(
                f"Git Worktree 创建失败 (task={task.id}): {we}",
                is_timeout=False,
            )

        # 2a. 确保子 Agent 的 session 已登记到 SessionManager
        # subagent session_id 为 "{master_session}:{task_id}" 的复合字符串，
        # 构造后从未调用 create_session，Agent.process_message 首行即会抛
        # ValueError("会话不存在")，子任务将在进入 ReAct 前即失败。
        await orchestrator.session_manager.get_or_create_subagent_session(
            snapshot.node.session_id,
            title=f"Subagent {task.id}",
            parent_session_id=ctx.master_node.session_id,
        )
        # 记录到 ctx，用于 DAG 完成后统一清理 subagent session
        ctx.subagent_session_ids.add(snapshot.node.session_id)

        # 2b. Worktree 失败策略：如果需要 worktree 但创建失败，直接中断
        if subagent_config.requires_worktree:
            if exec_ctx.workspace is None or exec_ctx.workspace.worktree_path is None:
                logger.error("Worktree 创建失败，任务中断: %s", task.id)
                raise DispatcherError("Git Worktree 创建失败，无法保证文件隔离安全")

        # 3. 重置验证门禁器（每个子任务独立验证状态）
        orchestrator.reset_validation()

        # 4. 构建隔离上下文
        isolator = ContextIsolator()

        # 收集依赖任务的执行结果
        dep_results = {}
        dep_structured_results = {}
        for dep_id in task.depends_on:
            if dep_id in ctx.task_results:
                dep_results[dep_id] = ctx.task_results[dep_id]
            if dep_id in ctx.task_structured_results:
                dep_structured_results[dep_id] = ctx.task_structured_results[dep_id]

        isolated_context = isolator.isolate(
            instruction=task.instruction,
            task_dependencies=dep_results,
            dependency_structured_results=dep_structured_results,
        )

        # 4b. 合并系统提示词 + 依赖上下文
        # 注意：Critic 警示**不**在这里注入。Critic 是 per-task 的（见下方
        # task_critic_injector 创建处），subagent 的失败模式只在它自己的
        # ReAct 循环里反思，绝不污染兄弟 / 后续 subagent 的 prompt。
        # 跨 task 的失败传递走 master：失败 task 的 outcome 会以
        # TaskGraphResult 的形式回到 master，master 自行决定如何反思和重派。
        system_prompt = subagent_config.system_prompt

        # 将依赖任务的结果/文件清单拼进 system_prompt，使 subagent 能感知上游产物。
        # to_messages() 会把 [依赖任务结果]/[相关文件]/[上下文信息] 作为 system 消息生成；
        # 直接把 content 追加到 system_prompt，就能随 Agent 的每次 LLM 调用一起送出，
        # 而不会依赖 session 历史（subagent 是新 session，历史为空）
        isolated_context.system_prompt = system_prompt
        aux_system_blocks: list[str] = []
        for aux_msg in isolated_context.to_messages():
            if aux_msg.get("role") != "system":
                continue
            content = aux_msg.get("content", "")
            if content and content != system_prompt:
                aux_system_blocks.append(content)
        if aux_system_blocks:
            system_prompt = system_prompt + "\n\n" + "\n\n".join(aux_system_blocks)

        # 5. 通过 Factory 构建 per-task ToolRegistry
        #    declarative：SubagentFactory.build_tool_registry 根据 config 的
        #    allowed_tools/forbidden_tools 过滤 master 工具并注入隔离版/验证版
        user_message = Message(
            role=MsgRole.USER,
            content=isolated_context.instruction,
        )

        worktree_path = await executor.get_workspace_path(task.id)

        # 隔离版文件/终端工具（仅 worktree 模式下提供）
        isolated_tools = None
        if subagent_config.requires_worktree and exec_ctx.tool_set:
            isolated_tools = list(exec_ctx.tool_set.get_tools())
            logger.info(
                "任务 %s 使用 IsolatedToolSet (worktree=%s)",
                task.id, worktree_path,
            )

        # 验证工具 + 验证门禁同时受两道门控制：
        #   1. requires_worktree —— 没有 worktree 就没有可验证的产物（Planner/Searcher）
        #   2. requires_validation —— 即便在 worktree 里，只读角色（Reviewer）也不需要验证
        # 任一为 False 就跳过 run_linter / run_tests 注入和 ValidationAware 包装。
        # 否则 Reviewer 看到工具表里有 run_linter，会被"提交前必须验证"提示词推入死循环。
        validation_enabled = (
            subagent_config.requires_worktree
            and subagent_config.requires_validation
        )

        extra_tools: list = []
        if validation_enabled:
            validation_root = worktree_path if worktree_path else orchestrator.repo_root
            extra_tools = [
                RunLinterTool(worktree_root=validation_root),
                RunTestsTool(worktree_root=validation_root),
            ]
            logger.info("验证工具已注册: run_linter, run_tests (root=%s)", validation_root)
        else:
            logger.info(
                "任务 %s 跳过验证工具注入（worktree=%s, validation=%s, 角色=%s）",
                task.id,
                subagent_config.requires_worktree,
                subagent_config.requires_validation,
                task.role,
            )

        if subagent_config.role_name == "CleanupAgent":
            from taskweave.tools.builtin.delete_artifact import (
                DeleteArtifactTool,
                extract_explicit_artifact_paths,
            )

            allowed_delete_paths = extract_explicit_artifact_paths(task.instruction)
            extra_tools.append(
                DeleteArtifactTool(
                    repo_root=orchestrator.repo_root,
                    allowed_paths=allowed_delete_paths,
                )
            )
            logger.info(
                "CleanupAgent delete_artifact allowlist: %s",
                allowed_delete_paths,
            )

        # submit_task_result 包装链（从内到外）：
        #   SubmitTaskResultTool          —— 实际落账
        #     └─ ValidationAwareSubmitTool —— 验证闭环（仅写代码角色）
        #          └─ FileClaimAwareSubmitTool —— 自报核对（A+B，所有角色）
        #
        # 内层 → 外层的语义递进：
        #   - SubmitTaskResultTool：最基本的结构化落账
        #   - ValidationAware：拦截"未通过 lint/test 就声明完成"
        #   - FileClaimAware：拦截"没写工具/不进 worktree 却声称改了文件"
        # FileClaimAware 必须在最外层，因为它的检查不依赖验证状态，所有角色
        # （包括 Planner/Searcher 这种不走 ValidationAware 的）都要被它管。
        original_submit = orchestrator.master_tool_registry.get("submit_task_result")
        submit_tool = None
        if original_submit:
            inner: Any = original_submit
            if (validation_enabled
                    and orchestrator.settings.subagent.require_validation
                    and orchestrator.validation_gatekeeper):
                inner = ValidationAwareSubmitTool(
                    gatekeeper=orchestrator.validation_gatekeeper,
                    original_submit_tool=inner,
                )
                logger.info("submit_task_result 已包装为 ValidationAwareSubmitTool")

            # 自报核对包装器始终启用 —— 任何角色都不应在 files_changed 里撒谎
            from taskweave.agents.critic.submit_claim_checker import (
                FileClaimAwareSubmitTool,
            )
            submit_tool = FileClaimAwareSubmitTool(
                inner_tool=inner,
                allowed_tools=subagent_config.allowed_tools,
                forbidden_tools=subagent_config.forbidden_tools,
                requires_worktree=subagent_config.requires_worktree,
                role_name=subagent_config.role_name,
            )
            logger.info(
                "submit_task_result 已包装为 FileClaimAwareSubmitTool "
                "(can_write=%s, requires_worktree=%s)",
                submit_tool._can_write, subagent_config.requires_worktree,
            )

        task_tool_registry = orchestrator.factory.build_tool_registry(
            config=subagent_config,
            master_tools=list(orchestrator.master_tool_registry.list_tools()),
            isolated_tools=isolated_tools,
            extra_tools=extra_tools,
            submit_tool=submit_tool,
        )

        try:
            # 6. 创建并执行 Subagent Agent
            from taskweave.agents.agent import Agent
            from taskweave.agents.critic.circuit_breaker import (
                CircuitBreaker,
                CircuitBreakerConfig,
            )
            from taskweave.config.settings import AgentConfig

            # 上下文窗口必须够装下：系统提示词 + 数次工具调用的 JSON 结果（验证工具结果可达数 KB）+ 历史对话。
            # 之前硬编码 4000 在 9-12 条消息就触发自动压缩，反复丢上下文 → agent 重复试错。
            # 这里给到 16000，配合 compaction_threshold=0.9 让压缩只在真的接近上限时发生。
            task_agent_config = AgentConfig(
                system_prompt=system_prompt,
                max_iterations=subagent_config.max_steps,
                max_context_tokens=16000,
                compaction_threshold=0.9,
            )

            # 为本 task 单独造一个工具级 CircuitBreaker，阈值从配置读取。
            # 跨 task 不共享：单个任务里的反复失败不应殃及其他任务。
            sub_cfg = orchestrator.settings.subagent
            task_tool_breaker = CircuitBreaker(CircuitBreakerConfig(
                failure_threshold=sub_cfg.tool_breaker_total_threshold,
                same_error_threshold=sub_cfg.tool_breaker_same_error_threshold,
            ))

            # Critic 注入器同样 per-task：subagent 的 ReAct 循环内自纠（"上次
            # 失败了，请指出错在哪、不要重复"）只在本 task 内有意义。如果共享
            # orchestrator 单例，task A 的失败上下文会泄漏到 task B 的 prompt，
            # 导致 task B 看到与自身无关的"你之前在 run_tests 失败了"警示。
            # 跨 task 的失败传递通过 TaskGraphResult 回到 master 自行反思。
            from taskweave.agents.critic.critic_injector import (
                CriticInjector,
                CriticConfig,
            )
            task_critic_injector = CriticInjector(CriticConfig())

            task_agent = Agent(
                config=task_agent_config,
                memory_config=orchestrator.settings.memory,
                provider_registry=orchestrator.master_provider_registry,
                tool_registry=task_tool_registry,
                session_manager=orchestrator.session_manager,
                memory_store=orchestrator.memory_store,
                validation_gatekeeper=orchestrator.validation_gatekeeper,
                tool_circuit_breaker=task_tool_breaker,
                critic_injector=task_critic_injector,
            )

            # 执行 ReAct 循环（带超时硬约束）
            # SubagentSpec.timeout_ms 之前只是元数据，没人监控；现在真正强制：
            # 在 task 层面用 asyncio.timeout() 包住整个 ReAct 流，超时 → 取消
            # 异步生成器，抛 TimeoutError，外层翻译为 DispatcherError(is_timeout=True)。
            timeout_seconds = max(subagent_config.timeout_ms / 1000.0, 1.0)
            accumulated_result = ""
            # 累计每种工具的调用次数；最终塞进 SystemObservation.tool_call_counts，
            # 让 master 在 timeout / circuit_breaker 失败时能看到 agent 的行为模式
            # （例：web_search×15 而 submit_task_result×0 = 死循环搜索）。
            tool_call_counts: dict[str, int] = {}
            event_stream = task_agent.process_message(
                user_message, snapshot.node.session_id
            )
            try:
                async with asyncio.timeout(timeout_seconds):
                    async for event in event_stream:
                        if event.event_type.value == "text_delta":
                            accumulated_result += event.data.get("text", "")
                        elif event.event_type.value == "text_done":
                            pass
                        elif event.event_type.value == "tool_call_result":
                            tool_name = event.data.get("name", "")
                            if tool_name:
                                tool_call_counts[tool_name] = (
                                    tool_call_counts.get(tool_name, 0) + 1
                                )
                            if tool_name == "submit_task_result":
                                result_data = event.data.get("result", "")
                                # 检查是否被 ValidationAwareSubmitTool 拦截（返回阻止消息）
                                if result_data.startswith("**提交被阻止**"):
                                    logger.warning(
                                        "任务 %s 的 submit_task_result 被验证门禁拦截",
                                        task.id,
                                    )
                                    accumulated_result += result_data
                                    continue

                                # 尝试解析结构化结果
                                import json
                                try:
                                    parsed = json.loads(result_data)
                                    result = validate_result(parsed)
                                    await executor.collect_result(task.id, parsed)

                                    # 存储结构化结果到上下文（供下游依赖任务使用）
                                    ctx.task_structured_results[task.id] = parsed

                                    # 任务成功 → 自动 git commit -am（匹配原始设计 ③.1）。
                                    # 所有要求 worktree 的成功任务都立即 commit：
                                    #   - 有下游：下游可基于 subagent-<id> 分支看到产物
                                    #   - 叶子：在 _finalize_dag_worktrees 阶段被合并到用户分支
                                    #
                                    # 关键：commit 失败现在是硬失败，不是 warning。
                                    # 旧行为：commit 失败 → logger.warning + return success
                                    # 新行为：commit 失败 → 抛 DispatcherError + 记录 outcome
                                    #          (CREATED_BUT_FAILED, WORKTREE_CREATION)
                                    # 这是断掉"Master 撒谎"的关键 —— 系统未能落地产物
                                    # 时，绝不让 task 报告为 success。
                                    commit_hash: str | None = None
                                    actual_files: list[str] = []
                                    workspace_state: WorkspaceFinalState
                                    if subagent_config.requires_worktree:
                                        try:
                                            commit_msg = (
                                                f"Subagent {task.id} task complete: "
                                                f"{result.summary[:80] if result.summary else ''}"
                                            )
                                            commit_info = await executor.commit_workspace(
                                                task.id, message=commit_msg,
                                            )
                                            if commit_info.get("committed"):
                                                commit_hash = commit_info.get("commit_hash")
                                                # commit 成功后查看实际变更（git 视角）
                                                actual_files = list(parsed.get("files_changed", []))
                                                logger.info(
                                                    "任务 %s worktree 已提交: commit=%s, files=%d",
                                                    task.id,
                                                    (commit_hash or "")[:8],
                                                    len(actual_files),
                                                )
                                                # 成功提交 → COMMITTED；DAG 收尾再 merge 到 main
                                                workspace_state = WorkspaceFinalState.COMMITTED_NOT_MERGED
                                            else:
                                                # 无变更也算"成功完成但没产物"
                                                logger.info(
                                                    "任务 %s worktree 无变更可提交 (%s)",
                                                    task.id,
                                                    commit_info.get("reason", "unknown"),
                                                )
                                                workspace_state = WorkspaceFinalState.COMMITTED_NOT_MERGED
                                        except WorktreeError as commit_err:
                                            # 类化错误：直接拿到 category，向 master 上报"系统判失败"
                                            logger.error(
                                                "任务 %s worktree 提交失败 [%s]: %s — 把 task 标为系统失败",
                                                task.id,
                                                commit_err.category.value,
                                                commit_err,
                                            )
                                            ctx.task_outcomes[task.id] = TaskOutcome(
                                                task_id=task.id,
                                                agent_report=AgentReport.from_structured(result),
                                                system_observation=SystemObservation(
                                                    workspace_final_state=WorkspaceFinalState.CREATED_BUT_FAILED,
                                                    failure_category=commit_err.category,
                                                    failure_reason=str(commit_err)[:200],
                                                    branch_name=f"subagent-{task.id}",
                                                    tool_call_counts=dict(tool_call_counts),
                                                ),
                                            )
                                            raise DispatcherError(
                                                f"Worktree 提交失败 (task={task.id}): {commit_err}",
                                                is_timeout=False,
                                            )
                                        except Exception as commit_err:
                                            # 非 WorktreeError 的兜底
                                            logger.error(
                                                "任务 %s worktree 提交未预期异常: %s — 把 task 标为系统失败",
                                                task.id, commit_err,
                                            )
                                            ctx.task_outcomes[task.id] = TaskOutcome(
                                                task_id=task.id,
                                                agent_report=AgentReport.from_structured(result),
                                                system_observation=SystemObservation(
                                                    workspace_final_state=WorkspaceFinalState.CREATED_BUT_FAILED,
                                                    failure_category=FailureCategory.WORKTREE_CREATION,
                                                    failure_reason=f"未预期异常: {type(commit_err).__name__}: {commit_err}",
                                                    branch_name=f"subagent-{task.id}",
                                                    tool_call_counts=dict(tool_call_counts),
                                                ),
                                            )
                                            raise DispatcherError(
                                                f"Worktree 提交失败 (task={task.id}): {commit_err}",
                                                is_timeout=False,
                                            )
                                    else:
                                        # 不需要 worktree 的角色
                                        workspace_state = WorkspaceFinalState.NOT_APPLICABLE
                                        actual_files = []

                                    # 记录"提交阶段成功"的 outcome —— 后续 DAG 收尾的 merge 步
                                    # 会更新 workspace_final_state 为 MERGED（如果 merge 成功）
                                    ctx.task_outcomes[task.id] = TaskOutcome(
                                        task_id=task.id,
                                        agent_report=AgentReport.from_structured(result),
                                        system_observation=SystemObservation(
                                            workspace_final_state=workspace_state,
                                            actual_files_landed=actual_files,
                                            branch_name=(
                                                f"subagent-{task.id}"
                                                if subagent_config.requires_worktree else None
                                            ),
                                            commit_hash=commit_hash,
                                            tool_call_counts=dict(tool_call_counts),
                                        ),
                                    )

                                    # 成功提交 — 记录到 CircuitBreaker
                                    if orchestrator.circuit_breaker:
                                        orchestrator.circuit_breaker.record_success()
                                    orchestrator.reset_critic()

                                    return result.to_master_context()
                                except json.JSONDecodeError:
                                    accumulated_result += result_data

                        elif event.event_type.value == "error":
                            error_msg = event.data.get("message", "未知错误")

                            # 记录到 orchestrator 级 Circuit Breaker（跨 task 模式追踪）。
                            # critic_injector 不在这里记录：task 即将抛 DispatcherError
                            # 终止，per-task 的 critic 也跟着废弃，记录是浪费。跨 task
                            # 的失败信号通过 TaskGraphResult 回到 master 反思。
                            if orchestrator.circuit_breaker:
                                orchestrator.circuit_breaker.record_failure(
                                    Exception(error_msg)
                                )

                            # 写 outcome 让 master 看到 tool_call_counts —— 否则
                            # 这条失败路径走 reconcile 的"无 outcome"分支，行为统计丢失
                            ctx.task_outcomes[task.id] = TaskOutcome(
                                task_id=task.id,
                                agent_report=AgentReport.absent(
                                    reason=f"Agent 执行错误: {error_msg[:120]}"
                                ),
                                system_observation=SystemObservation(
                                    workspace_final_state=(
                                        WorkspaceFinalState.CREATED_BUT_FAILED
                                        if subagent_config.requires_worktree
                                        else WorkspaceFinalState.NOT_APPLICABLE
                                    ),
                                    failure_category=FailureCategory.PROVIDER_ERROR,
                                    failure_reason=f"Agent 执行错误: {error_msg[:200]}",
                                    branch_name=(
                                        f"subagent-{task.id}"
                                        if subagent_config.requires_worktree else None
                                    ),
                                    tool_call_counts=dict(tool_call_counts),
                                ),
                            )
                            # breaker_recorded=True：本路径上面已调 record_failure，
                            # 标记后外层 catch-all 不会再记一次（双重记录 bug 修复）
                            raise DispatcherError(
                                f"Agent 执行错误: {error_msg}",
                                is_timeout=False,
                                breaker_recorded=True,
                            )
            except asyncio.TimeoutError:
                # 超时硬切断：尽量把异步生成器关掉，避免 leak
                import contextlib
                with contextlib.suppress(Exception):
                    await event_stream.aclose()
                logger.error(
                    "任务 %s 执行超时 (%dms)",
                    task.id, subagent_config.timeout_ms,
                )
                if orchestrator.circuit_breaker:
                    orchestrator.circuit_breaker.record_failure(
                        TimeoutError(
                            f"task {task.id} timeout {subagent_config.timeout_ms}ms"
                        )
                    )
                # 写 outcome 让 master 看到 tool_call_counts —— 这是诊断 timeout 类型 A/B
                # 的关键证据（agent 真的在干活 vs 死循环没 submit）
                ctx.task_outcomes[task.id] = TaskOutcome(
                    task_id=task.id,
                    agent_report=AgentReport.absent(
                        reason=f"Agent 执行超时: {subagent_config.timeout_ms}ms"
                    ),
                    system_observation=SystemObservation(
                        workspace_final_state=(
                            WorkspaceFinalState.CREATED_BUT_FAILED
                            if subagent_config.requires_worktree
                            else WorkspaceFinalState.NOT_APPLICABLE
                        ),
                        failure_category=FailureCategory.TIMEOUT,
                        failure_reason=f"任务执行超时: {subagent_config.timeout_ms}ms",
                        branch_name=(
                            f"subagent-{task.id}"
                            if subagent_config.requires_worktree else None
                        ),
                        tool_call_counts=dict(tool_call_counts),
                    ),
                )
                # breaker_recorded=True：本路径上面已调 record_failure，
                # 标记后外层 catch-all 不会再记一次（双重记录 bug 修复）
                raise DispatcherError(
                    f"任务执行超时: {subagent_config.timeout_ms}ms",
                    is_timeout=True,
                    breaker_recorded=True,
                )

            # Agent 未调用 submit_task_result — 用累积结果构建默认结果
            if accumulated_result:
                default_parsed = {
                    "status": "partial_success",
                    "summary": accumulated_result[:200],
                    "files_changed": [],
                    "unresolved_issues": "Agent 未调用 submit_task_result 提交结构化结果",
                }
                default_result = validate_result(default_parsed)

                # 存储结构化结果到上下文（供下游依赖任务使用）
                ctx.task_structured_results[task.id] = default_parsed

                # 部分成功也记录
                if orchestrator.circuit_breaker:
                    orchestrator.circuit_breaker.record_success()
                orchestrator.reset_critic()

                return default_result.to_master_context()
            else:
                return f"任务 '{task.id}' 执行完成，但未产生输出"

        except Exception as e:
            logger.error("Agent 执行失败: task=%s, error=%s", task.id, e)

            # 记录到 orchestrator 级 Circuit Breaker（追踪跨 task 失败模式）。
            # critic_injector 不在这里记录：per-task 的 critic 已经在 try 块退栈
            # 时随 task_critic_injector 一起废弃，跨 task 的失败信号由 master
            # 通过 TaskGraphResult 自行反思。
            #
            # 关键：跳过已被内层 timeout/error 分支记录过的失败 —— 它们抛出的
            # DispatcherError 自带 breaker_recorded=True 标记，外层再记会让
            # 同一次失败被算两次（bug：failure_count 翻倍 + fingerprint 不一致
            # 让"同种错误反复"检测失灵）。
            already_recorded = isinstance(e, DispatcherError) and getattr(
                e, "breaker_recorded", False,
            )
            if orchestrator.circuit_breaker and not already_recorded:
                orchestrator.circuit_breaker.record_failure(e)

            # 包装后透传 breaker_recorded 标记，让更上层调用栈也能识别
            # "本失败已记录过"，避免在更外层（如 _run_task_with_semaphore）
            # 万一也加了熔断时再记一次。
            # 注：本次修复范围内 is_timeout 信号丢失（原 DispatcherError.is_timeout
            # 在包装时被覆盖成 False）暂不修，后续讨论。
            raise DispatcherError(
                f"Agent 执行失败: {str(e)}",
                is_timeout=False,
                breaker_recorded=True,
            )

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
            "CleanupAgent": AgentRole.CLEANUP,
            "Cleanup": AgentRole.CLEANUP,
            "MasterAgent": AgentRole.MASTER,
            "Master": AgentRole.MASTER,
        }

        return role_mapping.get(role_name, AgentRole.GENERIC)

    def _get_or_create_dag(self, ctx: GraphExecutionContext):
        """获取或创建 DAG 对象

        在执行期间维护一个持久的 DAG，而不是每次调用都重新创建。
        """
        from taskweave.types.turn_snapshot import TaskDAG

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

    def _is_dependency_task(
        self,
        task: TaskNode,
        ctx: GraphExecutionContext,
    ) -> bool:
        """检查该任务是否有下游依赖（其他任务的 depends_on 包含此任务）

        有下游依赖的任务在执行结束后仅提交变更，不合并/清理 worktree，
        使依赖任务可基于此分支创建 worktree。
        """
        for other_task in ctx.request.tasks:
            if task.id in other_task.depends_on:
                return True
        return False

    async def _finalize_dag_worktrees(self, ctx: GraphExecutionContext) -> None:
        """DAG 完成后统一收尾：把成功子图的叶子分支合并回用户工作分支，再清理。

        匹配原始设计 ③.3 ——
            "Master Agent 收到结果报告后，在主干目录执行 git merge subagent-{taskId}"

        合并策略（关键）：
          只合并"成功子图的叶子任务"到用户分支。
          因为依赖关系是通过 git 父子分支表达的（test 的 worktree 基于
          subagent-implement 创建），叶子分支自动包含整条链路上所有上游的
          commit。把叶子合并到 main 即让 main 得到全部产物，而非依赖分支。

        旧逻辑的问题：
          - merge_workspace_to_parent 用的是 worktree.parent_branch，
            对叶子任务而言那是依赖分支（如 subagent-implement）而不是 main，
            导致叶子的产物只到达依赖分支，最终回不到主干。
          - 只在 is_committed=True 时合并，但叶子任务从未被 commit
            （旧的 _is_dependency_task 只 commit 有下游的任务）。

        失败任务的 worktree 同样统一清理，防止残留。
        """
        if self._orchestrator is None:
            return

        executor = self._orchestrator.subagent_executor
        auto_cleanup = executor.auto_cleanup
        auto_merge = self._orchestrator.settings.subagent.auto_merge

        # 1. 检测用户工作分支（master 启动时 repo_root 的当前分支）
        user_branch: str | None = None
        if auto_merge:
            try:
                user_branch = await executor.detect_user_branch()
                logger.info(
                    "DAG 完成，最终合并目标分支: %s (任务=%d 成功 / %d 失败)",
                    user_branch,
                    len(ctx.completed_tasks), len(ctx.failed_tasks),
                )
            except Exception as e:
                logger.warning(
                    "无法检测用户工作分支，跳过自动合并 — 产物仍在 subagent-* 分支可手动恢复: %s",
                    e,
                )

        # 2. 找成功子图的"叶子"任务：成功且没有任何已成功的下游任务
        completed = set(ctx.completed_tasks)
        has_completed_downstream: set[str] = set()
        for task in ctx.request.tasks:
            if task.id not in completed:
                continue
            for dep_id in task.depends_on:
                if dep_id in completed:
                    has_completed_downstream.add(dep_id)
        leaf_completed = completed - has_completed_downstream

        # 3. 合并叶子分支到用户分支
        # 记录成功 merge 的 leaf task_id —— 后续用来回溯 ancestors 的 outcome 升级
        merged_leaf_task_ids: set[str] = set()
        # 记录冲突的分支：master 收到结果后用 git_resolve_conflict 介入
        conflict_branches: list[dict[str, Any]] = []
        if user_branch and auto_merge:
            for original_leaf_id in leaf_completed:
                # B' 安全网：leaf 自身没 worktree（Searcher/Cleanup/Planner 当 leaf）
                # 时，沿 depends_on 找最近的有 worktree 的祖先，用它的分支当 effective
                # leaf 做 merge。避免"leaf 无 worktree → 整条上游产物丢失"的死结。
                workspace = executor.get_workspace_snapshot(original_leaf_id)
                task_id = original_leaf_id
                if workspace is None or workspace.worktree_info is None:
                    anchor_id = self._find_nearest_worktree_ancestor(
                        ctx, original_leaf_id, executor,
                    )
                    if anchor_id is None:
                        logger.debug(
                            "leaf %s 无 worktree 且无 worktree 祖先，跳过 merge"
                            "（合理：纯查询/纯通知 DAG）",
                            original_leaf_id,
                        )
                        continue
                    fallback_workspace = executor.get_workspace_snapshot(anchor_id)
                    if (
                        fallback_workspace is None
                        or fallback_workspace.worktree_info is None
                    ):
                        logger.warning(
                            "leaf %s 的祖先 %s 快照不可用，跳过 merge",
                            original_leaf_id, anchor_id,
                        )
                        continue
                    logger.info(
                        "leaf %s 无 worktree → 回退到祖先 %s 的分支 %s 做 merge"
                        "（B' 安全网）",
                        original_leaf_id, anchor_id,
                        fallback_workspace.worktree_info.branch_name,
                    )
                    task_id = anchor_id
                    workspace = fallback_workspace
                if not workspace.is_committed:
                    logger.warning(
                        "叶子任务 %s 未提交（is_committed=False），无法合并到 %s — 产物丢失",
                        task_id, user_branch,
                    )
                    continue
                branch = workspace.worktree_info.branch_name
                try:
                    merge_result = await executor.merge_branch_to_target(
                        task_id=task_id,
                        target_branch=user_branch,
                        delete_branch=False,  # 后面统一清理
                    )
                    workspace.is_merged = merge_result.get("merged", False)
                    if workspace.is_merged:
                        merged_leaf_task_ids.add(task_id)
                        logger.info(
                            "DAG 叶子已合并到 %s: task=%s, branch=%s",
                            user_branch, task_id, branch,
                        )
                        # 把 outcome 推到 MERGED —— 这是任务真正落地成功的唯一信号
                        existing = ctx.task_outcomes.get(task_id)
                        if existing is not None:
                            existing.system_observation = SystemObservation(
                                workspace_final_state=WorkspaceFinalState.MERGED,
                                actual_files_landed=existing.system_observation.actual_files_landed,
                                branch_name=existing.system_observation.branch_name,
                                commit_hash=existing.system_observation.commit_hash,
                            )
                    elif merge_result.get("has_conflicts"):
                        # 冲突：分支保留 + 记录到 ctx.result.task_errors，让 master
                        # 拿到清晰的"用 git_resolve_conflict 介入"指示。
                        conflict_files = merge_result.get("conflict_files") or []
                        logger.error(
                            "DAG 叶子合并到 %s 冲突: task=%s, files=%s",
                            user_branch, task_id, conflict_files,
                        )
                        conflict_branches.append({
                            "task_id": task_id,
                            "branch": branch,
                            "conflict_files": conflict_files,
                        })
                        # 把 outcome 标为 COMMITTED_NOT_MERGED + 冲突原因
                        # 系统观察：分支上有 commit，但用户分支没拿到产物
                        existing = ctx.task_outcomes.get(task_id)
                        if existing is not None:
                            existing.system_observation = SystemObservation(
                                workspace_final_state=WorkspaceFinalState.COMMITTED_NOT_MERGED,
                                actual_files_landed=[],  # 用户分支上没拿到产物
                                failure_category=FailureCategory.WORKTREE_CREATION,
                                failure_reason=(
                                    f"merge 到 {user_branch} 冲突，"
                                    f"涉及 {len(conflict_files)} 个文件"
                                ),
                                branch_name=existing.system_observation.branch_name,
                                commit_hash=existing.system_observation.commit_hash,
                            )
                        # 在 result 上记一条人读得懂的错误，使 master 看到任务图结果
                        # 时知道要调 git_resolve_conflict
                        if hasattr(ctx, "result") and ctx.result is not None:
                            ctx.result.task_errors[task_id] = (
                                f"merge 到 {user_branch} 出现冲突，"
                                f"涉及文件 {conflict_files[:5]}。"
                                "请调用 git_resolve_conflict(action=list) 查看，"
                                "再用 take_ours/take_theirs/mark_resolved 解决，"
                                "最后 finalize 或 abort。"
                            )
                    else:
                        logger.warning(
                            "DAG 叶子合并失败: task=%s, error=%s",
                            task_id, merge_result.get("error"),
                        )
                        # merge 失败但非冲突 —— 系统判定 task 没落地
                        existing = ctx.task_outcomes.get(task_id)
                        if existing is not None:
                            existing.system_observation = SystemObservation(
                                workspace_final_state=WorkspaceFinalState.CREATED_BUT_FAILED,
                                actual_files_landed=[],
                                failure_category=FailureCategory.WORKTREE_CREATION,
                                failure_reason=(
                                    f"merge 失败: {merge_result.get('error') or 'unknown'}"
                                )[:200],
                                branch_name=existing.system_observation.branch_name,
                                commit_hash=existing.system_observation.commit_hash,
                            )
                except Exception as e:
                    logger.warning(
                        "DAG 叶子合并异常: task=%s, error=%s", task_id, e,
                    )
                    existing = ctx.task_outcomes.get(task_id)
                    if existing is not None:
                        existing.system_observation = SystemObservation(
                            workspace_final_state=WorkspaceFinalState.CORRUPTED,
                            actual_files_landed=[],
                            failure_category=FailureCategory.WORKTREE_CREATION,
                            failure_reason=f"merge 异常: {type(e).__name__}: {e}"[:200],
                            branch_name=existing.system_observation.branch_name,
                            commit_hash=existing.system_observation.commit_hash,
                        )

        # 3b. 回溯 ancestors：把"被下游 leaf 间接 merge 进主干"的中间节点
        #     outcome 从 COMMITTED_NOT_MERGED 升级为 MERGED_VIA_DESCENDANT。
        #
        #     设计动机：DAG 中间节点（如 implement，被 test 依赖）的 worktree
        #     从不被独立 merge 到用户分支——只有 leaf（test）会 merge。但 leaf
        #     的 worktree 基于 ancestor 的 subagent 分支创建，git 父子链让
        #     leaf 的 commit 包含整条 ancestor 链。所以 leaf 一旦 merge 进主干，
        #     ancestor 的产物**事实上已经落地**。
        #
        #     不做这一步的后果：task_outcome.final_status 把 COMMITTED_NOT_MERGED
        #     恒判 FAILED，reconcile 会把这些"实际成功"的中间节点错误地从
        #     completed 移到 failed，导致 master 看到假阴性的 partial 结果，
        #     做出无意义的 retry。
        if merged_leaf_task_ids:
            self._propagate_merged_via_descendants(ctx, merged_leaf_task_ids)

        # 冲突分支不能 auto_cleanup —— 否则用户的 commit 同时丢失。
        # 在闭包里记录这些 branch 名，下面清理逻辑会跳过。
        protected_task_ids = {c["task_id"] for c in conflict_branches}
        if conflict_branches:
            logger.warning(
                "保留 %d 个冲突 worktree（master 可调用 git_resolve_conflict 介入）: %s",
                len(conflict_branches),
                [c["branch"] for c in conflict_branches],
            )

        # 4. 清理所有活跃的 worktree（不再需要保留）。
        #    冲突中的 worktree 跳过，让 master 有机会通过 git_resolve_conflict 介入。
        for task_id in list(executor.list_active_task_ids()):
            if task_id in protected_task_ids:
                logger.info("跳过冲突 worktree 清理: task=%s（保留供 master 解决）", task_id)
                continue
            try:
                if auto_cleanup:
                    await executor.remove_worktree(task_id, force=True)
                    workspace = executor.get_workspace_snapshot(task_id)
                    if workspace:
                        workspace.is_active = False
                    logger.info("DAG worktree 已清理: task=%s", task_id)
            except Exception as e:
                logger.warning(
                    "DAG worktree 清理失败: task=%s, error=%s", task_id, e,
                )

        # 5. 安全清理本 DAG 产生的所有 subagent-* 分支（叶子 + 中间节点）
        # 改用 git branch -d 的"已合并才删"语义，与启动期 prune 共用一套实现：
        #   - 叶子分支：被 step 3 显式 merge 进 user_branch，-d 必成功
        #   - 中间节点分支：自身没被独立 merge，但 commit 通过 git 父子链
        #     随叶子 merge 进了 user_branch，对 -d 而言同样"已合并" → 删除
        #   - 未合并分支（DAG 部分失败、叶子未 merge）：git -d 会拒绝，保留
        # 冲突分支（protected_task_ids）显式跳过，让 master 调
        # git_resolve_conflict 介入。
        if auto_cleanup:
            dag_branches: list[str] = []
            for task_id in completed:
                if task_id in protected_task_ids:
                    continue
                ws = executor.get_workspace_snapshot(task_id)
                if ws is not None and ws.worktree_info is not None:
                    dag_branches.append(ws.worktree_info.branch_name)
            if dag_branches:
                try:
                    prune_result = await executor.safe_prune_subagent_branches(
                        branch_names=dag_branches,
                    )
                    if prune_result["deleted"]:
                        logger.info(
                            "DAG 完成清理 %d 个 subagent 分支: %s",
                            len(prune_result["deleted"]),
                            prune_result["deleted"][:5],
                        )
                except Exception as e:
                    logger.warning("DAG subagent 分支清理异常: %s", e)

        # 6. 清理多依赖中间分支（merge-deps-* 由 merge_branches_into_new_branch 创建）
        # 这些不属于 subagent-* 命名空间，仍用 -D 强删
        for branch_name in ctx.intermediate_branches:
            try:
                await executor.delete_branch(branch_name)
                logger.info("中间分支已清理: %s", branch_name)
            except Exception as e:
                logger.warning("中间分支清理失败: %s, error=%s", branch_name, e)

    def _find_nearest_worktree_ancestor(
        self,
        ctx: GraphExecutionContext,
        task_id: str,
        executor: Any,
    ) -> str | None:
        """BFS 沿 depends_on 找最近的有 worktree 的祖先任务 ID。

        用于 leaf-merge 的 B' 安全网：当 leaf 自身不进 worktree（如 Searcher/
        Cleanup/Planner 当 leaf）时，回退到上游最近的有 worktree 的任务，用其
        分支作为 effective leaf merge 到用户分支。

        Returns:
            最近祖先的 task_id，或 None 表示这条链上没有任何任务进 worktree
            （合理场景：纯查询 / 纯通知 DAG，确实没有可落地的产物）
        """
        task_map = {t.id: t for t in ctx.request.tasks}
        if task_id not in task_map:
            return None
        visited: set[str] = set()
        queue: list[str] = list(task_map[task_id].depends_on)
        while queue:
            dep_id = queue.pop(0)
            if dep_id in visited or dep_id not in task_map:
                continue
            visited.add(dep_id)
            ws = executor.get_workspace_snapshot(dep_id)
            if ws is not None and ws.worktree_info is not None:
                return dep_id
            queue.extend(task_map[dep_id].depends_on)
        return None

    def _propagate_merged_via_descendants(
        self,
        ctx: GraphExecutionContext,
        merged_leaf_task_ids: set[str],
    ) -> None:
        """从成功 merge 的 leaf 出发回溯 DAG，把 COMMITTED_NOT_MERGED 状态的
        ancestor 升级为 MERGED_VIA_DESCENDANT。

        只升级 COMMITTED_NOT_MERGED：
          - MERGED：本身就是 leaf，已经直接 merge，不动
          - NOT_APPLICABLE：Planner/Searcher 这种不进 worktree 的角色，不动
          - CREATED_BUT_FAILED / CORRUPTED / CLEANED_NO_PRODUCT：已经在失败终态，
            不应被 ancestor 路径"洗白"成成功
          - NEVER_CREATED：连 worktree 都没创建，不可能产物在主干上
        """
        task_map = {t.id: t for t in ctx.request.tasks}
        visited: set[str] = set()

        def walk(tid: str) -> None:
            if tid in visited or tid not in task_map:
                return
            visited.add(tid)
            for dep_id in task_map[tid].depends_on:
                walk(dep_id)
                existing = ctx.task_outcomes.get(dep_id)
                if existing is None:
                    continue
                current_state = existing.system_observation.workspace_final_state
                if current_state != WorkspaceFinalState.COMMITTED_NOT_MERGED:
                    # 仅升级"已 commit 但未独立 merge"的中间节点
                    continue
                existing.system_observation = SystemObservation(
                    workspace_final_state=WorkspaceFinalState.MERGED_VIA_DESCENDANT,
                    actual_files_landed=existing.system_observation.actual_files_landed,
                    branch_name=existing.system_observation.branch_name,
                    commit_hash=existing.system_observation.commit_hash,
                )
                logger.info(
                    "中间任务 %s outcome 升级 MERGED_VIA_DESCENDANT "
                    "(产物经下游 leaf 进入主干)",
                    dep_id,
                )

        for leaf_id in merged_leaf_task_ids:
            walk(leaf_id)

    def _reconcile_outcomes_into_result(self, ctx: GraphExecutionContext) -> None:
        """系统观察 vs agent 自报的最终对账。

        在 _execute_levels 完成后、status 判定前调用一次：
          1. 把每个 task 的 TaskOutcome 同步到 result.task_outcomes
          2. 对于 outcome.final_status == FAILED 但仍在 completed_tasks 的 id：
             - 从 completed_tasks 移除
             - 加进 failed_tasks
             - 用 outcome 信息填充 task_errors / task_failure_categories
          3. 反向：outcome 显示成功但仍在 failed_tasks 的（极罕见，merge 异常恢复）
             也尊重 outcome
        """
        from taskweave.types.structured_result import TaskStatus
        result = ctx.result

        # 1. outcome 同步到 result（master 据此读 discrepancy 等字段）
        for task_id, outcome in ctx.task_outcomes.items():
            result.task_outcomes[task_id] = outcome

        # 2. 重判：以 outcome.final_status 为权威
        for task_id, outcome in list(ctx.task_outcomes.items()):
            final = outcome.final_status

            in_completed = task_id in result.completed_tasks
            in_failed = task_id in result.failed_tasks

            if final == TaskStatus.FAILED and in_completed and not in_failed:
                # Agent 报告成功但系统判失败 —— 调度器需要纠偏
                obs = outcome.system_observation

                # 兜底：reconcile 时 failure_reason 必须非空，否则 master 拿到的
                # task_errors 是空字符串，根本无法反思。任何把 task 翻成 FAILED
                # 的代码路径都应该先把 failure_reason 填上；这里做最后一道补救：
                # 没填就基于现有 outcome 字段合成一段可读说明，并同步回 obs，
                # 让"(无原因)" 这种"调度器自己也说不清" 的状态永远不出现。
                if not obs.failure_reason:
                    files_preview = obs.actual_files_landed[:3]
                    agent_summary = (outcome.agent_report.summary or "").strip()
                    synthesized = (
                        f"workspace 终态={obs.workspace_final_state.value}; "
                        f"实际落入主干文件={files_preview} "
                        f"(共 {len(obs.actual_files_landed)} 个); "
                        f"agent 自报 {outcome.agent_report.status.value}: "
                        f"{agent_summary[:120]}"
                    )
                    obs.failure_reason = synthesized
                    logger.warning(
                        "reconcile 检测到 task %s outcome 缺失 failure_reason，"
                        "已合成兜底说明（请检查上游 outcome 写入路径是否漏填）",
                        task_id,
                    )

                logger.warning(
                    "outcome 重判：task %s 从 completed → failed（系统观察）：%s",
                    task_id,
                    obs.failure_reason,
                )
                result.completed_tasks.remove(task_id)
                if task_id not in result.failed_tasks:
                    result.failed_tasks.append(task_id)
                ctx.completed_tasks.discard(task_id)
                ctx.failed_tasks.add(task_id)
                # 把失败原因和分类写进 result（master 提示词模板会读）
                # failure_reason 经过上面的兜底已保证非空
                result.task_errors[task_id] = obs.failure_reason
                if obs.failure_category is not None:
                    result.task_failure_categories[task_id] = obs.failure_category

            elif final == TaskStatus.SUCCESS and in_failed and not in_completed:
                # 极罕见：agent 自认失败但系统观察落地（基本不会发生）
                logger.info(
                    "outcome 重判：task %s 从 failed → completed（系统观察）",
                    task_id,
                )
                result.failed_tasks.remove(task_id)
                if task_id not in result.completed_tasks:
                    result.completed_tasks.append(task_id)
                ctx.failed_tasks.discard(task_id)
                ctx.completed_tasks.add(task_id)
                result.task_errors.pop(task_id, None)
                result.task_failure_categories.pop(task_id, None)

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
        self.task_results: dict[str, str] = {}  # task_id -> result (text)

        # 结构化结果 (task_id -> StructuredResult dict with status, files_changed, summary, etc.)
        self.task_structured_results: dict[str, dict[str, Any]] = {}

        # 双层结果：每个 task 的 TaskOutcome（含 agent_report + system_observation）。
        # 这才是 Master 决策的真理来源 —— task_structured_results 只是 agent 自报，
        # 不区分"agent 说的"和"系统看到的"。Master 拿 outcome 来判定 final_status。
        self.task_outcomes: dict[str, TaskOutcome] = {}

        # 多依赖合并产生的中间分支，DAG 完成后需清理
        self.intermediate_branches: set[str] = set()

        # 本次 DAG 派生出的 subagent session id，DAG 完成后归档
        self.subagent_session_ids: set[str] = set()


# 导出
__all__ = ["TaskScheduler", "GraphExecutionContext"]
