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
            # 状态分级，反映真实结果：
            #   - 全失败  → failed
            #   - 有失败  → partial（之前错报为 completed，导致 Master 误判）
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
        from miniclaw.agents.context_isolator import ContextIsolator
        from miniclaw.agents.critic.validation_gatekeeper import ValidationAwareSubmitTool
        from miniclaw.agents.critic.validation_tools import RunLinterTool, RunTestsTool
        from miniclaw.tools.registry import ToolRegistry
        from miniclaw.types.structured_result import validate_result
        from miniclaw.types.messages import Message
        from miniclaw.types.enums import Role as MsgRole

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
        exec_ctx = await executor.prepare_for_execution(
            task=task,
            config=subagent_config,
            agent_id=snapshot.node.agent_id,
            session_id=snapshot.node.session_id,
            parent_branch=parent_branch,
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

        # 4b. 合并系统提示词 + Critic 警示注入 + 依赖上下文
        system_prompt = subagent_config.system_prompt
        if orchestrator.critic_injector and orchestrator.critic_injector.has_recent_failure():
            warning = orchestrator.critic_injector.get_warning_prompt()
            system_prompt += "\n\n" + warning
            logger.info("Critic 警示已注入到任务 %s 的提示词", task.id)

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

        # submit_task_result（仅产出代码的角色才包装为 ValidationAwareSubmitTool）
        original_submit = orchestrator.master_tool_registry.get("submit_task_result")
        submit_tool = None
        if original_submit:
            if (validation_enabled
                    and orchestrator.settings.subagent.require_validation
                    and orchestrator.validation_gatekeeper):
                submit_tool = ValidationAwareSubmitTool(
                    gatekeeper=orchestrator.validation_gatekeeper,
                    original_submit_tool=original_submit,
                )
                logger.info("submit_task_result 已包装为 ValidationAwareSubmitTool")
            else:
                submit_tool = original_submit

        task_tool_registry = orchestrator.factory.build_tool_registry(
            config=subagent_config,
            master_tools=list(orchestrator.master_tool_registry.list_tools()),
            isolated_tools=isolated_tools,
            extra_tools=extra_tools,
            submit_tool=submit_tool,
        )

        try:
            # 6. 创建并执行 Subagent Agent
            from miniclaw.agents.agent import Agent
            from miniclaw.config.settings import AgentConfig

            # 上下文窗口必须够装下：系统提示词 + 数次工具调用的 JSON 结果（验证工具结果可达数 KB）+ 历史对话。
            # 之前硬编码 4000 在 9-12 条消息就触发自动压缩，反复丢上下文 → agent 重复试错。
            # 这里给到 16000，配合 compaction_threshold=0.9 让压缩只在真的接近上限时发生。
            task_agent_config = AgentConfig(
                system_prompt=system_prompt,
                max_iterations=subagent_config.max_steps,
                max_context_tokens=16000,
                compaction_threshold=0.9,
            )

            task_agent = Agent(
                config=task_agent_config,
                memory_config=orchestrator.settings.memory,
                provider_registry=orchestrator.master_provider_registry,
                tool_registry=task_tool_registry,
                session_manager=orchestrator.session_manager,
                memory_store=orchestrator.memory_store,
                validation_gatekeeper=orchestrator.validation_gatekeeper,
            )

            # 执行 ReAct 循环
            accumulated_result = ""
            async for event in task_agent.process_message(
                user_message, snapshot.node.session_id
            ):
                if event.event_type.value == "text_delta":
                    accumulated_result += event.data.get("text", "")
                elif event.event_type.value == "text_done":
                    pass
                elif event.event_type.value == "tool_call_result":
                    tool_name = event.data.get("name", "")
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
                            # 之前这里有 _is_dependency_task 判断，只对"有下游"的任务 commit；
                            # 这导致叶子任务的产物从未进入 git 历史，worktree 清理后直接丢失。
                            # 改为：所有要求 worktree 的成功任务都立即 commit，
                            #   - 有下游：下游可基于 subagent-<id> 分支看到产物
                            #   - 叶子：在 _finalize_dag_worktrees 阶段被合并到用户分支
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
                                        logger.info(
                                            "任务 %s worktree 已提交: commit=%s, files=%d",
                                            task.id,
                                            (commit_info.get("commit_hash") or "")[:8],
                                            len(parsed.get("files_changed", [])),
                                        )
                                    else:
                                        # 无变更（比如纯审查任务）— 不算错误
                                        logger.info(
                                            "任务 %s worktree 无变更可提交 (%s)",
                                            task.id,
                                            commit_info.get("reason", "unknown"),
                                        )
                                except Exception as commit_err:
                                    # commit 失败但任务本身成功 — 记录但不阻塞。
                                    # finalize 阶段会因 is_committed=False 跳过 merge，
                                    # 用户会在主干看不到该任务产物，需要人工恢复 subagent-<id> 分支。
                                    logger.warning(
                                        "任务 %s worktree 提交失败: %s — 主干将看不到产物",
                                        task.id, commit_err,
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

                    # 记录到 Circuit Breaker
                    if orchestrator.circuit_breaker:
                        pattern = orchestrator.circuit_breaker.record_failure(
                            Exception(error_msg)
                        )
                        # 记录到 Critic 注入器
                        if orchestrator.critic_injector:
                            orchestrator.critic_injector.record_failure(
                                tool_name="agent_execution",
                                tool_arguments={"task_id": task.id},
                                error=error_msg,
                                error_pattern=pattern,
                            )

                    raise DispatcherError(f"Agent 执行错误: {error_msg}", is_timeout=False)

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

            # 记录到 Circuit Breaker
            if orchestrator.circuit_breaker:
                orchestrator.circuit_breaker.record_failure(e)

            # 记录到 Critic 注入器
            if orchestrator.critic_injector:
                orchestrator.critic_injector.record_failure(
                    tool_name="agent_execution",
                    tool_arguments={"task_id": task.id},
                    error=str(e),
                )

            raise DispatcherError(f"Agent 执行失败: {str(e)}", is_timeout=False)

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
        merged_branches: set[str] = set()
        if user_branch and auto_merge:
            for task_id in leaf_completed:
                workspace = executor.get_workspace_snapshot(task_id)
                if workspace is None or workspace.worktree_info is None:
                    continue
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
                        merged_branches.add(branch)
                        logger.info(
                            "DAG 叶子已合并到 %s: task=%s, branch=%s",
                            user_branch, task_id, branch,
                        )
                    elif merge_result.get("has_conflicts"):
                        # 真实冲突：原设计要 master 介入解决，目前先记 ERROR
                        logger.error(
                            "DAG 叶子合并到 %s 冲突: task=%s, files=%s "
                            "(分支保留供人工解决)",
                            user_branch, task_id, merge_result.get("conflict_files"),
                        )
                    else:
                        logger.warning(
                            "DAG 叶子合并失败: task=%s, error=%s",
                            task_id, merge_result.get("error"),
                        )
                except Exception as e:
                    logger.warning(
                        "DAG 叶子合并异常: task=%s, error=%s", task_id, e,
                    )

        # 4. 清理所有活跃的 worktree（不再需要保留）
        for task_id in list(executor.list_active_task_ids()):
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

        # 5. 清理已合并的 subagent-* 分支
        # 未合并的（冲突 / 失败）保留，便于用户手动恢复
        if auto_cleanup:
            for branch_name in merged_branches:
                try:
                    await executor.delete_branch(branch_name)
                except Exception:
                    pass

        # 6. 清理多依赖中间分支
        for branch_name in ctx.intermediate_branches:
            try:
                await executor.delete_branch(branch_name)
                logger.info("中间分支已清理: %s", branch_name)
            except Exception as e:
                logger.warning("中间分支清理失败: %s, error=%s", branch_name, e)

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

        # 多依赖合并产生的中间分支，DAG 完成后需清理
        self.intermediate_branches: set[str] = set()

        # 本次 DAG 派生出的 subagent session id，DAG 完成后归档
        self.subagent_session_ids: set[str] = set()


# 导出
__all__ = ["TaskScheduler", "GraphExecutionContext"]