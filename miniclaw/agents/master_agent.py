"""
Master Agent — 多智能体架构的主调度器

Master Agent 的职责:
  1. 分析用户请求，判断是否需要分解任务
  2. 使用 create_task_graph 工具构建 DAG
  3. 监控子任务执行状态
  4. 根据结果做出后续决策

执行模式采用状态机循环（而非硬编码两轮交互）:
  - 每轮: Agent 分析 → (创建 task_graph?) → 执行 → 注入结果 → 下一轮
  - Agent 自己决定下一步: 重试、新建图、或直接总结
  - max_task_graph_rounds 防止无限循环

Master Agent 不直接执行具体工作，只做调度和决策。
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator, Any

from miniclaw.agents.agent import Agent
from miniclaw.agents.critic.circuit_breaker import CircuitBreakerError
from miniclaw.agents.prompts import MASTER_AGENT_PROMPT
from miniclaw.agents.subagent_orchestrator import SubagentOrchestrator
from miniclaw.config.settings import AgentConfig, Settings
from miniclaw.types.enums import AgentRole, Role
from miniclaw.types.events import Event
from miniclaw.types.messages import Message
from miniclaw.types.task_graph import (
    FailureCategory,
    TaskGraphResult,
    compute_dag_signature,
)
from miniclaw.types.task_outcome import (
    TaskOutcome,
    WorkspaceFinalState,
)
from miniclaw.types.turn_snapshot import AgentNode
from miniclaw.utils.logging import get_logger

logger = get_logger(__name__)


class MasterAgent:
    """Master Agent — 主调度器
        如果 Agent 直接回复（未建图）→ 结束
    """

    def __init__(
        self,
        base_agent: Agent,
        orchestrator: SubagentOrchestrator,
        settings: Settings,
    ):
        self._agent = base_agent
        self._orchestrator = orchestrator
        self._settings = settings

    async def process_message(
        self,
        user_message: Message,
        session_id: str,
    ) -> AsyncIterator[Event]:
        """处理用户消息 — 状态机循环模式

        流程:
          1. 创建根 DAG + master_node（Dispatcher 状态跟踪）
          2. Agent 分析任务
          3. 如果建图 → 执行 DAG → 注入结果 → 回到步骤 2
          4. 如果未建图 → Agent 直接回复 → 结束
          5. 达到 max_task_graph_rounds → 强制总结轮
        """
        max_rounds = self._settings.subagent.max_task_graph_rounds
        round_num = 0
        current_message = user_message

        # 创建 Master Agent 节点 + 根 DAG（确保 Dispatcher 有完整 DAG 树）
        master_node = AgentNode(
            agent_id="master-001",
            role=AgentRole.MASTER,
            depth=0,
            session_id=session_id,
        )
        await self._orchestrator.create_root_dag(
            root_agent_id=master_node.agent_id,
            session_id=session_id,
        )

        logger.info("Master Agent 开始处理消息: session=%s, max_rounds=%d", session_id, max_rounds)
        yield Event.thinking(session_id)

        # 累计跨轮的状态，供最后兜底摘要使用
        seen_dag_signatures: dict[str, int] = {}  # signature -> first round
        all_graph_results: list[TaskGraphResult] = []
        # last_text_emitted 跟踪 agent 最近一次"无工具调用"的文本回复
        # 用作 UX 契约：master 退出时如果没给用户任何文字，要兜底一段总结
        last_text_emitted: bool = False
        original_user_input = user_message.content or ""

        try:
            while round_num < max_rounds:
                round_num += 1
                logger.info("Master Agent 第 %d/%d 轮", round_num, max_rounds)

                # 运行 Agent — 它可能创建 task_graph 或直接回复
                graph_id = None
                has_task_graph = False
                pending_tool_name: str | None = None

                # 关键: 拿到 graph_id 立刻打断内层 ReAct 循环
                # 否则内层 agent 看到 create_task_graph 只返回一个计划 ack
                # （还没有实际执行结果），会以为"啥都没发生"而连续重复调用
                # 直到撞上自己的 max_iterations 才停 — 形成可观测的"狂刷 DAG 死循环。要在外层接管：拿到 graph_id 就 break，去 _wait_for_graph_result
                # 跑真正的执行，再把结果喂回下一轮
                event_iter = self._agent.process_message(current_message, session_id)
                try:
                    async for event in event_iter:
                        ev_type = event.event_type.value

                        if ev_type == "tool_call_start":
                            pending_tool_name = event.data.get("name")
                            if pending_tool_name == "create_task_graph":
                                has_task_graph = True
                                logger.info(
                                    "检测到 create_task_graph 调用 (round %d)", round_num,
                                )

                        if ev_type == "tool_call_result":
                            tool_name = event.data.get("name") or pending_tool_name
                            if tool_name == "create_task_graph":
                                result_str = event.data.get("result", "")
                                gid = self._extract_graph_id(result_str)
                                if gid:
                                    graph_id = gid
                            pending_tool_name = None

                        # 用户契约：跟踪 agent 实质性文字输出
                        if ev_type == "text_done":
                            text = event.data.get("text", "") or ""
                            if text.strip():
                                last_text_emitted = True

                        yield event

                        # 拿到有效 graph_id → 立即跳出，交给 _wait_for_graph_result
                        if has_task_graph and graph_id:
                            break
                finally:
                    # 显式关闭异步生成器，触发其内部 finally/cleanup
                    await event_iter.aclose()

                # Agent 未创建任务图 → 直接回复 → 结束
                if not has_task_graph or not graph_id:
                    logger.info("Master Agent 未创建任务图，结束循环 (round %d)", round_num)
                    break

                # 执行任务图 → 注入结果 → 下一轮 Agent 自主决策
                logger.info("任务图 %s 已创建，执行并注入结果 (round %d)", graph_id, round_num)
                # 提前抓 request，因为 _wait_for_graph_result 会 pop 掉它，但 _build_result_message 需要从中取失败任务的原始 instruction
                request_for_retry = self._orchestrator.pending_graph_store.get(graph_id)

                # 反模式检测：本轮 DAG 在指纹层面与历史某轮等价（仅改了 task.id
                # 或排版的"假重派"）。Master 第二次撞同样的 DAG 几乎一定再撞同
                # 一面墙——直接拒绝执行，给 LLM 一段强行更弦换辙的指令
                if request_for_retry is not None:
                    sig = compute_dag_signature(request_for_retry)
                    if sig in seen_dag_signatures:
                        prior_round = seen_dag_signatures[sig]
                        logger.warning(
                            "DAG 指纹重复 (round %d ↔ round %d, sig=%s) — 拒绝执行同样的任务图",
                            round_num, prior_round, sig,
                        )
                        # 抛弃该 graph，给 master 一段直白的纠偏 prompt
                        try:
                            await self._orchestrator.pending_graph_store.pop(graph_id)
                        except Exception:
                            pass
                        current_message = self._build_duplicate_dag_message(
                            prior_round=prior_round,
                            current_round=round_num,
                        )
                        continue
                    seen_dag_signatures[sig] = round_num

                graph_result = await self._wait_for_graph_result(graph_id, master_node)
                if graph_result is not None:
                    all_graph_results.append(graph_result)
                current_message = self._build_result_message(
                    graph_result, request_for_retry,
                )
                # 循环继续 — Agent 根据结果决定下一步

            # 达到最大轮数 — 强制总结轮
            if has_task_graph and round_num >= max_rounds:
                logger.warning(
                    "达到最大任务图轮数 %d，请求 Agent 最终总结",
                    max_rounds,
                )
                summary_prompt = (
                    "请根据以上所有执行结果，给出最终总结。"
                    "不要再创建新的任务图（已达到最大轮数限制）。"
                    "如果存在未解决的问题，请在总结中说明。"
                )
                summary_message = Message(role=Role.USER, content=summary_prompt)
                # 防御层: summary 轮明确禁止建图，但 LLM 可能不听话。
                # 一旦它仍然尝试 create_task_graph，立即吞掉该次调用并打断
                # 内层循环 — 拒绝执行图、避免重新进入"狂刷 DAG"状态。
                summary_iter = self._agent.process_message(summary_message, session_id)
                try:
                    async for event in summary_iter:
                        ev_type = event.event_type.value
                        if (ev_type == "tool_call_start"
                                and event.data.get("name") == "create_task_graph"):
                            logger.warning(
                                "Summary 轮违规调用 create_task_graph，已忽略并提前终止"
                            )
                            yield Event.text_done(
                                "(系统) 已达到最大任务图轮数，本轮拒绝再创建任务图。",
                                session_id,
                            )
                            last_text_emitted = True
                            break
                        if ev_type == "text_done":
                            text = event.data.get("text", "") or ""
                            if text.strip():
                                last_text_emitted = True
                        yield event
                finally:
                    await summary_iter.aclose()

            # UX 契约兜底：master 流程退出前，若 LLM 始终没产出实质性文本，
            # 系统必须给用户一段"做了什么/为什么没成"的总结。否则用户面对
            # 空白屏幕只能从日志推断状态
            if not last_text_emitted:
                fallback = self._build_fallback_summary(
                    user_input=original_user_input,
                    graph_results=all_graph_results,
                )
                if fallback:
                    yield Event.text_delta(fallback, session_id)
                    yield Event.text_done(fallback, session_id)

        except CircuitBreakerError as e:
            logger.error("熔断触发: %s", e)
            yield Event.error(f"任务执行中断: {str(e)}", session_id)

        except Exception as e:
            logger.error("Master Agent 处理失败: %s", e)
            yield Event.error(f"处理失败: {str(e)}", session_id)

        finally:
            yield Event.done(session_id)

    def _extract_graph_id(self, result_str: str) -> str | None:
        """从工具结果中提取 graph_id"""
        import json
        try:
            data = json.loads(result_str)
            return data.get("graph_id")
        except json.JSONDecodeError:
            import re
            match = re.search(r'"graph_id":\s*"([^"]+)"', result_str)
            if match:
                return match.group(1)
        return None

    async def _wait_for_graph_result(
        self,
        graph_id: str,
        master_node: AgentNode,
        timeout_seconds: int = 300,
    ) -> TaskGraphResult | None:
        """等待任务图执行完成

        通过 Orchestrator 调度执行子任务图，并等待结果返回。
        """
        # 优先从 Orchestrator 的私有 store 取 request，回退到模块级默认 store
        # 以保证尚未升级的调用路径仍能工作
        request = self._orchestrator.pending_graph_store.get(graph_id)
        if request is None:
            from miniclaw.tools.builtin.create_task_graph import get_pending_graph
            request = get_pending_graph(graph_id)
        if request is None:
            logger.warning("任务图请求不存在: %s", graph_id)
            return None

        try:
            graph_result = await self._orchestrator.execute_task_graph(
                request, master_node, timeout_seconds=timeout_seconds,
                graph_id=graph_id,
            )
            # DAG 执行完后显式清理 store，避免长期积压
            await self._orchestrator.pending_graph_store.pop(graph_id)

            logger.info(
                "任务图 %s 执行完成: status=%s, completed=%d, failed=%d",
                graph_id,
                graph_result.status if graph_result else "unknown",
                len(graph_result.completed_tasks) if graph_result else 0,
                len(graph_result.failed_tasks) if graph_result else 0,
            )

            return graph_result

        except Exception as e:
            logger.error("任务图执行失败: %s", e)
            return TaskGraphResult(
                graph_id=graph_id,
                total_tasks=len(request.tasks),
                max_depth=0,
                execution_order=[],
                dynamic_roles=[],
                status="failed",
                failed_tasks=[t.id for t in request.tasks],
            )

    def _build_duplicate_dag_message(
        self,
        prior_round: int,
        current_round: int,
    ) -> Message:
        """当 Master 重派了与之前等价的 DAG 时，注入一段强纠偏 prompt
        强信号 + 具体备选项，避免 LLM 又交出第三张同样的图
        """
        content = (
            f"## ⚠️ 系统拒绝执行：重复 DAG\n\n"
            f"你在第 {current_round} 轮提交的任务图与第 {prior_round} 轮"
            f"在指纹层面**完全等价**（仅 task.id 或排版不同），系统已拒绝执行。\n\n"
            "重复同一张失败 DAG 不会得到不同的结果。请在下一步选择以下之一：\n\n"
            "**A. 真正改写策略**（如果你认为还能成）：\n"
            "  - 把失败任务的 instruction 写得更小、更具体、更可验证\n"
            "  - 换一个角色（CoderAgent 反复失败时考虑 PlannerAgent 先拆解）\n"
            "  - 拆分依赖结构（一个大任务 → 多个独立小任务）\n"
            "  - 用 `custom_role_prompt` 定义一个更窄、更明确的角色\n\n"
            "**B. 如实告诉用户失败**（推荐，如果连续失败已很明显）：\n"
            "  - 不要再调用 `create_task_graph`\n"
            "  - 直接用文本回复用户：「<原始任务> 没能完成，原因是 ...」\n"
            "  - 说明你已尝试的策略和卡在哪里，让用户决定要不要换工具/手动介入\n\n"
            "**禁止**：\n"
            "  - 改个 task.id 又交回类似的 DAG（系统会再次拒绝）\n"
            "  - 自己 write_file / terminal 直接做用户要的事（违反硬约束）\n"
            "  - 沉默退出（用户必须看到一段说明）"
        )
        return Message(role=Role.USER, content=content)

    def _build_fallback_summary(
        self,
        user_input: str,
        graph_results: list[TaskGraphResult],
    ) -> str:
        """LLM 未给用户任何文本时，系统兜底输出

        此函数永远不会返回 None — 即使没有任何 graph 也给出一行"已收到请求"
        的 placeholder，确保用户屏幕不空白
        """
        if not graph_results:
            return (
                "(系统) 已收到你的请求，但本次会话没有进入任务调度流程。"
                "如果觉得回复不完整，请重新提问或换种描述方式。"
            )

        total_completed = sum(len(r.completed_tasks) for r in graph_results)
        total_failed = sum(len(r.failed_tasks) for r in graph_results)
        rounds = len(graph_results)

        # 收集失败原因分类
        category_count: dict[str, int] = {}
        for r in graph_results:
            for cat in r.task_failure_categories.values():
                category_count[cat.value] = category_count.get(cat.value, 0) + 1

        # 收集所有"系统观察到的真实落地文件"
        # 不用 agent 自报的 files_changed 可能撒谎；用 outcome 的 actual_files_landed 才是 git 视角的事实
        produced_files: set[str] = set()
        discrepancy_count = 0
        for r in graph_results:
            for task_id, outcome in r.task_outcomes.items():
                if isinstance(outcome, TaskOutcome):
                    if outcome.system_observation.succeeded:
                        produced_files.update(outcome.system_observation.actual_files_landed)
                    if outcome.discrepancy:
                        discrepancy_count += 1

        lines = [
            "## 任务执行总结（系统兜底）",
            "",
            (
                f"针对你的请求"
                f"「{user_input.strip().splitlines()[0][:80] if user_input.strip() else '(空)'}」"
            ),
            "",
            f"- 调度了 {rounds} 轮任务图",
            f"- 子任务：{total_completed} 成功 / {total_failed} 失败",
        ]
        if produced_files:
            file_list = ", ".join(sorted(produced_files)[:10])
            if len(produced_files) > 10:
                file_list += f" ... 共 {len(produced_files)} 个"
            lines.append(f"- 实际落入主干的文件：{file_list}")
        elif total_completed > 0:
            lines.append("- 实际落入主干的文件：**（无）** — agent 报告完成但系统未观察到产物")
        if discrepancy_count > 0:
            lines.append(
                f"- ⚠️ **{discrepancy_count} 个任务存在 agent 自报与系统观察的差异** —— "
                "通常意味着 agent 声称做完了某事但产物未真正合入仓库。"
            )
        if category_count:
            cat_brief = ", ".join(
                f"{name}×{count}" for name, count in sorted(category_count.items())
            )
            lines.append(f"- 失败原因分类：{cat_brief}")
        lines.append("")
        if total_failed > 0 and total_completed == 0:
            lines.append(
                "**结论**：本次任务**没有完成**。多次重派均未通过验证或在子 Agent 里走入死循环，"
                "系统已停止重试避免浪费资源。"
            )
            lines.append("")
            lines.append("建议：")
            if "validation_unmet" in category_count:
                lines.append(
                    "- 验证门禁反复未满足。请检查 worktree 内是否有 lint/test 工具，"
                    "或者把要做的事描述得更具体（避免子 Agent 走偏）。"
                )
            if "agent_max_iterations" in category_count:
                lines.append(
                    "- 子 Agent 走入死循环。通常源于指令含混或工具表里有冲突项；"
                    "可以试着把任务拆得更小再发起。"
                )
            if "worktree_creation" in category_count:
                lines.append(
                    "- Worktree 创建失败。请检查仓库是不是 Git repo、是否有未解的合并状态。"
                )
        elif total_failed > 0:
            lines.append(
                f"**结论**：部分完成 — {total_completed} 个子任务的产出已合入工作分支，"
                f"另有 {total_failed} 个子任务失败但已知风险已记录。"
            )
        else:
            lines.append("**结论**：所有子任务都成功了。")
        return "\n".join(lines)

    @staticmethod
    def _is_cleanup_dag(original_request: Any | None) -> bool:
        """判断 DAG 是否为 cleanup-only DAG（所有 task 都派给 CleanupAgent）。

        依赖项目硬约束：cleanup 必须派 CleanupAgent，且 cleanup DAG 不能与
        build 任务混跑（见 master_prompt §过程产物清理工作流）。所以"全是
        CleanupAgent"是足够可靠的反向识别口径，无需新加 schema 字段。
        """
        if original_request is None:
            return False
        tasks = getattr(original_request, "tasks", None) or []
        if not tasks:
            return False
        for t in tasks:
            role_name = (getattr(t, "role", "") or "").strip()
            # 容忍大小写差异同时接受 'CleanupAgent' / 'Cleanup'
            if role_name.lower() not in {"cleanupagent", "cleanup"}:
                return False
        return True

    @staticmethod
    def _collect_landed_files(result: TaskGraphResult) -> list[str]:
        """汇总 DAG 跑完后**实际进入主干**的全部文件。

        范围：所有 task 的 outcome.system_observation.actual_files_landed，
        只要 workspace_final_state ∈ (MERGED, MERGED_VIA_DESCENDANT) 就计入。
        中间任务（MERGED_VIA_DESCENDANT）的文件也要列——这是用户视角"主干
        多了什么"的真实清单，master 据此做交付审视/cleanup 决策。
        """
        landed: set[str] = set()
        for outcome in result.task_outcomes.values():
            if not isinstance(outcome, TaskOutcome):
                continue
            obs = outcome.system_observation
            if obs.workspace_final_state in (
                WorkspaceFinalState.MERGED,
                WorkspaceFinalState.MERGED_VIA_DESCENDANT,
            ):
                landed.update(obs.actual_files_landed)
        return sorted(landed)

    def _build_result_message(
        self,
        result: TaskGraphResult | None,
        original_request: Any | None = None,
    ) -> Message:
        """构建结果消息（注入回对话供 Agent 决策）

        失败任务会被显式标红，并附带原始 instruction（截断），让 Master 能
        基于具体上下文做精准重派——只针对失败任务发新 task_graph，**不要重新
        派发已成功任务**（避免覆盖现有产物）。

        成功路径按 build / cleanup 区分：
          - build DAG 成功 → 引导 master 走交付审视 + [CLEANUP_DECISION]
          - cleanup DAG 成功 → 禁止再建图，立即总结给用户

        禁止 Master 自己直接写代码（这是 Master Agent 的硬约束）。
        """
        # result 为 None 时（_wait_for_graph_result 找不到 request 或调度异常返回 None）给 master 一段明确的"调度失败"提示，避免崩
        if result is None:
            return Message(
                role=Role.USER,
                content=(
                    "## 任务图调度失败\n\n"
                    "刚才创建的任务图请求已丢失或调度未返回结果。"
                    "请直接告诉用户「系统出现内部错误，未能执行你的任务图」"
                    "并停止建图。"
                ),
            )

        # 构建 task_id 原始 instruction 映射，便于在失败列表里附带摘要
        instruction_map: dict[str, str] = {}
        role_map: dict[str, str] = {}
        if original_request is not None:
            for t in getattr(original_request, "tasks", []):
                instruction_map[t.id] = (getattr(t, "instruction", "") or "")
                role_map[t.id] = (getattr(t, "role", "") or "")

        lines = [
            "## 任务图执行结果",
            "",
            f"- **状态**: {result.status}",
            f"- **总任务数**: {result.total_tasks}",
            f"- **已完成**: {len(result.completed_tasks)}",
            f"- **失败**: {len(result.failed_tasks)}",
            "",
        ]

        if result.completed_tasks:
            lines.append("### 已完成任务（**禁止重新派发**）")
            for task_id in result.completed_tasks:
                # 优先用系统观察的 actual_files_landed 文件清单，而不是 agent 自报的 files_changed
                outcome = result.task_outcomes.get(task_id)
                files_str = ""
                discrepancy_str = ""
                if isinstance(outcome, TaskOutcome):
                    actual = outcome.system_observation.actual_files_landed
                    if actual:
                        files_preview = ", ".join(actual[:5])
                        if len(actual) > 5:
                            files_preview += f" ... 共 {len(actual)} 个"
                        files_str = f" — 落地文件: {files_preview}"
                    d = outcome.discrepancy
                    if d:
                        discrepancy_str = f"\n  - ⚠️ {d}"
                lines.append(f"- ✅ `{task_id}` — 已成功，产物已合入{files_str}{discrepancy_str}")
            lines.append("")

        if result.failed_tasks:
            lines.append("### 失败任务清单（重派时只针对这些 ID）")
            for task_id in result.failed_tasks:
                error_detail = result.task_errors.get(task_id, "未知错误")
                category = result.task_failure_categories.get(task_id)
                role = role_map.get(task_id, "")
                instruction = instruction_map.get(task_id, "")
                if instruction:
                    instr_preview = instruction.strip().splitlines()[0][:120]
                    if len(instruction) > 120:
                        instr_preview += "..."
                else:
                    instr_preview = "(原始指令未保留)"
                role_tag = f" [role={role}]" if role else ""
                category_tag = f" [category={category.value}]" if category else ""

                # 关键：把 outcome 的 discrepancy 显式标出 ——
                # 当 agent 自报 success 但系统判失败时，这是最重要的诚实信号
                outcome = result.task_outcomes.get(task_id)
                discrepancy_block = ""
                if isinstance(outcome, TaskOutcome):
                    d = outcome.discrepancy
                    if d:
                        discrepancy_block = (
                            f"\n  - 🔴 **诚实性提醒**: {d}"
                        )
                    obs = outcome.system_observation
                    if obs.workspace_final_state != WorkspaceFinalState.NOT_APPLICABLE:
                        discrepancy_block += (
                            f"\n  - Worktree 终态: `{obs.workspace_final_state.value}`"
                        )

                    # 工具调用统计 —— master 据此分辨 timeout 是 A 型（真不够）
                    # 还是 B 型（agent 死循环没 submit）。展示 top 5 + 是否调过 submit
                    if obs.tool_call_counts:
                        counts = obs.tool_call_counts
                        sorted_counts = sorted(
                            counts.items(), key=lambda x: -x[1],
                        )[:5]
                        counts_str = ", ".join(
                            f"{name}×{n}" for name, n in sorted_counts
                        )
                        submit_n = counts.get("submit_task_result", 0)
                        if submit_n == 0:
                            submit_marker = " ⚠️ **submit_task_result 调用次数=0（agent 未退出）**"
                        else:
                            submit_marker = ""
                        discrepancy_block += (
                            f"\n  - 工具调用: {counts_str}{submit_marker}"
                        )

                lines.append(
                    f"- ❌ `{task_id}`{role_tag}{category_tag}\n"
                    f"  - 原指令: {instr_preview}\n"
                    f"  - 失败原因: {error_detail[:200]}"
                    f"{discrepancy_block}"
                )

                # 针对失败分类提供具体的"下一步建议"，避免 Master 又重派同样的 DAG
                hint = self._hint_for_failure_category(category)
                if hint:
                    lines.append(f"  - 建议: {hint}")
            lines.append("")
            lines.append("### ⚠️ 你必须二选一")
            lines.append(
                "**A. 精准重派**：调用 `create_task_graph` 仅针对**上面列出的失败 task_id**生成新任务。"
                "**关键**：必须根据每个任务的 `category` 和「建议」改写策略，"
                "重新生成「等价 DAG」（仅改 task.id 或换措辞）会被系统拒绝执行。"
                "**绝对不要**重新派发任何「已完成任务」——它们的产物已经合入仓库，重派会冲突或回退工作。"
            )
            lines.append(
                "**B. 如实汇报**：如果判断无法继续推进，直接告诉用户"
                "「<原始任务> 没有完成，原因是 ...」并停止。"
            )
            lines.append("")
            lines.append(
                "**禁令**：不要自己 `write_file` 写代码、不要改主题去做别的、"
                "不要把局部产物当成功汇报。"
            )
        else:
            is_cleanup = self._is_cleanup_dag(original_request)

            if is_cleanup:
                # cleanup DAG 成功 → 立即终止建图循环
                lines.append("### Cleanup 已完成 — **立即总结，禁止再建图**")
                lines.append("")
                lines.append(
                    "刚才的 cleanup DAG 已经把 instruction 里列出的过程产物"
                    "从主干删除。**现在你必须直接用文本回复用户最终交付总结**。"
                )
                lines.append("")
                lines.append("**硬性禁令**（违反 = 进入 build → cleanup → 又 build 死循环）：")
                lines.append("- ❌ **禁止**再调用 `create_task_graph`（不管是再 build 还是再 cleanup）")
                lines.append("- ❌ **禁止**再写 `[CLEANUP_DECISION]` 标记（cleanup 已结束）")
                lines.append("- ❌ **禁止**自己 `write_file` / `edit_file` / `terminal`")
                lines.append("")
                lines.append("**应该做的**：用 `## 终交付` + `## 过程产物已清理` 两段格式直接回复用户。")
            else:
                # build DAG 成功 → 走交付审视 + cleanup 决策
                landed_files = self._collect_landed_files(result)
                lines.append("### 下一步：交付审视 + cleanup 决策（必走）")
                lines.append("")

                # 给 master 一个统一的"主干增量"视图，避免它逐 task 自己拼
                if landed_files:
                    lines.append("**本次 build DAG 实际合入主干的全部文件**：")
                    for f in landed_files:
                        lines.append(f"- `{f}`")
                    lines.append("")
                else:
                    lines.append(
                        "**本次 build DAG 没有文件合入主干**（可能是纯调研类 DAG）。"
                    )
                    lines.append("")

                lines.append(
                    "所有子任务成功执行。**现在你必须做交付审视**："
                )
                lines.append("")
                lines.append(
                    "1. 对照上面的**落地文件清单**和用户**原始请求**，逐文件归类："
                )
                lines.append("   - **终交付**：用户真正想要的产物（类型/数量/命名都对得上）")
                lines.append("   - **过程产物**：做事过程产生但用户不关心的中间文件")
                lines.append("")
                lines.append("2. 在 `<thinking>` 块里**显式输出**清理决策（格式严格遵守）：")
                lines.append("")
                lines.append("   `[CLEANUP_DECISION] needed=<yes|no>  reason=<一句话理由>`")
                lines.append("")
                lines.append("3. 根据决策走对应分支（**只有这两条路，不允许第三方案**）：")
                lines.append(
                    "   - `needed=yes` → **本轮**调用 `create_task_graph` 派一个"
                    "**single-task cleanup-only DAG**（role=`CleanupAgent`，instruction 显式"
                    "列出要删的文件）。详见 system prompt §过程产物清理工作流。"
                )
                lines.append(
                    "   - `needed=no` → 直接给用户总结，**禁止**再创建任何任务图。"
                )
                lines.append("")
                lines.append("**判断口诀**：")
                lines.append(
                    "- 用户最终消费的是**人读的文档/报告/数据**且落地里有 `.py`/`.json`/"
                    "`.log` 等中间产物 → **needed=yes**"
                )
                lines.append("- 用户最终消费的是**机器执行的代码/工具** → **needed=no**")
                lines.append("- 用户**显式**说过『也保留 X』 → **needed=no**")
                lines.append("")
                lines.append("**第三方案禁令**（违反 = 任务失败）：")
                lines.append(
                    "- ❌ **禁止**自创『保留过程产物 + 让用户手动删除』这种第三方案。"
                    "用户没主动要保留的中间文件，要么由 CleanupAgent 删掉，要么承担"
                    "脏出后果——**不允许把决策推给用户**。"
                )
                lines.append(
                    "- ❌ **禁止**重新派发已完成任务（产物已合入主干，重派会冲突或回退工作）。"
                )
                lines.append(
                    "- ❌ **禁止**自己 `write_file` / `terminal` 删文件"
                    "（你的工具白名单不含这些）。"
                )

        return Message(
            role=Role.USER,
            content="\n".join(lines),
        )

    def _hint_for_failure_category(
        self,
        category: FailureCategory | None,
    ) -> str:
        """针对失败分类给 Master 一段精炼的纠偏提示。
        这些 hint 是"如何改 DAG 让它有可能成功"的具体动作，不是描述。
        """
        if category is None:
            return ""
        if category == FailureCategory.VALIDATION_UNMET:
            return (
                "子 Agent 未通过验证闭环（lint/tests 没成功调用即提交）。"
                "重派时把 instruction 写成「先写 X，再调 run_linter 验证，最后 submit」，"
                "或者换 TesterAgent 单独跑测试。"
            )
        if category == FailureCategory.AGENT_MAX_ITERATIONS:
            return (
                "子 Agent 走入死循环（达到 max_iterations）。"
                "通常源于 instruction 太宽泛或 worktree 里有干扰文件。"
                "拆得更细：每个子任务只负责一个动作（写 1 个文件 / 跑 1 个工具）。"
            )
        if category == FailureCategory.TIMEOUT:
            return (
                "子 Agent 超时。**先看工具调用统计**判断是哪种 timeout：\n"
                "  - **类型 A：任务真的需要更多时间**（工具调用次数合理、有真实进展）"
                "→ 调高 timeout_ms 或拆小任务\n"
                "  - **类型 B：agent 走偏了不肯 submit**（同一类工具反复调 10+ 次、"
                "submit_task_result 调用次数=0）→ **再调 timeout 也无用**，必须改派单："
                "(1) 换更克制的角色 (2) 把 instruction 改成『最多调 N 次 X 工具，"
                "无果立即 submit partial_success』(3) 或如实告诉用户『信息源不可用』停止重试\n"
                "**强烈警告**：如果上一轮已经调过 timeout 还是 timeout，几乎确定是类型 B，"
                "盲目继续调高 timeout_ms 会再次失败、浪费 round budget。"
            )
        if category == FailureCategory.CIRCUIT_BREAKER:
            return (
                "熔断器跳闸：同一种工具失败反复出现。重派时换工具或换角色，"
                "不要再走相同路径。"
            )
        if category == FailureCategory.DEPENDENCY_FAILED:
            return (
                "上游依赖失败导致本任务被短路。先修上游，或把本任务从 DAG 中独立出来。"
            )
        if category == FailureCategory.WORKTREE_CREATION:
            return (
                "Git worktree 创建失败（分支冲突 / 仓库状态异常）。"
                "建议直接告诉用户检查 git 状态，不要再重派。"
            )
        if category == FailureCategory.WORKSPACE_CORRUPTED:
            return (
                "Worktree 在执行中状态被毁（内存/磁盘/git 视图漂移）。"
                "这是基础设施级故障，不要重派 —— 直接告诉用户系统出问题，"
                "建议清空 .agents/worktrees 目录后重启。"
            )
        if category == FailureCategory.SANDBOX_VIOLATION:
            return (
                "Sandbox 被破坏：agent 试图在已不存在的 worktree 根写文件。"
                "源头通常是上游 worktree 创建失败，先修上游。"
            )
        if category == FailureCategory.COMMIT_FAILED:
            return (
                "Git commit 失败（pre-commit hook 拒绝 / 索引被占用）。"
                "请用户检查仓库 hook 配置或者其他进程占用情况。"
            )
        if category == FailureCategory.PROVIDER_ERROR:
            return (
                "AI 服务调用失败（外部依赖问题）。可以原样重派一次试试，"
                "若仍失败请告知用户。"
            )
        return "失败分类未知，请仔细看错误原文再决定。"

    async def get_status(self) -> dict[str, Any]:
        """获取状态"""
        return {
            "type": "master_agent",
            "orchestrator": await self._orchestrator.get_status(),
        }


def create_master_agent_config() -> AgentConfig:
    """创建 Master Agent 配置"""
    return AgentConfig(
        system_prompt=MASTER_AGENT_PROMPT,
        max_iterations=50,  # Master 需要更多迭代
        max_context_tokens=16000,  # 更大的上下文
        compaction_threshold=0.9,
    )


# 导出
__all__ = [
    "MasterAgent",
    "create_master_agent_config",
]