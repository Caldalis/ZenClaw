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
from miniclaw.types.task_graph import TaskGraphResult
from miniclaw.types.turn_snapshot import AgentNode
from miniclaw.utils.logging import get_logger

logger = get_logger(__name__)


class MasterAgent:
    """Master Agent — 主调度器

    与普通 Agent 的区别:
      1. 使用专门的 Master 系统提示词
      2. 内置 create_task_graph 工具
      3. 与 SubagentOrchestrator 集成
      4. 状态机循环：Agent 自主决定每轮行动

    执行流程（状态机循环）:
      ANALYZE → (create_task_graph?) → EXECUTE → REVIEW → ANALYZE → ... → DONE

      每轮:
        1. 运行 Agent.process_message()
        2. 如果检测到 create_task_graph → 执行 DAG，注入结果，继续循环
        3. 如果 Agent 直接回复（未建图）→ 结束

      max_task_graph_rounds 防止无限建图循环。
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

        try:
            while round_num < max_rounds:
                round_num += 1
                logger.info("Master Agent 第 %d/%d 轮", round_num, max_rounds)

                # 运行 Agent — 它可能创建 task_graph 或直接回复
                graph_id = None
                has_task_graph = False

                async for event in self._agent.process_message(current_message, session_id):
                    # 检测 create_task_graph 调用
                    if event.event_type.value == "tool_call_start":
                        if event.data.get("name") == "create_task_graph":
                            has_task_graph = True
                            logger.info("检测到 create_task_graph 调用 (round %d)", round_num)

                    if event.event_type.value == "tool_call_result":
                        result_str = event.data.get("result", "")
                        graph_id = self._extract_graph_id(result_str)

                    yield event

                # Agent 未创建任务图 → 直接回复 → 结束
                if not has_task_graph or not graph_id:
                    logger.info("Master Agent 未创建任务图，结束循环 (round %d)", round_num)
                    break

                # 执行任务图 → 注入结果 → 下一轮 Agent 自主决策
                logger.info("任务图 %s 已创建，执行并注入结果 (round %d)", graph_id, round_num)
                graph_result = await self._wait_for_graph_result(graph_id, master_node)
                current_message = self._build_result_message(graph_result)
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
                async for event in self._agent.process_message(summary_message, session_id):
                    yield event

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
        from miniclaw.tools.builtin.create_task_graph import get_pending_graph

        # 获取任务图请求
        request = get_pending_graph(graph_id)
        if request is None:
            logger.warning("任务图请求不存在: %s", graph_id)
            return None

        try:
            graph_result = await self._orchestrator.execute_task_graph(
                request, master_node, timeout_seconds=timeout_seconds,
                graph_id=graph_id,
            )

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

    def _build_result_message(self, result: TaskGraphResult) -> Message:
        """构建结果消息（注入回对话供 Agent 决策）

        合入了 Orchestrator 的格式化逻辑，展示任务完成/失败详情。
        """
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
            lines.append("### 已完成任务")
            for task_id in result.completed_tasks:
                lines.append(f"- {task_id}")
            lines.append("")

        if result.failed_tasks:
            lines.append("### 失败任务")
            for task_id in result.failed_tasks:
                error_detail = result.task_errors.get(task_id, "未知错误")
                lines.append(f"- {task_id}: {error_detail}")
            lines.append("")

        lines.append("请根据以上结果决定下一步行动：")
        lines.append("- 如果所有任务成功，直接总结最终结果")
        lines.append("- 如果部分任务失败，可以创建新的任务图来处理失败部分")
        lines.append("- 如果问题无法通过任务图解决，直接说明情况")

        return Message(
            role=Role.USER,
            content="\n".join(lines),
        )

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