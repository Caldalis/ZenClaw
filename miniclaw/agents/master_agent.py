"""
Master Agent — 多智能体架构的主调度器

Master Agent 的职责:
  1. 分析用户请求，判断是否需要分解任务
  2. 使用 create_task_graph 工具构建 DAG
  3. 监控子任务执行状态
  4. 根据结果做出后续决策

Master Agent 不直接执行具体工作，只做调度和决策。
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator, Any

from miniclaw.agents.agent import Agent
from miniclaw.agents.prompts import MASTER_AGENT_PROMPT
from miniclaw.agents.subagent_orchestrator import SubagentOrchestrator
from miniclaw.config.settings import AgentConfig, Settings
from miniclaw.dispatcher.subagent_registry import SubagentRegistry
from miniclaw.memory.base import MemoryStore
from miniclaw.sessions.manager import SessionManager
from miniclaw.tools.registry import ToolRegistry
from miniclaw.types.enums import Role
from miniclaw.types.events import Event
from miniclaw.types.messages import Message
from miniclaw.types.task_graph import TaskGraphResult
from miniclaw.utils.logging import get_logger

logger = get_logger(__name__)


class MasterAgent:
    """Master Agent — 主调度器

    与普通 Agent 的区别:
      1. 使用专门的 Master 系统提示词
      2. 内置 create_task_graph 工具
      3. 与 SubagentOrchestrator 集成
      4. 自动处理任务图执行结果
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
        """处理用户消息

        流程:
          1. 分析任务复杂度
          2. 决定是否分解任务
          3. 如果分解，调度执行
          4. 返回最终结果
        """
        logger.info("Master Agent 开始处理消息: session=%s", session_id)

        yield Event.thinking(session_id)

        try:
            # 第一轮：让 Agent 分析和决策
            accumulated_content = ""
            graph_id = None
            has_task_graph = False

            async for event in self._agent.process_message(user_message, session_id):
                # 检测 create_task_graph 调用
                if event.event_type.value == "tool_call_start":
                    if event.data.get("name") == "create_task_graph":
                        has_task_graph = True
                        logger.info("检测到 create_task_graph 调用")

                if event.event_type.value == "tool_call_result":
                    # 解析 graph_id
                    result_str = event.data.get("result", "")
                    graph_id = self._extract_graph_id(result_str)

                if event.event_type.value == "text_delta":
                    accumulated_content += event.data.get("text", "")

                yield event

            # 如果创建了任务图，等待执行并返回结果
            if has_task_graph and graph_id:
                logger.info("任务图已创建: %s，等待执行结果", graph_id)

                # 获取执行结果
                graph_result = await self._wait_for_graph_result(graph_id)

                if graph_result:
                    # 将结果注入回对话
                    result_message = self._build_result_message(graph_result)

                    # 第二轮：让 Agent 总结结果
                    async for event in self._agent.process_message(result_message, session_id):
                        yield event

        except Exception as e:
            logger.error("Master Agent 处理失败: %s", e)
            yield Event.error(f"处理失败: {str(e)}", session_id)

        finally:
            yield Event.done(session_id)

    def _extract_graph_id(self, result_str: str) -> str | None:
        """从工具结果中提取 graph_id"""
        import json
        try:
            # 尝试解析 JSON
            data = json.loads(result_str)
            return data.get("graph_id")
        except json.JSONDecodeError:
            # 尝试正则匹配
            import re
            match = re.search(r'"graph_id":\s*"([^"]+)"', result_str)
            if match:
                return match.group(1)
        return None

    async def _wait_for_graph_result(
        self,
        graph_id: str,
        timeout_seconds: int = 300,
    ) -> TaskGraphResult | None:
        """等待任务图执行完成"""
        from miniclaw.tools.builtin.create_task_graph import get_pending_graph

        # 获取任务图请求
        request = get_pending_graph(graph_id)
        if request is None:
            logger.warning("任务图请求不存在: %s", graph_id)
            return None

        # 使用 orchestrator 的 scheduler 执行
        # 这里简化处理，实际应该更复杂
        # TODO: 完善任务图执行逻辑

        return TaskGraphResult(
            graph_id=graph_id,
            total_tasks=len(request.tasks),
            max_depth=0,
            execution_order=[],
            dynamic_roles=[],
        )

    def _build_result_message(self, result: TaskGraphResult) -> Message:
        """构建结果消息"""
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
                lines.append(f"- ✅ {task_id}")
            lines.append("")

        if result.failed_tasks:
            lines.append("### 失败任务")
            for task_id in result.failed_tasks:
                lines.append(f"- ❌ {task_id}")
            lines.append("")

        lines.append("请根据以上结果总结完成情况，或决定是否需要调整策略。")

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