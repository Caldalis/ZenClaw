"""
工具调用执行器

职责:
  1. 接收 AI 返回的 ToolCall 列表
  2. 从 SkillRegistry 中查找对应的 Skill
  3. 执行 Skill 并收集结果
  4. 将结果封装为 ToolResult 消息

这是 Agent 工具调用循环的关键组件:
  AI 决定要调用哪些工具 → ToolExecutor 执行 → 结果返回给 AI
"""

from __future__ import annotations

from typing import Any

from miniclaw.memory.context_guard import ContextGuard
from miniclaw.tools.registry import ToolRegistry
from miniclaw.types.enums import Role
from miniclaw.types.events import Event
from miniclaw.types.messages import Message, ToolCall, ToolResult
from miniclaw.utils.logging import get_logger

logger = get_logger(__name__)


class ToolExecutor:
    """工具调用执行器"""

    def __init__(
        self,
        tool_registry: ToolRegistry,
        tool_result_max_bytes: int = 102400,
        validation_gatekeeper: Any = None,
        tool_circuit_breaker: Any = None,
    ):
        self._registry = tool_registry
        self._max_bytes = tool_result_max_bytes
        self._gatekeeper = validation_gatekeeper
        # 可选：本 Agent 局部的 CircuitBreaker，用于检测"工具结构化失败重复"
        # （例如 run_linter 反复返回 status=error 而不抛异常）。
        self._tool_breaker = tool_circuit_breaker

    async def execute(
        self,
        tool_calls: list[ToolCall],
        session_id: str | None = None,
    ) -> tuple[list[Message], list[Event]]:
        """执行一组工具调用

        Args:
            tool_calls: AI 请求的工具调用列表
            session_id: 关联的会话 ID

        Returns:
            (tool_messages, events)
            - tool_messages: 工具结果消息列表（将加入对话历史）
            - events: 过程中产生的事件（用于流式展示）
        """
        messages: list[Message] = []
        events: list[Event] = []

        for tc in tool_calls:
            logger.info("执行工具调用: %s(%s)", tc.name, tc.arguments)

            # 发出工具调用开始事件
            events.append(Event.tool_call_start(tc.name, tc.arguments, session_id))

            # 查找技能
            skill = self._registry.get(tc.name)
            if skill is None:
                result_text = f"未知工具: {tc.name}"
                is_error = True
                logger.warning("未知工具: %s", tc.name)
            else:
                # 执行技能
                try:
                    result_text = await skill.execute(**tc.arguments)
                    is_error = False
                except Exception as e:
                    result_text = f"工具执行错误: {e}"
                    is_error = True
                    logger.error("工具 %s 执行失败: %s", tc.name, e, exc_info=True)

                # 记录验证工具结果到 Gatekeeper
                if not is_error and self._gatekeeper and tc.name in ("run_linter", "run_tests"):
                    self._record_validation(tc.name, result_text)

            # 工具输出截断（仅截断成功结果，错误信息通常较短）
            if not is_error:
                result_text = ContextGuard.truncate_tool_result(result_text, self._max_bytes)

            # 构建工具结果消息
            tool_result = ToolResult(
                tool_call_id=tc.id,
                content=result_text,
                is_error=is_error,
            )
            messages.append(Message(
                role=Role.TOOL,
                content=result_text,
                tool_result=tool_result,
                session_id=session_id,
            ))

            # 发出工具调用结果事件
            events.append(Event.tool_call_result(tc.name, result_text, session_id))
            logger.info("工具 %s 结果: %s", tc.name, result_text[:100])

        return messages, events

    def _record_validation(self, tool_name: str, result_text: str) -> None:
        """将验证工具的执行结果记录到 Gatekeeper"""
        import json
        from miniclaw.agents.critic.validation_tools import ValidationResult, ValidationStatus

        try:
            data = json.loads(result_text)
        except json.JSONDecodeError:
            logger.warning("验证结果 JSON 解析失败: %s", tool_name)
            return

        status_str = data.get("status", "error")
        status = ValidationStatus.PASSED if status_str == "passed" else (
            ValidationStatus.FAILED if status_str == "failed" else ValidationStatus.ERROR
        )

        result = ValidationResult(
            tool_name=tool_name,
            status=status,
            message=data.get("message", ""),
            errors=data.get("errors", []),
            warnings=data.get("warnings", []),
        )

        self._gatekeeper.record_validation(tool_name, result)
        logger.info("验证结果已记录: tool=%s, status=%s", tool_name, status.value)

        # 把"结构化失败"喂给熔断器，让它在 N 次相同失败后跳闸
        # （否则 agent 看到 ERROR 后会反复试，没有任何机制能打断）
        if self._tool_breaker and status in (ValidationStatus.ERROR, ValidationStatus.FAILED):
            self._tool_breaker.record_tool_failure(
                tool_name=tool_name,
                status=status.value,
                message=data.get("message", ""),
            )
