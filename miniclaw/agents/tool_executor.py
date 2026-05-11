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
        critic_injector: Any = None,
    ):
        self._registry = tool_registry
        self._max_bytes = tool_result_max_bytes
        self._gatekeeper = validation_gatekeeper
        # 工具结构化失败的两个汇入点：breaker 决定是否提前终止 ReAct，
        # critic_injector 决定下一轮 LLM 调用要附带什么警示。两者并行喂数据。
        self._tool_breaker = tool_circuit_breaker
        self._critic_injector = critic_injector

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
                    # 真实异常路径同时喂给 breaker + critic，与"结构化失败"路径
                    # 并行处理（结构化失败走 _record_validation 内部）。
                    if self._tool_breaker is not None:
                        try:
                            self._tool_breaker.record_failure(e)
                        except Exception as breaker_err:
                            logger.debug("breaker 记录异常失败（吞掉）: %s", breaker_err)
                    if self._critic_injector is not None:
                        try:
                            self._critic_injector.record_failure(
                                tool_name=tc.name,
                                tool_arguments=tc.arguments,
                                error=e,
                            )
                        except Exception as critic_err:
                            logger.debug("critic 记录异常失败（吞掉）: %s", critic_err)

                # 记录验证工具结果到 Gatekeeper
                if not is_error and self._gatekeeper and tc.name in ("run_linter", "run_tests"):
                    self._record_validation(tc.name, result_text)

                # submit_task_result 被门禁拒绝时，本质上和"工具反复返回 ERROR"等价：
                # 字符串里没有抛异常，所以走不到 record_failure 路径，但确实是
                # 一个停滞模式。喂给 breaker，让它在 N 次相同拒绝后跳闸 —
                # 没这一刀，agent 会硬撞 max_iterations 才停（日志里典型现象）。
                if (not is_error
                        and tc.name == "submit_task_result"
                        and isinstance(result_text, str)
                        and result_text.startswith("**提交被阻止**")):
                    if self._tool_breaker is not None:
                        try:
                            self._tool_breaker.record_tool_failure(
                                tool_name="submit_task_result",
                                status="rejected_by_gatekeeper",
                                # 把拒绝理由的前若干字节作为 message —— breaker
                                # 用它做指纹，相同理由的拒绝会落在同一指纹上，
                                # 触发 same_error_threshold 时熔断
                                message=result_text[:160],
                            )
                        except Exception as breaker_err:
                            logger.debug(
                                "breaker 记录 submit 拒绝失败（吞掉）: %s",
                                breaker_err,
                            )
                    if self._critic_injector is not None:
                        try:
                            tool_err_cls = type(
                                "ToolResult_submit_rejected",
                                (RuntimeError,),
                                {},
                            )
                            self._critic_injector.record_failure(
                                tool_name=tc.name,
                                tool_arguments=tc.arguments,
                                error=tool_err_cls(result_text[:200]),
                            )
                        except Exception as critic_err:
                            logger.debug(
                                "critic 记录 submit 拒绝失败（吞掉）: %s",
                                critic_err,
                            )

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

        # ERROR 与 FAILED 含义不同，分别处理：
        #   - ERROR  ：工具不可用（环境问题），喂 breaker 防止死循环重试，
        #             但**不**喂 critic_injector —— 这不是代码错误，
        #             不应让 LLM 反思"我哪里写错了"。
        #   - FAILED ：真实验证失败（lint 错、test 红），喂 breaker + critic。
        #   - PASSED ：成功 → 清除 critic 上一次失败的污染，避免持续注入。
        msg = data.get("message", "")
        if status == ValidationStatus.ERROR:
            if self._tool_breaker is not None:
                self._tool_breaker.record_tool_failure(
                    tool_name=tool_name,
                    status=status.value,
                    message=msg,
                )
        elif status == ValidationStatus.FAILED:
            if self._tool_breaker is not None:
                self._tool_breaker.record_tool_failure(
                    tool_name=tool_name,
                    status=status.value,
                    message=msg,
                )
            if self._critic_injector is not None:
                # 用动态命名的子类，让 critic_injector 拿到的 error_type 能区分
                # 工具来源（ToolResult_run_linter / ToolResult_run_tests …）。
                tool_err_cls = type(
                    f"ToolResult_{tool_name}",
                    (RuntimeError,),
                    {},
                )
                synthetic_err = tool_err_cls(f"[{status.value}] {msg}")
                self._critic_injector.record_failure(
                    tool_name=tool_name,
                    tool_arguments={"status": status.value},
                    error=synthetic_err,
                )
        elif status == ValidationStatus.PASSED:
            # 验证通过 → agent 已自我修复，清掉 critic 旧失败上下文，
            # 避免后续 prompt 仍带着"你失败了"的警示。
            if self._critic_injector is not None and self._critic_injector.has_recent_failure():
                self._critic_injector.clear()
                logger.debug(
                    "验证通过，清除 critic 失败上下文: tool=%s", tool_name,
                )
