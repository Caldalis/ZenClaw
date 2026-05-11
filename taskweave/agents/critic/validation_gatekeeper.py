"""
验证门禁器 — 强制验证闭环

  1. 跟踪验证工具的调用状态
  2. 强制要求至少一次成功的验证
  3. 拦截未验证就声明成功的 submit_task_result

  - 没有 run_linter 或 run_tests 的成功调用
  - submit_task_result 将被拒绝
  - 返回错误提示要求先验证
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

from taskweave.agents.critic.validation_tools import ValidationStatus, ValidationResult
from taskweave.tools.base import Tool
from taskweave.utils.logging import get_logger

logger = get_logger(__name__)


class ValidationRequirement(str, Enum):
    NONE = "none"              # 无需验证
    ANY = "any"                # 任意一个验证通过即可
    LINTER = "linter"          # 必须通过 linter
    TESTS = "tests"            # 必须通过测试
    BOTH = "both"              # 必须同时通过 linter 和测试


@dataclass
class ValidationRecord:
    """验证记录"""
    tool_name: str
    """工具名称"""

    status: ValidationStatus
    """验证状态"""

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    """验证时间"""

    details: dict[str, Any] = field(default_factory=dict)
    """详细信息"""


@dataclass
class GatekeeperConfig:
    """门禁配置"""
    requirement: ValidationRequirement = ValidationRequirement.ANY
    """验证要求"""

    allow_skip_with_reason: bool = True
    """是否允许带理由跳过验证"""

    max_validation_age_seconds: int = 300
    """验证结果有效期（秒）"""

    blocked_tools_without_validation: list[str] = field(
        default_factory=lambda: ["submit_task_result"]
    )
    """未验证时阻止的工具"""


class ValidationGatekeeper:
    """验证门禁器 — 强制验证闭环
    """

    def __init__(self, config: GatekeeperConfig | None = None):
        self.config = config or GatekeeperConfig()
        self._validation_records: list[ValidationRecord] = []
        self._skip_reason: str | None = None
        # 同一任务里 submit 被拒的累计次数。LLM 第二次还在重复同样的错误说明
        # 它没读息懂第一条拒绝消——这时升级为更强硬的指令式提示词
        # 第三次还重复=让 ValidationAwareSubmitTool 把状态降级为 partial_success
        # 自动放行避免反复撞墙浪费max次数
        self._submit_rejection_count: int = 0

    def record_validation(
        self,
        tool_name: str,
        result: ValidationResult,
    ) -> None:
        """记录验证结果

        Args:
            tool_name: 工具名称
            result: 验证结果
        """
        record = ValidationRecord(
            tool_name=tool_name,
            status=result.status,
            details={
                "message": result.message,
                "error_count": len(result.errors),
                "warning_count": len(result.warnings),
            },
        )

        self._validation_records.append(record)
        self._skip_reason = None  # 有验证后清除跳过理由

        logger.info(
            "验证记录: tool=%s, status=%s, errors=%d",
            tool_name, result.status.value, len(result.errors)
        )

    def can_submit(self) -> tuple[bool, str]:
        """检查是否允许提交

        Returns:
            (allowed, reason): 是否允许，不允许时的原因
        """
        if self.config.requirement == ValidationRequirement.NONE:
            return True, ""

        # 检查是否有跳过理由
        if self._skip_reason and self.config.allow_skip_with_reason:
            logger.info("验证已跳过: %s", self._skip_reason)
            return True, f"跳过验证: {self._skip_reason}"

        # 检查验证记录
        valid_records = self._get_valid_records()

        if not valid_records:
            return False, self._get_validation_required_message()

        # 根据要求检查
        if self.config.requirement == ValidationRequirement.ANY:
            if self._has_passed_validation(valid_records):
                return True, ""

        elif self.config.requirement == ValidationRequirement.LINTER:
            if self._has_passed_linter(valid_records):
                return True, ""

        elif self.config.requirement == ValidationRequirement.TESTS:
            if self._has_passed_tests(valid_records):
                return True, ""

        elif self.config.requirement == ValidationRequirement.BOTH:
            if self._has_passed_linter(valid_records) and self._has_passed_tests(valid_records):
                return True, ""

        return False, self._get_validation_failed_message()

    def can_execute_tool(self, tool_name: str) -> tuple[bool, str]:
        """检查是否允许执行工具

        Args:
            tool_name: 工具名称

        Returns:
            (allowed, reason): 是否允许，不允许时的原因
        """
        if tool_name not in self.config.blocked_tools_without_validation:
            return True, ""

        return self.can_submit()

    def skip_validation(self, reason: str) -> bool:
        """请求跳过验证
        Args:
            reason: 跳过理由
        Returns:
            是否允许跳过
        """
        if not self.config.allow_skip_with_reason:
            logger.warning("跳过验证被拒绝: 不允许跳过")
            return False
        if len(reason) < 20:
            logger.warning("跳过验证被拒绝: 理由太短")
            return False

        self._skip_reason = reason
        logger.info("验证跳过已记录: %s", reason[:50])
        return True

    def reset(self) -> None:
        """重置验证状态"""
        self._validation_records.clear()
        self._skip_reason = None
        self._submit_rejection_count = 0
        logger.debug("验证门禁已重置")

    def note_submit_rejected(self) -> int:
        """登记一次 submit 被拒，返回累计计数。

        ValidationAwareSubmitTool 在拒绝 submit 时调用这个方法，让门禁能
        感知"agent 没有按提示去补做验证，而是反复硬撞 submit"的死循环。
        """
        self._submit_rejection_count += 1
        return self._submit_rejection_count

    @property
    def submit_rejection_count(self) -> int:
        return self._submit_rejection_count

    def should_force_release(self) -> bool:
        """是否应该强制放行 submit（避免无限拒绝循环）。

        触发条件：submit 已被拒过 >=2 次，且当前还是没有任何验证记录。
        这是兜底机制 —— Agent 看了两次升级提示仍不会调验证工具，说明
        prompt 已经无法纠偏，再拒第三次只是浪费 ReAct 步数；放行并把
        status 强制降级到 partial_success，把决策权交还给 Master。
        """
        return (
            self._submit_rejection_count >= 2
            and not self._get_valid_records()
            and not self._skip_reason
        )

    def _get_valid_records(self) -> list[ValidationRecord]:
        """获取有效期内的验证记录"""
        if self.config.max_validation_age_seconds <= 0:
            return self._validation_records

        cutoff = datetime.now(timezone.utc).timestamp() - self.config.max_validation_age_seconds

        return [
            r for r in self._validation_records
            if r.timestamp.timestamp() >= cutoff
        ]

    # 视为"已尝试且无阻碍"的状态：
    #   - PASSED: 工具通过 —— 这是唯一允许 submit 的真实信号
    # RunLinterTool 在检测不到 ruff/flake8/pylint 时，自动降级
    # 到 stdlib `python -m py_compile` 做语法检查。py_compile 不需要装外部包，能给出真正的 PASSED / FAILED 信号。所以 ERROR 现在只在罕见情况
    # （worktree 没 .py 文件、py_compile 自身炸了等出现，不再作为常规死锁规避手段。
    _ACCEPTABLE_STATUSES = (ValidationStatus.PASSED,)

    def _has_passed_validation(self, records: list[ValidationRecord]) -> bool:
        """是否有真正通过的验证记录"""
        return any(r.status in self._ACCEPTABLE_STATUSES for r in records)

    def _has_passed_linter(self, records: list[ValidationRecord]) -> bool:
        """是否有 linter 真正通过的记录"""
        return any(
            r.status in self._ACCEPTABLE_STATUSES and "linter" in r.tool_name
            for r in records
        )

    def _has_passed_tests(self, records: list[ValidationRecord]) -> bool:
        """是否有 test 真正通过的记录"""
        return any(
            r.status in self._ACCEPTABLE_STATUSES and "test" in r.tool_name
            for r in records
        )

    def _get_validation_required_message(self) -> str:
        """获取验证要求消息

        消息会随被拒次数升级 —— 同一段 prompt 既然第一次没让 LLM 改路径，
        第二次必须更具指令性，否则只是浪费 token。
        """
        rej = self._submit_rejection_count
        # 第一次：教学式说明 + 列工具
        # 第二次及以上：指令式 + 抨击 anti-pattern（terminal 跑 pytest 不算验证）
        if self.config.requirement == ValidationRequirement.ANY:
            if rej <= 1:
                return (
                    "**验证要求**: 在调用 submit_task_result 前，必须先运行 "
                    "`run_linter` 或 `run_tests`，并且拿到 `status=passed`。\n\n"
                    "请执行以下操作之一:\n"
                    "1. 调用 `run_linter` 工具检查代码质量\n"
                    "2. 调用 `run_tests` 工具运行测试\n\n"
                    "注意：\n"
                    "- 这是**工具调用**，不是 terminal 命令。"
                    "用 `terminal` 执行 `python -m flake8 / pytest / py_compile` "
                    "**不会**被识别为验证。\n"
                    "- `run_linter` 在没装 ruff/flake8 的机器上会自动降级到 "
                    "stdlib `py_compile` 做语法检查 —— 你只管调用，不要自己装包。\n"
                    "- 只有 `status=passed` 才会被门禁认可。`status=failed` 表示"
                    "代码真的有问题，需要修复后重新跑。`status=error` 表示工具"
                    "本身跑不起来（罕见），属于环境问题需要单独处理。"
                )
            # 第二次拒绝：升级到指令式
            return (
                f"**第 {rej} 次提交被拦截 — 你没有按上一条提示行动。**\n\n"
                "你现在必须立刻、单独地调用一次 `run_linter` 工具（参数留空即可），"
                "或者 `run_tests` 工具。**这是工具调用**，不是 `terminal` 命令。\n\n"
                "❌ 不要：再次调用 submit_task_result\n"
                "❌ 不要：用 terminal 跑 flake8 / pytest / py_compile（不被识别）\n"
                "❌ 不要：先去 read_file 或 ls 探索环境\n"
                "✅ 必做：现在这一步，工具调用 `run_linter`，拿到 status=passed 后再 submit。\n\n"
                "若 `run_linter` 返回 `status=failed`，说明代码真的有问题，先修代码再重跑。\n"
                "若返回 `status=error`（罕见，py_compile fallback 都跑不起来），"
                "用 `skip_validation(reason=...)` 显式说明无法验证的真实原因 —— "
                "门禁不再静默放行未验证的提交。"
            )

        elif self.config.requirement == ValidationRequirement.LINTER:
            base = "**验证要求**: 在调用 submit_task_result 前，必须先运行 `run_linter` 工具并确保通过。"
            if rej >= 2:
                base += f"\n\n这是第 {rej} 次拦截。请直接调用 `run_linter` 工具（不是 terminal）。"
            return base

        elif self.config.requirement == ValidationRequirement.TESTS:
            base = "**验证要求**: 在调用 submit_task_result 前，必须先运行 `run_tests` 工具并确保通过。"
            if rej >= 2:
                base += f"\n\n这是第 {rej} 次拦截。请直接调用 `run_tests` 工具（不是 terminal）。"
            return base

        elif self.config.requirement == ValidationRequirement.BOTH:
            return (
                "**验证要求**: 在调用 submit_task_result 前，必须同时:\n"
                "1. 运行 `run_linter` 工具并确保通过\n"
                "2. 运行 `run_tests` 工具并确保通过"
            )

        return "需要验证"

    def _get_validation_failed_message(self) -> str:
        """获取验证失败消息

        到这里说明已经有验证记录但都不是 PASSED/ERROR — 即至少有一次 FAILED。
        """
        last_failed = next(
            (r for r in reversed(self._validation_records)
             if r.status == ValidationStatus.FAILED),
            None,
        )
        detail = ""
        if last_failed is not None:
            detail = (
                f"\n\n最近一次失败: tool={last_failed.tool_name}, "
                f"errors={last_failed.details.get('error_count', 0)}, "
                f"message={last_failed.details.get('message', '')[:120]}"
            )
        return (
            "**验证未通过**: 你的代码验证失败，请修复问题后重新验证。\n\n"
            "1. 根据错误信息修复代码\n"
            "2. 重新运行验证工具\n"
            "3. 验证通过后再调用 submit_task_result"
            + detail
        )

    def get_status(self) -> dict[str, Any]:
        """获取状态"""
        valid_records = self._get_valid_records()

        return {
            "requirement": self.config.requirement.value,
            "validation_count": len(self._validation_records),
            "valid_record_count": len(valid_records),
            "has_passed": self._has_passed_validation(valid_records),
            "has_passed_linter": self._has_passed_linter(valid_records),
            "has_passed_tests": self._has_passed_tests(valid_records),
            "skip_reason": self._skip_reason,
            "can_submit": self.can_submit()[0],
            "recent_validations": [
                {
                    "tool": r.tool_name,
                    "status": r.status.value,
                    "time": r.timestamp.isoformat(),
                }
                for r in valid_records[-5:]  # 最近 5 条
            ],
        }


class ValidationAwareSubmitTool(Tool):
    """验证感知的提交工具包装器

    在 submit_task_result 工具执行前检查验证状态。
    继承 Tool 以兼容 ToolRegistry.register()。
    """

    def __init__(
        self,
        gatekeeper: ValidationGatekeeper,
        original_submit_tool,  # SubmitTaskResultTool
    ):
        self._gatekeeper = gatekeeper
        self._original_tool = original_submit_tool

    @property
    def name(self) -> str:
        return self._original_tool.name

    @property
    def description(self) -> str:
        return self._original_tool.description + """
**验证要求**: 调用此工具前，必须先运行 `run_linter` 或 `run_tests` 验证代码。"""

    @property
    def parameters(self) -> dict[str, Any]:
        return self._original_tool.parameters

    async def execute(self, **kwargs: Any) -> str:
        """执行提交（带验证检查）

        拒绝阶梯：
          第 1 次：教学式提示，告诉 agent 调 run_linter / run_tests
          第 2 次：指令式提示，明确拒绝 terminal 替代
          第 3 次：强制放行（status 降级为 partial_success）— Master 接管决策

        强制放行的设计动机：当 prompt 已经无法纠偏 agent 行为时，继续拒绝
        只是消耗 ReAct 步数最终走向 max_iterations。倒不如把任务作为
        partial_success 交付，把"验证缺失"作为已知风险写进 unresolved_issues，
        让 Master 在收到结果后明知缺陷做出选择（重派给 Tester / 接受 / 失败）。
        """
        can_submit, reason = self._gatekeeper.can_submit()

        if can_submit:
            return await self._original_tool.execute(**kwargs)

        # 登记一次拒绝并取最新计数
        rej_count = self._gatekeeper.note_submit_rejected()

        # 阶梯三：强制放行
        if self._gatekeeper.should_force_release():
            logger.warning(
                "submit 已被拒绝 %d 次，agent 仍未调用验证工具 — 强制放行并降级为 partial_success",
                rej_count,
            )
            patched = dict(kwargs)
            # 把状态降级，避免假阳性"成功"
            patched["status"] = "partial_success"
            existing_unresolved = (kwargs.get("unresolved_issues") or "").strip()
            forced_note = (
                "[系统注记] 验证门禁已被强制放行：agent 多次拒绝后未调用 "
                "run_linter / run_tests，产物未通过验证闭环，建议下游任务（如 Tester）补做验证。"
            )
            if existing_unresolved:
                patched["unresolved_issues"] = existing_unresolved + " | " + forced_note
            else:
                patched["unresolved_issues"] = forced_note
            return await self._original_tool.execute(**patched)

        # 仍然拒绝
        return f"**提交被阻止**\n\n{reason}"

    def to_tool_schema(self) -> dict[str, Any]:
        return self._original_tool.to_tool_schema()


# 导出
__all__ = [
    "ValidationRequirement",
    "ValidationRecord",
    "GatekeeperConfig",
    "ValidationGatekeeper",
    "ValidationAwareSubmitTool",
]