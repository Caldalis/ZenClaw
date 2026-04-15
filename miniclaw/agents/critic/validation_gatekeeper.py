"""
验证门禁器 — 强制验证闭环

核心职责:
  1. 跟踪验证工具的调用状态
  2. 强制要求至少一次成功的验证
  3. 拦截未验证就声明成功的 submit_task_result

设计原则:
  - 没有 run_linter 或 run_tests 的成功调用
  - submit_task_result 将被拒绝
  - 返回错误提示要求先验证
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

from miniclaw.agents.critic.validation_tools import ValidationStatus, ValidationResult
from miniclaw.utils.logging import get_logger

logger = get_logger(__name__)


class ValidationRequirement(str, Enum):
    """验证要求级别"""
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

    使用方式:
        gatekeeper = ValidationGatekeeper(config)

        # 记录验证结果
        gatekeeper.record_validation("run_linter", result)

        # 检查是否允许提交
        if gatekeeper.can_submit():
            await submit_task_result(...)
        else:
            return "请先运行 run_linter 或 run_tests 验证代码"
    """

    def __init__(self, config: GatekeeperConfig | None = None):
        self.config = config or GatekeeperConfig()
        self._validation_records: list[ValidationRecord] = []
        self._skip_reason: str | None = None

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

        # 验证理由不能太短
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
        logger.debug("验证门禁已重置")

    def _get_valid_records(self) -> list[ValidationRecord]:
        """获取有效期内的验证记录"""
        if self.config.max_validation_age_seconds <= 0:
            return self._validation_records

        cutoff = datetime.now(timezone.utc).timestamp() - self.config.max_validation_age_seconds

        return [
            r for r in self._validation_records
            if r.timestamp.timestamp() >= cutoff
        ]

    def _has_passed_validation(self, records: list[ValidationRecord]) -> bool:
        """是否有通过验证的记录"""
        return any(r.status == ValidationStatus.PASSED for r in records)

    def _has_passed_linter(self, records: list[ValidationRecord]) -> bool:
        """是否有通过 linter 的记录"""
        return any(
            r.status == ValidationStatus.PASSED and "linter" in r.tool_name
            for r in records
        )

    def _has_passed_tests(self, records: list[ValidationRecord]) -> bool:
        """是否有通过测试的记录"""
        return any(
            r.status == ValidationStatus.PASSED and "test" in r.tool_name
            for r in records
        )

    def _get_validation_required_message(self) -> str:
        """获取验证要求消息"""
        if self.config.requirement == ValidationRequirement.ANY:
            return """**验证要求**: 在调用 submit_task_result 前，必须先运行 `run_linter` 或 `run_tests` 并确保通过。

请执行以下操作之一:
1. 调用 `run_linter` 检查代码质量
2. 调用 `run_tests` 运行测试

如果有充分的理由无法验证，请在 submit_task_result 的 unresolved_issues 中说明原因。"""

        elif self.config.requirement == ValidationRequirement.LINTER:
            return "**验证要求**: 在调用 submit_task_result 前，必须先运行 `run_linter` 并确保通过。"

        elif self.config.requirement == ValidationRequirement.TESTS:
            return "**验证要求**: 在调用 submit_task_result 前，必须先运行 `run_tests` 并确保通过。"

        elif self.config.requirement == ValidationRequirement.BOTH:
            return """**验证要求**: 在调用 submit_task_result 前，必须同时:
1. 运行 `run_linter` 并确保通过
2. 运行 `run_tests` 并确保通过"""

        return "需要验证"

    def _get_validation_failed_message(self) -> str:
        """获取验证失败消息"""
        return """**验证未通过**: 你的代码验证失败，请修复问题后重新验证。

1. 根据错误信息修复代码
2. 重新运行验证工具
3. 验证通过后再调用 submit_task_result"""

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


class ValidationAwareSubmitTool:
    """验证感知的提交工具包装器

    在 submit_task_result 工具执行前检查验证状态。
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
        """执行提交（带验证检查）"""
        can_submit, reason = self._gatekeeper.can_submit()

        if not can_submit:
            return f"**提交被阻止**\n\n{reason}"

        # 检查是否有跳过理由（放在 unresolved_issues 中）
        unresolved = kwargs.get("unresolved_issues", "")
        if unresolved and len(unresolved) >= 20:
            # 如果有未解决问题，可能是不需要验证的场景
            pass

        return await self._original_tool.execute(**kwargs)

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