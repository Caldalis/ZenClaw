"""
结构化结果数据模型

定义 Subagent 必须返回的结构化结果格式。

核心概念:
  - TaskStatus: 任务状态枚举
  - StructuredResult: 结构化结果模型
  - ResultValidator: 结果验证器
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class TaskStatus(str, Enum):
    """任务完成状态"""
    SUCCESS = "success"              # 完全成功
    PARTIAL_SUCCESS = "partial_success"  # 部分成功
    FAILED = "failed"                # 失败


class StructuredResult(BaseModel):
    """结构化结果 — Subagent 必须返回的格式

    这是防止"结果爆炸"的关键机制：
      - 限制返回内容的长度
      - 强制结构化格式
      - 便于 Master Agent 理解和决策
    """

    status: TaskStatus
    """任务状态: success, partial_success, failed"""

    files_changed: list[str] = Field(default_factory=list)
    """修改/创建的文件列表（仅文件名，不含完整路径）"""

    summary: str
    """任务完成摘要（建议 50-200 字）"""

    unresolved_issues: str = ""
    """遗留问题或待后续处理的项（可选）"""

    # 扩展字段（可选）
    details: dict[str, Any] = Field(default_factory=dict)
    """额外的详细信息（建议保持精简）"""

    # 元数据（系统填充）
    agent_id: str | None = None
    """执行该任务的 Agent ID"""

    turn_id: str | None = None
    """Turn ID（用于追踪）"""

    execution_time_ms: int | None = None
    """执行耗时（毫秒）"""

    @field_validator("summary")
    @classmethod
    def validate_summary_length(cls, v: str) -> str:
        """验证摘要长度（防止过长）"""
        if len(v) > 500:
            # 截断并添加省略号
            return v[:497] + "..."
        return v

    @field_validator("unresolved_issues")
    @classmethod
    def validate_issues_length(cls, v: str) -> str:
        """验证遗留问题长度"""
        if len(v) > 300:
            return v[:297] + "..."
        return v

    def is_success(self) -> bool:
        """是否成功完成"""
        return self.status == TaskStatus.SUCCESS

    def is_partial(self) -> bool:
        """是否部分成功"""
        return self.status == TaskStatus.PARTIAL_SUCCESS

    def is_failed(self) -> bool:
        """是否失败"""
        return self.status == TaskStatus.FAILED

    def to_master_context(self) -> str:
        """转换为适合 Master Agent 上下文的精简文本

        这是防止上下文污染的关键方法：
          - 只保留关键信息
          - 格式标准化
          - 长度受限
        """
        lines = [
            f"**任务状态**: {self.status.value}",
            f"**摘要**: {self.summary}",
        ]

        if self.files_changed:
            lines.append(f"**修改文件**: {', '.join(self.files_changed[:10])}")
            if len(self.files_changed) > 10:
                lines.append(f"  ... 等 {len(self.files_changed)} 个文件")

        if self.unresolved_issues:
            lines.append(f"**遗留问题**: {self.unresolved_issues}")

        return "\n".join(lines)

    def to_json(self) -> dict[str, Any]:
        """转换为 JSON 格式（用于工具返回）"""
        return {
            "status": self.status.value,
            "files_changed": self.files_changed,
            "summary": self.summary,
            "unresolved_issues": self.unresolved_issues,
        }


class ResultValidationConfig(BaseModel):
    """结果验证配置"""
    max_summary_length: int = 500
    max_issues_length: int = 300
    max_files_count: int = 50
    max_details_keys: int = 10


class ResultValidator:
    """结果验证器

    验证并清理 Subagent 返回的结构化结果。
    """

    def __init__(self, config: ResultValidationConfig | None = None):
        self.config = config or ResultValidationConfig()

    def validate(self, result: dict[str, Any]) -> StructuredResult:
        """验证并清理结果

        Args:
            result: 原始结果字典

        Returns:
            验证后的 StructuredResult

        Raises:
            ValueError: 结果格式严重错误
        """
        # 提取必填字段
        status_str = result.get("status", "failed")
        try:
            status = TaskStatus(status_str)
        except ValueError:
            status = TaskStatus.FAILED

        # 清理 summary
        summary = str(result.get("summary", "无摘要"))
        if len(summary) > self.config.max_summary_length:
            summary = summary[:self.config.max_summary_length - 3] + "..."

        # 清理 files_changed
        files = result.get("files_changed", [])
        if not isinstance(files, list):
            files = []
        files = [str(f) for f in files[:self.config.max_files_count]]

        # 清理 unresolved_issues
        issues = str(result.get("unresolved_issues", ""))
        if len(issues) > self.config.max_issues_length:
            issues = issues[:self.config.max_issues_length - 3] + "..."

        # 清理 details
        details = result.get("details", {})
        if not isinstance(details, dict):
            details = {}
        details = dict(list(details.items())[:self.config.max_details_keys])

        return StructuredResult(
            status=status,
            files_changed=files,
            summary=summary,
            unresolved_issues=issues,
            details=details,
        )

    def validate_to_json(self, result: dict[str, Any]) -> dict[str, Any]:
        """验证并返回 JSON 格式"""
        validated = self.validate(result)
        return validated.to_json()


# 全局验证器实例
_default_validator = ResultValidator()


def validate_result(result: dict[str, Any]) -> StructuredResult:
    """验证结果（使用默认配置）"""
    return _default_validator.validate(result)


def result_to_master_context(result: dict[str, Any]) -> str:
    """将结果转换为 Master Agent 上下文文本"""
    validated = validate_result(result)
    return validated.to_master_context()


# 导出
__all__ = [
    "TaskStatus",
    "StructuredResult",
    "ResultValidationConfig",
    "ResultValidator",
    "validate_result",
    "result_to_master_context",
]