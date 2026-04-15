"""
submit_task_result 工具 — Subagent 必须调用的结果提交工具

这是结构化输出协议的核心：
  - Subagent 结束循环前必须调用此工具
  - 参数必须是严格的 JSON schema
  - 系统抓取精简 JSON 喂回 Master Agent 上下文
"""

from __future__ import annotations

import json
from typing import Any

from miniclaw.tools.base import Tool
from miniclaw.types.structured_result import (
    StructuredResult,
    TaskStatus,
    validate_result,
)
from miniclaw.utils.logging import get_logger

logger = get_logger(__name__)


# 全局结果存储（用于 TaskScheduler 回收结果）
_pending_results: dict[str, StructuredResult] = {}


def get_pending_result(task_id: str) -> StructuredResult | None:
    """获取待处理的结果"""
    return _pending_results.pop(task_id, None)


def store_result(task_id: str, result: StructuredResult) -> None:
    """存储结果"""
    _pending_results[task_id] = result


class SubmitTaskResultTool(Tool):
    """任务结果提交工具 — Subagent 必须调用的结束工具

    这是防止"结果爆炸"的关键机制：
      1. 强制结构化格式
      2. 长度限制
      3. 标准化输出

    Subagent 在完成任务后，必须调用此工具提交结果。
    """

    @property
    def name(self) -> str:
        return "submit_task_result"

    @property
    def description(self) -> str:
        return """提交任务完成结果。

**这是必须调用的工具**。完成任务后，你必须调用此工具提交结果，否则任务不会被标记为完成。

结果格式要求：
- status: "success"（成功）| "partial_success"（部分成功）| "failed"（失败）
- files_changed: 修改/创建的文件列表
- summary: 任务完成摘要（50-200字，精简描述）
- unresolved_issues: 遗留问题（可选）

示例：
```json
{
    "status": "success",
    "files_changed": ["src/api.js", "src/utils.js"],
    "summary": "重构了网络层，提取了公共请求方法，添加了错误处理",
    "unresolved_issues": "兼容 IE11 的问题待后续处理"
}
```"""

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["success", "partial_success", "failed"],
                    "description": "任务完成状态。success=完全成功，partial_success=部分成功，failed=失败",
                },
                "files_changed": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "修改或创建的文件列表（仅文件名）",
                    "default": [],
                },
                "summary": {
                    "type": "string",
                    "description": "任务完成摘要，建议 50-200 字，精简描述做了什么",
                },
                "unresolved_issues": {
                    "type": "string",
                    "description": "遗留问题或待后续处理的项（可选）",
                    "default": "",
                },
                "details": {
                    "type": "object",
                    "description": "额外的详细信息（可选，建议保持精简）",
                    "default": {},
                },
            },
            "required": ["status", "summary"],
        }

    async def execute(self, **kwargs: Any) -> str:
        """执行结果提交

        1. 验证参数格式
        2. 清理和截断
        3. 存储结果
        4. 返回确认
        """
        try:
            # 验证并清理结果
            validated = validate_result(kwargs)

            # 生成结果 ID
            result_id = f"result_{id(validated)}"

            # 存储结果
            store_result(result_id, validated)

            logger.info(
                "任务结果已提交: status=%s, files=%d",
                validated.status.value,
                len(validated.files_changed),
            )

            # 返回确认信息
            return self._format_confirmation(validated, result_id)

        except Exception as e:
            logger.error("任务结果提交失败: %s", e)
            return f"错误: 结果提交失败 - {str(e)}"

    def _format_confirmation(self, result: StructuredResult, result_id: str) -> str:
        """格式化确认信息"""
        return json.dumps({
            "confirmed": True,
            "result_id": result_id,
            "status": result.status.value,
            "message": "结果已提交，任务完成。系统会将结果返回给调度者。",
        }, ensure_ascii=False, indent=2)


class SubmitTaskResultWrapper:
    """结果提交包装器

    用于在 Agent 执行过程中收集结果，确保最终提交。
    """

    def __init__(self, task_id: str):
        self.task_id = task_id
        self._result: StructuredResult | None = None
        self._submitted = False

    def set_result(self, result: StructuredResult) -> None:
        """设置结果"""
        self._result = result

    def set_result_from_dict(self, data: dict[str, Any]) -> None:
        """从字典设置结果"""
        self._result = validate_result(data)

    def submit(self) -> StructuredResult | None:
        """提交结果"""
        if self._result and not self._submitted:
            self._result.turn_id = self.task_id
            store_result(self.task_id, self._result)
            self._submitted = True
            return self._result
        return None

    @property
    def is_submitted(self) -> bool:
        """是否已提交"""
        return self._submitted

    @property
    def result(self) -> StructuredResult | None:
        """获取结果"""
        return self._result


# 工具实例
submit_task_result_tool = SubmitTaskResultTool()


# 导出
__all__ = [
    "SubmitTaskResultTool",
    "SubmitTaskResultWrapper",
    "submit_task_result_tool",
    "get_pending_result",
    "store_result",
]