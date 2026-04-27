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
from miniclaw.types.structured_result import validate_result
from miniclaw.utils.logging import get_logger

logger = get_logger(__name__)


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
        3. 返回完整的结构化结果（以便调度器从 tool_call_result 事件解析）

        """
        try:
            # 验证并清理结果
            validated = validate_result(kwargs)

            logger.info(
                "任务结果已提交: status=%s, files=%d",
                validated.status.value,
                len(validated.files_changed),
            )

            payload = validated.to_json()
            payload["confirmed"] = True
            payload["message"] = "结果已提交，任务完成。系统会将结果返回给调度者。"
            return json.dumps(payload, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error("任务结果提交失败: %s", e)
            return f"错误: 结果提交失败 - {str(e)}"


# 工具实例
submit_task_result_tool = SubmitTaskResultTool()


# 导出
__all__ = [
    "SubmitTaskResultTool",
    "submit_task_result_tool",
]