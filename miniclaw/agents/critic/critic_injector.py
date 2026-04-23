"""
Critic 注入器 — 动态注入错误警示提示词

核心职责:
  1. 截获错误信息 (isError: true)
  2. 在下一个 Prompt 中注入警示
  3. 强制 Agent 思考失败原因并提供新方向

设计原则:
  - "你上一次的尝试失败了，请在思考中明确指出错在哪里"
  - "严禁重复上述代码"
  - 提供上下文帮助 Agent 理解问题
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from miniclaw.agents.critic.circuit_breaker import ErrorPattern
from miniclaw.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FailureContext:
    """失败上下文 — 记录最近一次失败的信息"""
    tool_name: str
    """失败的工具名称"""

    tool_arguments: dict[str, Any]
    """工具参数"""

    error_message: str
    """错误消息"""

    error_type: str = "UnknownError"
    """错误类型"""

    error_pattern: ErrorPattern | None = None
    """错误模式（如果识别出）"""

    attempt_number: int = 1
    """尝试次数"""

    previous_attempts: list[dict[str, Any]] = field(default_factory=list)
    """之前的尝试记录"""

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class CriticConfig:
    """Critic 配置"""
    max_previous_attempts: int = 3
    """最多保留多少次之前的尝试"""

    include_code_snippet: bool = True
    """是否包含代码片段"""

    include_stack_trace: bool = False
    """是否包含完整堆栈"""

    force_thinking_block: bool = True
    """是否强制要求 <thinking> 块"""


class CriticInjector:
    """Critic 注入器 — 在错误后注入警示提示

    使用方式:
        injector = CriticInjector()

        # 记录失败
        injector.record_failure(tool_name, tool_args, error)

        # 获取警示提示
        warning = injector.get_warning_prompt()

        # 添加到下一个消息的 system prompt
        augmented_prompt = base_prompt + warning
    """

    def __init__(self, config: CriticConfig | None = None):
        self.config = config or CriticConfig()
        self._failure_context: FailureContext | None = None
        self._failure_history: list[FailureContext] = []

    def record_failure(
        self,
        tool_name: str,
        tool_arguments: dict[str, Any],
        error: Exception | str,
        error_pattern: ErrorPattern | None = None,
    ) -> None:
        """记录失败

        Args:
            tool_name: 工具名称
            tool_arguments: 工具参数
            error: 错误对象或错误消息
            error_pattern: 可选的错误模式
        """
        error_message = str(error)
        error_type = type(error).__name__ if isinstance(error, Exception) else "ToolError"

        # 更新历史
        if self._failure_context:
            self._failure_history.append(self._failure_context)
            # 限制历史长度
            if len(self._failure_history) > self.config.max_previous_attempts:
                self._failure_history.pop(0)

        # 创建新的失败上下文
        self._failure_context = FailureContext(
            tool_name=tool_name,
            tool_arguments=tool_arguments,
            error_message=error_message,
            error_type=error_type,
            error_pattern=error_pattern,
            attempt_number=len(self._failure_history) + 1,
            previous_attempts=[
                {
                    "tool": f.tool_name,
                    "error": f.error_message[:100],
                }
                for f in self._failure_history
            ],
        )

        logger.warning(
            "Critic 记录失败: tool=%s, error_type=%s, attempt=%d",
            tool_name, error_type, self._failure_context.attempt_number
        )

    def get_warning_prompt(self) -> str:
        """获取警示提示词

        Returns:
            警示提示词（添加到 system prompt）
        """
        if self._failure_context is None:
            return ""

        parts = []

        # 主警示
        parts.append(self._build_main_warning())

        # 详细错误信息
        parts.append(self._build_error_details())

        # 之前尝试摘要
        if self._failure_context.previous_attempts:
            parts.append(self._build_previous_attempts_summary())

        # 行为指导
        parts.append(self._build_guidance())

        return "\n\n".join(parts)

    def _build_main_warning(self) -> str:
        """构建主警示"""
        ctx = self._failure_context
        attempt_word = "第一次" if ctx.attempt_number == 1 else f"第 {ctx.attempt_number} 次"

        return f"""## ⚠️ 注意：你的上一次尝试失败了

你在调用 `{ctx.tool_name}` 时遇到了 **{ctx.error_type}** 错误。这是你{attempt_word}尝试。

**重要提醒**：
1. 请在 `<thinking>` 块中**明确指出错误原因**
2. 分析之前的代码/命令哪里出了问题
3. 提供**全新的解决方案**，**严禁重复相同的代码或命令**"""

    def _build_error_details(self) -> str:
        """构建错误详情"""
        ctx = self._failure_context

        parts = ["### 错误详情", f"- **工具**: `{ctx.tool_name}`"]

        # 错误类型和消息
        parts.append(f"- **错误类型**: {ctx.error_type}")
        parts.append(f"- **错误消息**: {ctx.error_message[:300]}")

        # 如果有错误模式信息
        if ctx.error_pattern:
            if ctx.error_pattern.location:
                parts.append(f"- **错误位置**: `{ctx.error_pattern.location}`")
            if ctx.error_pattern.count > 1:
                parts.append(f"- **重复次数**: 该错误已出现 {ctx.error_pattern.count} 次")

        # 工具参数（选择性显示）
        if ctx.tool_arguments:
            args_preview = self._format_arguments(ctx.tool_arguments)
            parts.append(f"- **参数**: {args_preview}")

        return "\n".join(parts)

    def _build_previous_attempts_summary(self) -> str:
        """构建之前尝试摘要"""
        ctx = self._failure_context

        if not ctx.previous_attempts:
            return ""

        lines = ["### 之前的尝试", "你已经尝试过以下方法但都失败了：", ""]

        for i, attempt in enumerate(ctx.previous_attempts, 1):
            lines.append(f"{i}. 调用 `{attempt['tool']}`: {attempt['error'][:80]}")

        lines.append("")
        lines.append("**请换一个完全不同的方法，不要重复上述尝试。**")

        return "\n".join(lines)

    def _build_guidance(self) -> str:
        """构建行动指导"""
        ctx = self._failure_context

        # 根据错误类型提供具体指导
        specific_guidance = self._get_specific_guidance(ctx.error_type, ctx.tool_name)

        return f"""### 建议的行动步骤

{specific_guidance}

请在回复前先进行思考：

<thinking>
1. 分析失败原因：[在此说明错误原因]
2. 确定新方向：[在此说明你的新计划]
3. 预期结果：[在此说明预期会发生什么]
</thinking>

然后采取行动。"""

    def _get_specific_guidance(self, error_type: str, tool_name: str) -> str:
        """获取针对特定错误类型的指导"""
        guidance_map = {
            "SyntaxError": """
- 检查代码语法：括号匹配、缩进、冒号等
- 特别注意字符串引号是否闭合
- 检查是否使用了 Python 保留字作为变量名""",

            "ImportError": """
- 确认模块是否已安装
- 检查模块名称拼写
- 确认导入路径是否正确""",

            "FileNotFoundError": """
- 确认文件路径是否正确
- 检查当前工作目录
- 使用相对路径时注意基准目录""",

            "PermissionError": """
- 检查文件权限
- 可能需要管理员权限
- 确认文件没有被其他进程锁定""",

            "TypeError": """
- 检查参数类型是否正确
- 确认函数签名
- 注意 None 值处理""",

            "ValueError": """
- 检查输入值的范围和格式
- 确认数据类型转换是否合理
- 验证业务逻辑约束""",

            "TimeoutError": """
- 操作可能耗时过长
- 考虑分解任务或使用异步方式
- 检查是否有死循环""",
        }

        # 工具特定指导
        tool_guidance_map = {
            "terminal": """
- 检查命令语法是否正确
- 确认命令在当前环境中可用
- 考虑使用更简单的命令替代""",
            "write_file": """
- 确认文件路径正确
- 检查是否有写入权限
- 验证内容格式是否正确""",
            "edit_file": """
- 确保 old_string 与文件内容精确匹配（包括缩进）
- 检查 old_string 在文件中是否唯一
- 确认文件路径正确""",
            "read_file": """
- 确认文件存在
- 检查文件路径和权限
- 确认编码格式正确""",
        }

        error_guidance = guidance_map.get(error_type, "- 仔细分析错误消息，找出根本原因")
        tool_guidance = tool_guidance_map.get(tool_name, "")

        return error_guidance + tool_guidance

    def _format_arguments(self, args: dict[str, Any]) -> str:
        """格式化参数"""
        items = []
        for key, value in args.items():
            if isinstance(value, str):
                if len(value) > 50:
                    value = value[:47] + "..."
                items.append(f"{key}=\"{value}\"")
            elif isinstance(value, dict):
                items.append(f"{key}={{...}}")
            elif isinstance(value, list):
                items.append(f"{key}=[...]")
            else:
                items.append(f"{key}={value}")

        return ", ".join(items[:3])  # 最多显示 3 个参数

    def clear(self) -> None:
        """清除失败记录"""
        self._failure_context = None
        self._failure_history.clear()
        logger.debug("Critic 记录已清除")

    def has_recent_failure(self) -> bool:
        """是否有最近的失败"""
        return self._failure_context is not None

    def get_failure_count(self) -> int:
        """获取失败次数"""
        return len(self._failure_history) + (1 if self._failure_context else 0)

    def get_status(self) -> dict[str, Any]:
        """获取状态"""
        return {
            "has_recent_failure": self.has_recent_failure(),
            "failure_count": self.get_failure_count(),
            "current_failure": {
                "tool": self._failure_context.tool_name,
                "error_type": self._failure_context.error_type,
                "attempt": self._failure_context.attempt_number,
            } if self._failure_context else None,
        }


# 导出
__all__ = [
    "FailureContext",
    "CriticConfig",
    "CriticInjector",
]