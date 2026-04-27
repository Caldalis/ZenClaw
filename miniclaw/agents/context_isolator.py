"""
上下文隔离器 — 确保 Subagent 上下文安全

核心职责:
  1. 不将 Master 几万 Token 的历史传给子节点
  2. 只传递 Instruction + 精炼的 Context 摘要 + System Prompt
  3. 让子节点注意力绝对集中

设计原则:
  - 最小化上下文传输
  - 保留关键决策信息
  - 过滤无关历史
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from miniclaw.types.messages import Message
from miniclaw.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class IsolationConfig:
    """上下文隔离配置"""
    max_context_tokens: int = 4000
    """传递给 Subagent 的最大 Token 数"""

    max_history_messages: int = 5
    """最多保留的历史消息条数"""

    include_system_prompt: bool = True
    """是否包含系统提示词"""

    include_relevant_memories: bool = True
    """是否包含相关记忆"""

    summary_max_length: int = 500
    """摘要最大长度"""

    max_dependency_result_length: int = 2000
    """每个依赖任务结果的最大长度"""

    max_dependency_total_length: int = 8000
    """所有依赖任务结果合计的最大长度"""


@dataclass
class IsolatedContext:
    """隔离后的上下文 — 传递给 Subagent 的最小上下文"""
    system_prompt: str
    """系统提示词"""

    instruction: str
    """任务指令"""

    context_summary: str = ""
    """上下文摘要"""

    relevant_files: list[str] = field(default_factory=list)
    """相关文件列表"""

    parent_decisions: list[str] = field(default_factory=list)
    """父 Agent 的关键决策"""

    dependencies_results: dict[str, str] = field(default_factory=dict)
    """依赖任务的执行结果文本摘要"""

    dependencies_files_changed: dict[str, list[str]] = field(default_factory=dict)
    """依赖任务修改/创建的文件列表 (task_id -> [file_paths])"""

    dependencies_status: dict[str, str] = field(default_factory=dict)
    """依赖任务的状态 (task_id -> 'success'/'partial_success'/'failed')"""

    def to_messages(self) -> list[dict[str, str]]:
        """转换为消息列表格式"""
        messages = []

        # 系统提示词
        if self.system_prompt:
            messages.append({
                "role": "system",
                "content": self.system_prompt,
            })

        # 上下文摘要
        if self.context_summary:
            messages.append({
                "role": "system",
                "content": f"[上下文信息]\n{self.context_summary}",
            })

        # 相关文件
        if self.relevant_files:
            files_text = "\n".join(f"- {f}" for f in self.relevant_files)
            messages.append({
                "role": "system",
                "content": f"[相关文件]\n{files_text}",
            })

        # 依赖任务结果 — 包含状态、摘要、修改文件列表
        if self.dependencies_results or self.dependencies_files_changed:
            deps_parts = []
            for task_id, result_text in self.dependencies_results.items():
                dep_lines = [f"**{task_id}**:"]
                if task_id in self.dependencies_status:
                    dep_lines.append(f"  状态: {self.dependencies_status[task_id]}")
                dep_lines.append(f"  {result_text}")
                if task_id in self.dependencies_files_changed and self.dependencies_files_changed[task_id]:
                    files = self.dependencies_files_changed[task_id]
                    dep_lines.append(f"  修改/创建的文件: {', '.join(files)}")
                deps_parts.append("\n".join(dep_lines))

            results_text = "\n\n".join(deps_parts)
            messages.append({
                "role": "system",
                "content": f"[依赖任务结果]\n{results_text}",
            })

        # 任务指令
        messages.append({
            "role": "user",
            "content": self.instruction,
        })

        return messages

    def estimate_tokens(self) -> int:
        """估算 Token 数"""
        total = 0
        for msg in self.to_messages():
            content = msg.get("content", "")
            # 粗略估算：英文 4 字符 = 1 token，中文 2 字符 = 1 token
            ascii_chars = sum(1 for c in content if ord(c) < 128)
            non_ascii = len(content) - ascii_chars
            total += ascii_chars // 4 + non_ascii // 2 + 1
        return total


class ContextIsolator:
    """上下文隔离器

    负责从 Master Agent 的庞大上下文中提取精简信息传递给 Subagent。
    """

    def __init__(self, config: IsolationConfig | None = None):
        self.config = config or IsolationConfig()

    def isolate(
        self,
        instruction: str,
        master_messages: list[Message] = [],
        task_dependencies: dict[str, str] = {},
        dependency_structured_results: dict[str, dict[str, Any]] | None = None,
        relevant_files: list[str] = [],
        context_summary: str = "",
    ) -> IsolatedContext:
        """隔离上下文

        Args:
            instruction: 任务指令（必须）
            master_messages: Master Agent 的历史消息（将被精简）
            task_dependencies: 依赖任务的执行结果文本摘要
            dependency_structured_results: 依赖任务的结构化结果 (task_id -> StructuredResult dict)
                包含 status, files_changed, summary 等完整信息
            relevant_files: 相关文件列表
            context_summary: 可选的上下文摘要

        Returns:
            IsolatedContext: 精简后的上下文
        """
        # 提取关键决策
        decisions = self._extract_key_decisions(master_messages)

        # 构建精简摘要
        summary = self._build_summary(
            messages=master_messages,
            context_summary=context_summary,
            decisions=decisions,
        )

        # 精简依赖结果文本
        trimmed_deps = self._trim_dependencies(task_dependencies)

        # 从结构化结果中提取文件变更和状态信息
        dep_files_changed: dict[str, list[str]] = {}
        dep_status: dict[str, str] = {}

        if dependency_structured_results:
            for task_id, result_data in dependency_structured_results.items():
                if isinstance(result_data, dict):
                    files = result_data.get("files_changed", [])
                    if isinstance(files, list):
                        dep_files_changed[task_id] = [str(f) for f in files]
                    status = result_data.get("status", "")
                    if status:
                        dep_status[task_id] = str(status)

                    # 如果没有文本摘要，用结构化结果的 summary 字段补充
                    if task_id not in trimmed_deps:
                        summary_text = result_data.get("summary", "")
                        if summary_text:
                            trimmed_deps[task_id] = summary_text

        return IsolatedContext(
            system_prompt="",  # 由 SubagentFactory 填充
            instruction=instruction,
            context_summary=summary,
            relevant_files=relevant_files[:10],  # 最多 10 个文件
            parent_decisions=decisions[:5],  # 最多 5 个决策
            dependencies_results=trimmed_deps,
            dependencies_files_changed=dep_files_changed,
            dependencies_status=dep_status,
        )

    def _extract_key_decisions(
        self,
        messages: list[Message],
    ) -> list[str]:
        """从消息中提取关键决策

        策略：
          1. 关注 ASSISTANT 消息中的决策性语句
          2. 忽略常规的工具调用和结果
        """
        decisions = []

        for msg in messages:
            if msg.role.value != "assistant":
                continue

            content = msg.content or ""
            if not content:
                continue

            # 简单的关键词检测（实际可使用更复杂的 NLP）
            decision_keywords = [
                "决定", "选择", "采用", "方案", "策略",
                "will", "decide", "choose", "plan", "strategy",
            ]

            for keyword in decision_keywords:
                if keyword in content.lower():
                    # 提取包含关键词的句子
                    sentences = content.split("。")
                    for sentence in sentences:
                        if keyword in sentence.lower():
                            decisions.append(sentence.strip()[:200])
                            break
                    break

            if len(decisions) >= 5:
                break

        return decisions

    def _build_summary(
        self,
        messages: list[Message],
        context_summary: str,
        decisions: list[str],
    ) -> str:
        """构建精简上下文摘要"""
        parts = []

        # 已有摘要
        if context_summary:
            parts.append(context_summary[:self.config.summary_max_length])

        # 关键决策
        if decisions:
            decisions_text = "\n".join(f"- {d}" for d in decisions[:3])
            parts.append(f"关键决策:\n{decisions_text}")

        # 最近几条消息的摘要
        recent = messages[-self.config.max_history_messages:] if messages else []
        if recent:
            recent_summary = self._summarize_recent_messages(recent)
            parts.append(recent_summary)

        return "\n\n".join(parts)

    def _summarize_recent_messages(
        self,
        messages: list[Message],
    ) -> str:
        """摘要最近的消息"""
        summaries = []

        for msg in messages:
            role = msg.role.value
            content = msg.content or ""

            # 截断长消息
            if len(content) > 100:
                content = content[:97] + "..."

            summaries.append(f"[{role}]: {content}")

        return "最近对话:\n" + "\n".join(summaries)

    def _trim_dependencies(
        self,
        dependencies: dict[str, str],
    ) -> dict[str, str]:
        """精简依赖结果 — 每个结果不超过 max_dependency_result_length，
        所有结果合计不超过 max_dependency_total_length"""
        trimmed = {}
        total_length = 0

        for task_id, result in dependencies.items():
            # 单个结果长度限制
            max_single = self.config.max_dependency_result_length
            if len(result) > max_single:
                result = result[:max_single - 3] + "..."

            # 总长度限制 — 如果累计已超出，截断后续结果
            remaining = self.config.max_dependency_total_length - total_length
            if remaining <= 0:
                break

            if len(result) > remaining:
                result = result[:remaining - 3] + "..."

            trimmed[task_id] = result
            total_length += len(result)

        return trimmed


def create_isolated_context(
    instruction: str,
    master_messages: list[Message] = [],
    task_dependencies: dict[str, str] = {},
    dependency_structured_results: dict[str, dict[str, Any]] | None = None,
    relevant_files: list[str] = [],
    context_summary: str = "",
) -> IsolatedContext:
    """创建隔离上下文（使用默认配置）"""
    isolator = ContextIsolator()
    return isolator.isolate(
        instruction=instruction,
        master_messages=master_messages,
        task_dependencies=task_dependencies,
        dependency_structured_results=dependency_structured_results,
        relevant_files=relevant_files,
        context_summary=context_summary,
    )


# 导出
__all__ = [
    "IsolationConfig",
    "IsolatedContext",
    "ContextIsolator",
    "create_isolated_context",
]