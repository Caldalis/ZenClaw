"""
Subagent 工厂 — 角色实例化与动态 Prompt 融合

核心职责:
  1. 检查请求的角色是否为预设角色
  2. 预设角色：加载预设 System Prompt
  3. 自定义角色：提取 custom_role_prompt 动态构建
  4. 合并工具权限配置
  5. 创建 Subagent 实例配置
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from miniclaw.agents.prompts.role_prompts import (
    PRESET_ROLES,
    build_dynamic_role_prompt,
    get_preset_role,
    is_preset_role,
    merge_role_config,
)
from miniclaw.dispatcher.subagent_registry import SubagentSpec
from miniclaw.types.enums import AgentRole
from miniclaw.types.task_graph import TaskNode
from miniclaw.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SubagentConfig:
    """Subagent 配置 — 创建 Subagent 所需的全部信息"""

    # 基础信息
    role_name: str
    """角色名称"""

    system_prompt: str
    """系统提示词"""

    # 任务信息
    instruction: str
    """任务指令"""

    # 工具权限
    allowed_tools: list[str] = field(default_factory=list)
    """允许使用的工具（空=允许所有）"""

    forbidden_tools: list[str] = field(default_factory=list)
    """禁止使用的工具"""

    # 执行限制
    max_steps: int = 15
    """最大 ReAct 步数"""

    timeout_ms: int = 120000
    """超时时间（毫秒）"""

    # 隔离配置
    requires_worktree: bool = False
    """是否需要 Git Worktree 物理隔离"""

    # 上下文信息
    context_summary: str = ""
    """精炼的上下文摘要"""

    parent_agent_id: str = ""
    """父 Agent ID"""

    session_id: str = ""
    """会话 ID"""

    # 任务追踪
    task_id: str = ""
    """任务 ID"""

    graph_id: str = ""
    """任务图 ID"""

    def to_spec(self) -> SubagentSpec:
        """转换为 SubagentSpec"""
        return SubagentSpec(
            role=AgentRole.GENERIC,  # 动态角色使用 GENERIC
            name=self.role_name,
            description=f"Subagent: {self.role_name}",
            system_prompt=self.system_prompt,
            max_steps=self.max_steps,
            timeout_ms=self.timeout_ms,
            allowed_tools=self.allowed_tools,
            forbidden_tools=self.forbidden_tools,
            requires_worktree=self.requires_worktree,
        )


class SubagentFactory:
    """Subagent 工厂

    负责:
      1. 角色实例化（预设 vs 动态）
      2. System Prompt 融合
      3. 工具权限配置
      4. 上下文准备
    """

    def __init__(self):
        self._preset_roles = PRESET_ROLES

    def create_config(
        self,
        task: TaskNode,
        context_summary: str = "",
        parent_agent_id: str = "",
        session_id: str = "",
        graph_id: str = "",
    ) -> SubagentConfig:
        """创建 Subagent 配置

        流程:
          1. 检查是否为预设角色
          2. 加载或构建 System Prompt
          3. 合并工具权限
          4. 组装完整配置

        Args:
            task: 任务节点（包含 role, instruction, custom_role_prompt 等）
            context_summary: 精炼的上下文摘要
            parent_agent_id: 父 Agent ID
            session_id: 会话 ID
            graph_id: 任务图 ID

        Returns:
            SubagentConfig: 完整的 Subagent 配置
        """
        role_name = task.role
        instruction = task.instruction

        # 检查是否为预设角色
        if is_preset_role(role_name):
            config = self._create_preset_config(
                role_name=role_name,
                instruction=instruction,
                custom_config=task.custom_role_config,
            )
            logger.info("使用预设角色: %s", role_name)
        else:
            config = self._create_dynamic_config(
                role_name=role_name,
                instruction=instruction,
                custom_role_prompt=task.custom_role_prompt,
                custom_config=task.custom_role_config,
            )
            logger.info("使用动态角色: %s", role_name)

        # 填充上下文信息
        config.context_summary = context_summary
        config.parent_agent_id = parent_agent_id
        config.session_id = session_id
        config.task_id = task.id
        config.graph_id = graph_id

        return config

    def _create_preset_config(
        self,
        role_name: str,
        instruction: str,
        custom_config: dict[str, Any] = {},
    ) -> SubagentConfig:
        """创建预设角色配置"""
        preset = get_preset_role(role_name)
        if preset is None:
            raise ValueError(f"预设角色不存在: {role_name}")

        # 合并自定义配置
        merged = merge_role_config(preset, custom_config)

        # 构建完整系统提示词（包含任务指令）
        full_prompt = self._build_full_prompt(
            base_prompt=merged["system_prompt"],
            instruction=instruction,
        )

        return SubagentConfig(
            role_name=role_name,
            system_prompt=full_prompt,
            instruction=instruction,
            allowed_tools=merged.get("allowed_tools", []),
            forbidden_tools=merged.get("forbidden_tools", []),
            max_steps=merged.get("max_steps", 15),
            timeout_ms=merged.get("timeout_ms", 120000),
            requires_worktree=merged.get("requires_worktree", False),
        )

    def _create_dynamic_config(
        self,
        role_name: str,
        instruction: str,
        custom_role_prompt: str | None,
        custom_config: dict[str, Any] = {},
    ) -> SubagentConfig:
        """创建动态角色配置

        当 Master 使用自定义角色时：
          1. 必须有 custom_role_prompt
          2. 动态构建专属 System Prompt
        """
        if not custom_role_prompt:
            logger.warning(
                "动态角色 %s 缺少 custom_role_prompt，使用默认提示词",
                role_name
            )
            custom_role_prompt = f"你是 {role_name}，负责执行分配的任务。"

        # 构建动态系统提示词
        full_prompt = build_dynamic_role_prompt(
            custom_role_prompt=custom_role_prompt,
            task_instruction=instruction,
            allowed_tools=custom_config.get("allowed_tools"),
            forbidden_tools=custom_config.get("forbidden_tools"),
        )

        return SubagentConfig(
            role_name=role_name,
            system_prompt=full_prompt,
            instruction=instruction,
            allowed_tools=custom_config.get("allowed_tools", []),
            forbidden_tools=custom_config.get("forbidden_tools", []),
            max_steps=custom_config.get("max_steps", 15),
            timeout_ms=custom_config.get("timeout_ms", 120000),
            requires_worktree=custom_config.get("requires_worktree", False),
        )

    def _build_full_prompt(
        self,
        base_prompt: str,
        instruction: str,
    ) -> str:
        """构建完整系统提示词

        将基础提示词和具体任务指令融合。
        """
        # 如果基础提示词已包含任务占位符，替换它
        if "{instruction}" in base_prompt:
            return base_prompt.replace("{instruction}", instruction)

        # 否则追加任务指令
        return f"{base_prompt}\n\n## 当前任务\n\n{instruction}"

    def get_available_roles(self) -> list[str]:
        """获取所有可用角色（预设 + 已注册动态角色）"""
        return list(self._preset_roles.keys())

    def validate_role_tools(
        self,
        role_name: str,
        requested_tools: list[str],
    ) -> tuple[list[str], list[str]]:
        """验证角色工具权限

        Args:
            role_name: 角色名称
            requested_tools: 请求使用的工具列表

        Returns:
            (allowed, forbidden): 允许的工具列表和禁止的工具列表
        """
        preset = get_preset_role(role_name)
        if preset is None:
            # 动态角色：默认允许所有
            return requested_tools, []

        allowed_set = set(preset.get("allowed_tools", []))
        forbidden_set = set(preset.get("forbidden_tools", []))

        if not allowed_set:
            # 空允许列表表示允许所有
            allowed_set = set(requested_tools)

        allowed = [t for t in requested_tools if t in allowed_set]
        forbidden = [t for t in requested_tools if t in forbidden_set]

        return allowed, forbidden


# 全局工厂实例
_factory = SubagentFactory()


def create_subagent_config(
    task: TaskNode,
    context_summary: str = "",
    parent_agent_id: str = "",
    session_id: str = "",
    graph_id: str = "",
) -> SubagentConfig:
    """创建 Subagent 配置（使用全局工厂）"""
    return _factory.create_config(
        task=task,
        context_summary=context_summary,
        parent_agent_id=parent_agent_id,
        session_id=session_id,
        graph_id=graph_id,
    )


# 导出
__all__ = [
    "SubagentConfig",
    "SubagentFactory",
    "create_subagent_config",
]