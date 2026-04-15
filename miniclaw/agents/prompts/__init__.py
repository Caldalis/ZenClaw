"""
Agent 提示词模块

包含各角色的系统提示词定义。
"""

from miniclaw.agents.prompts.master_prompt import (
    MASTER_AGENT_PROMPT,
    THINKING_TEMPLATE,
    format_thinking_block,
)
from miniclaw.agents.prompts.role_prompts import (
    PRESET_ROLES,
    build_dynamic_role_prompt,
    get_preset_role,
    is_preset_role,
    list_preset_roles,
    merge_role_config,
)

__all__ = [
    "MASTER_AGENT_PROMPT",
    "THINKING_TEMPLATE",
    "format_thinking_block",
    # 预设角色
    "PRESET_ROLES",
    "get_preset_role",
    "is_preset_role",
    "list_preset_roles",
    "build_dynamic_role_prompt",
    "merge_role_config",
]