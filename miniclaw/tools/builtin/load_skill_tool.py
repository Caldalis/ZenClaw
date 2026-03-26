"""
懒加载技能 (LoadSkillSkill)

这是一个特殊的内置技能：
当 Agent 遇到不熟悉的任务时，主动加载对应的 SKILL.md 内容。

它通过 SkillManager 访问所有已扫描的 SKILL.md 文件。
"""

from __future__ import annotations

from typing import Any

from miniclaw.tools.base import Tool
from miniclaw.tools.skill_md_loader import SkillManager

# 全局 SkillManager 实例（由 bootstrap 注入）
_skill_manager: SkillManager | None = None


def set_skill_manager(manager: SkillManager) -> None:
    """由 bootstrap 注入全局 SkillManager 实例"""
    global _skill_manager
    _skill_manager = manager


class LoadSkillTool(Tool):
    """懒加载技能 — 通过 SKILL.md 文件提供领域专用工作流

    当 Agent 识别到任务匹配某个技能时，调用此工具加载详细的 Markdown 内容。
    加载后的内容以 <skill_content> 标签包裹，便于 Agent 解析和遵守。
    """

    @property
    def name(self) -> str:
        return "load_skill"

    @property
    def description(self) -> str:
        if _skill_manager is None:
            return "Load a specialized skill (no skills available)."
        schemas = _skill_manager.get_load_skill_tool_schema()
        return schemas["function"]["description"]

    @property
    def parameters(self) -> dict[str, Any]:
        if _skill_manager is None:
            return {
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "enum": [],
                        "description": "No skills available.",
                    }
                },
                "required": ["skill_name"],
            }
        schema = _skill_manager.get_load_skill_tool_schema()
        return schema["function"]["parameters"]

    async def execute(self, skill_name: str, **kwargs: Any) -> str:
        """加载指定技能的内容并返回给 Agent"""
        if _skill_manager is None:
            return "[Error] SkillManager not initialized."

        content = _skill_manager.get_skill_content(skill_name)
        return content
