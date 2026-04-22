"""
SKILL.md 懒加载技能管理器

支持两种注入模式:
1. 静态预注入 (Pre-injection): 扫描本地 SKILL.md 并注入到系统提示词
2. 动态懒加载 (Lazy Loading): 通过 load_skill 工具按需加载技能内容
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

from miniclaw.config.settings import PROJECT_ROOT
from miniclaw.utils.logging import get_logger

logger = get_logger(__name__)


class SkillManager:
    """SKILL.md 文件管理器

    扫描本地技能目录，解析 Markdown 的 Frontmatter，
    提供技能清单和懒加载功能。
    """

    def __init__(self, skills_dirs: list[str] | None = None):
        """
        Args:
            skills_dirs: 要扫描的技能目录列表，默认为项目根目录下的 skills
        """
        self._skills_dirs = skills_dirs or [str(PROJECT_ROOT / "skills")]
        self._available_skills: dict[str, dict[str, str]] = {}
        self._scan_all()

    def _scan_all(self) -> None:
        """扫描所有配置的技能目录"""
        for dir_path in self._skills_dirs:
            self._scan_dir(dir_path)

    def _scan_dir(self, skills_dir: str) -> None:
        """扫描单个技能目录，查找 SKILL.md 文件"""
        path = Path(skills_dir)
        if not path.is_dir():
            logger.debug("技能目录不存在: %s", skills_dir)
            return

        for skill_folder in path.iterdir():
            if not skill_folder.is_dir():
                continue
            skill_md_path = skill_folder / "SKILL.md"
            if skill_md_path.is_file():
                self._parse_skill_md(skill_md_path, skill_folder.name)

    def _parse_skill_md(self, skill_path: Path, folder_name: str) -> None:
        """解析单个 SKILL.md 文件，提取 frontmatter 和内容"""
        try:
            with open(skill_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            logger.warning("读取技能文件失败: %s - %s", skill_path, e)
            return

        skill_name, description, markdown_body = self._extract_frontmatter(content, folder_name)

        self._available_skills[skill_name] = {
            "name": skill_name,
            "description": description,
            "content": markdown_body,
        }
        logger.info("加载技能: %s (%s)", skill_name, skill_path)

    def _extract_frontmatter(
        self, content: str, fallback_name: str
    ) -> tuple[str, str, str]:
        """提取 YAML frontmatter 和 markdown body

        Returns:
            (name, description, markdown_body)
        """
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                try:
                    meta = yaml.safe_load(parts[1]) or {}
                    name = meta.get("name", fallback_name)
                    description = meta.get("description", "")
                    markdown_body = parts[2].strip()
                    return name, description, markdown_body
                except yaml.YAMLError as e:
                    logger.warning("YAML 解析失败: %s", e)

        # 无 frontmatter 时，整个文件内容作为 markdown body
        return fallback_name, "", content.strip()

    @property
    def available_skills(self) -> dict[str, dict[str, str]]:
        """返回所有可用技能 {name: {name, description, content}}"""
        return self._available_skills

    def get_skill_content(self, skill_name: str) -> str:
        """获取技能内容，用于懒加载

        Returns:
            用 XML 标签包裹的 markdown 内容，
            让大模型更容易解析。
        """
        skill = self._available_skills.get(skill_name)
        if not skill:
            return f"[Error] Skill '{skill_name}' not found."

        return (
            f'<skill_content name="{skill_name}">\n'
            f"{skill['content']}\n"
            f"</skill_content>"
        )

    def get_load_skill_tool_schema(self) -> dict[str, Any]:
        """生成 load_skill 工具的 OpenAI tool schema

        把所有可用技能的清单告诉大模型，
        让大模型在遇到熟悉的任务时主动加载。
        """
        skill_names = list(self._available_skills.keys())

        if not skill_names:
            return {
                "type": "function",
                "function": {
                    "name": "load_skill",
                    "description": "Load a specialized skill that provides domain-specific instructions.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "skill_name": {
                                "type": "string",
                                "enum": ["__none__"],
                                "description": "No skills available.",
                            }
                        },
                        "required": ["skill_name"],
                    },
                },
            }

        descriptions = "\n".join(
            f"- {s['name']}: {s['description']}"
            for s in self._available_skills.values()
        )

        return {
            "type": "function",
            "function": {
                "name": "load_skill",
                "description": (
                    "Load a specialized skill that provides domain-specific instructions and workflows. "
                    "Use this tool when you recognize that a task matches one of the available skills.\n\n"
                    f"Available skills:\n{descriptions}"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "skill_name": {
                            "type": "string",
                            "enum": skill_names,
                            "description": "The exact name of the skill to load.",
                        }
                    },
                    "required": ["skill_name"],
                },
            },
        }

    def get_preinject_content(self) -> str:
        """生成预注入内容，将所有技能说明拼接到系统提示词

        用于静态预注入模式：在启动时把技能清单追加到 system prompt。
        """
        if not self._available_skills:
            return ""

        lines = [
            "\n\n## Available Skills (SKILL.md)",
            "The following specialized skills are available. Use the `load_skill` tool to load any skill you need.",
            "",
        ]
        for skill in self._available_skills.values():
            lines.append(f"- **{skill['name']}**: {skill['description']}")

        return "\n".join(lines)
