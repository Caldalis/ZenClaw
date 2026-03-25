"""
文件写入技能 — 内置技能
允许在当前目录及子目录下创建/修改文件。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from miniclaw.tools.base import Skill


class FileWriterSkill(Skill):
    """文件写入 — 在当前目录下创建或修改文件"""

    @property
    def name(self) -> str:
        return "file_writer"

    @property
    def description(self) -> str:
        return "在当前目录及子目录下创建或修改文件内容。如果文件不存在则创建，如果存在则覆盖。"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "文件相对路径（必须在当前目录下）",
                },
                "content": {
                    "type": "string",
                    "description": "要写入的文件内容",
                },
            },
            "required": ["path", "content"],
        }

    async def execute(self, **kwargs: Any) -> str:
        file_path = kwargs.get("path", "")
        content = kwargs.get("content", "")

        if not file_path:
            return "错误：请提供文件路径"

        base_dir = Path.cwd().resolve()
        target_path = Path(file_path).resolve()

        # 安全检查：只能修改当前目录及子目录下的文件
        try:
            target_path.relative_to(base_dir)
        except ValueError:
            return f"错误：安全限制，不允许操作当前目录（{base_dir}）之外的文件。"

        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(content, encoding="utf-8")
            return f"成功：文件 {target_path.relative_to(base_dir)} 写入完成。"
        except Exception as e:
            return f"错误：写入文件失败 - {str(e)}"
