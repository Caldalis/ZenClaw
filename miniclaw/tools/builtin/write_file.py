"""
文件写入工具 — 创建或覆盖文件

支持自动创建父目录，返回写入确认信息。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from miniclaw.tools.base import Tool


class WriteFileTool(Tool):
    """文件写入 — 创建新文件或覆盖现有文件"""

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return (
            "写入文件内容。如果文件不存在则创建，如果存在则覆盖。"
            "自动创建所需的父目录。"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "文件路径（相对或绝对路径）",
                },
                "content": {
                    "type": "string",
                    "description": "要写入的文件内容",
                },
                "create_dirs": {
                    "type": "boolean",
                    "description": "是否自动创建父目录，默认 true",
                    "default": True,
                },
            },
            "required": ["path", "content"],
        }

    async def execute(self, **kwargs: Any) -> str:
        file_path = kwargs.get("path", "")
        content = kwargs.get("content", "")
        create_dirs = kwargs.get("create_dirs", True)

        if not file_path:
            return "错误：请提供文件路径"

        base_dir = Path.cwd().resolve()
        target_path = Path(file_path).resolve()

        # 安全检查：只能写入当前目录及子目录下的文件
        try:
            target_path.relative_to(base_dir)
        except ValueError:
            return f"错误：安全限制，不允许操作当前目录（{base_dir}）之外的文件。"

        try:
            if create_dirs:
                target_path.parent.mkdir(parents=True, exist_ok=True)

            target_path.write_text(content, encoding="utf-8")
            line_count = len(content.splitlines())
            rel_path = target_path.relative_to(base_dir)
            return f"文件已写入: {rel_path} ({line_count} 行)"
        except Exception as e:
            return f"错误：写入文件失败 - {e}"