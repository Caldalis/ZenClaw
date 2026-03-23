"""
文件读取技能 — 内置技能示例

读取本地文件内容。演示 Skill 如何访问本地资源。
限制了可读取的文件大小和类型，防止安全问题。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from miniclaw.tools.base import Skill


# 安全限制
MAX_FILE_SIZE = 100 * 1024  # 100KB
ALLOWED_EXTENSIONS = {
    ".txt", ".md", ".py", ".js", ".ts", ".json", ".yaml", ".yml",
    ".toml", ".cfg", ".ini", ".csv", ".xml", ".html", ".css",
    ".sh", ".bash", ".log", ".env.example",
}


class FileReaderSkill(Skill):
    """文件读取 — 读取本地文本文件内容"""

    @property
    def name(self) -> str:
        return "file_reader"

    @property
    def description(self) -> str:
        return "读取本地文本文件的内容。支持常见文本格式（.txt, .md, .py, .json 等）。"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "文件路径（相对路径或绝对路径）",
                },
                "max_lines": {
                    "type": "integer",
                    "description": "最大读取行数（默认 100）",
                    "default": 100,
                },
            },
            "required": ["path"],
        }

    async def execute(self, **kwargs: Any) -> str:
        file_path = kwargs.get("path", "")
        max_lines = kwargs.get("max_lines", 100)

        if not file_path:
            return "请提供文件路径"

        path = Path(file_path)

        # 安全检查
        if not path.exists():
            return f"文件不存在: {file_path}"

        if not path.is_file():
            return f"不是文件: {file_path}"

        if path.suffix.lower() not in ALLOWED_EXTENSIONS:
            return f"不支持的文件类型: {path.suffix}（支持: {', '.join(sorted(ALLOWED_EXTENSIONS))}）"

        size = path.stat().st_size
        if size > MAX_FILE_SIZE:
            return f"文件过大: {size / 1024:.1f}KB（限制: {MAX_FILE_SIZE / 1024:.0f}KB）"

        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                content = path.read_text(encoding="gbk")
            except Exception:
                return "文件编码无法识别"

        lines = content.splitlines()
        total_lines = len(lines)

        if total_lines > max_lines:
            lines = lines[:max_lines]
            truncated = f"\n... (截断，共 {total_lines} 行，显示前 {max_lines} 行)"
        else:
            truncated = ""

        return f"📄 {path.name} ({total_lines} 行)\n\n" + "\n".join(lines) + truncated
