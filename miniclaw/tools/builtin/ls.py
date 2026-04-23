"""
目录列表工具 — 列出目录内容

支持递归列表，最大深度 4 层防止 runaway。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from miniclaw.tools.base import Tool

MAX_DEPTH = 4


class LsTool(Tool):
    """目录列表 — 列出指定目录的文件和子目录"""

    @property
    def name(self) -> str:
        return "ls"

    @property
    def description(self) -> str:
        return (
            "列出目录内容。显示文件和子目录，带类型标注（D=目录, F=文件）。"
            "recursive=true 时递归列出子目录内容，最大深度 4 层。"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "目录路径（默认当前工作目录）",
                    "default": ".",
                },
                "recursive": {
                    "type": "boolean",
                    "description": "是否递归列出子目录，默认 false",
                    "default": False,
                },
            },
            "required": [],
        }

    async def execute(self, **kwargs: Any) -> str:
        dir_path = kwargs.get("path", ".")
        recursive = kwargs.get("recursive", False)

        base_dir = Path.cwd().resolve()
        target = Path(dir_path).resolve()

        # 安全检查
        try:
            target.relative_to(base_dir)
        except ValueError:
            return f"错误：安全限制，不允许访问当前目录之外的路径。"

        if not target.exists():
            return f"目录不存在: {dir_path}"

        if not target.is_dir():
            return f"不是目录: {dir_path}"

        if recursive:
            return _list_recursive(target, base_dir, max_depth=MAX_DEPTH)
        else:
            return _list_flat(target, base_dir)


def _list_flat(target: Path, base_dir: Path) -> str:
    """列出目录内容（非递归）"""
    entries = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    lines = []
    for entry in entries:
        rel = entry.relative_to(base_dir)
        if entry.is_dir():
            lines.append(f"D  {rel}/")
        else:
            lines.append(f"F  {rel}")
    return "\n".join(lines) if lines else "(空目录)"


def _list_recursive(target: Path, base_dir: Path, max_depth: int, current_depth: int = 0) -> str:
    """递归列出目录内容（树状）"""
    if current_depth > max_depth:
        return "(深度限制，不再展开)"

    entries = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    lines = []
    indent = "  " * current_depth

    for entry in entries:
        if entry.is_dir():
            lines.append(f"{indent}D  {entry.name}/")
            sub = _list_recursive(entry, base_dir, max_depth, current_depth + 1)
            if sub != "(空目录)":
                lines.append(sub)
        else:
            lines.append(f"{indent}F  {entry.name}")

    return "\n".join(lines) if lines else "(空目录)"