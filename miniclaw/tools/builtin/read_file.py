"""
文件读取工具 — 读取本地文件内容

支持行号标注、偏移/限制读取、自动编码检测。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from miniclaw.tools.base import Tool


class ReadFileTool(Tool):
    """文件读取 — 读取本地文本文件内容，带行号标注"""

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return (
            "读取本地文件内容，输出带行号标注。"
            "支持 offset/limit 参数读取指定行范围，避免读取整个大文件。"
            "默认最多读取 2000 行。"
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
                "offset": {
                    "type": "integer",
                    "description": "起始行号（从 1 开始），默认 1",
                    "default": 1,
                },
                "limit": {
                    "type": "integer",
                    "description": "最大读取行数，默认 2000",
                    "default": 2000,
                },
            },
            "required": ["path"],
        }

    async def execute(self, **kwargs: Any) -> str:
        file_path = kwargs.get("path", "")
        offset = kwargs.get("offset", 1)
        limit = kwargs.get("limit", 2000)

        if not file_path:
            return "错误：请提供文件路径"

        path = Path(file_path)

        if not path.exists():
            return f"文件不存在: {file_path}"

        if not path.is_file():
            return f"不是文件: {file_path}"

        # 读取内容，自动编码检测
        content = _read_with_encoding(path)
        if content is None:
            return "文件编码无法识别（可能为二进制文件）"

        lines = content.splitlines()
        total_lines = len(lines)

        # 行范围切片（offset 从 1 开始）
        start = max(offset - 1, 0)
        end = min(start + limit, total_lines)
        selected = lines[start:end]

        # 带行号标注输出
        numbered = []
        for i, line in enumerate(selected, start=start + 1):
            numbered.append(f"{i}\t{line}")

        result = "\n".join(numbered)

        if end < total_lines:
            result += f"\n... (共 {total_lines} 行，显示 {start + 1}-{end} 行)"

        return result


def _read_with_encoding(path: Path) -> str | None:
    """尝试多种编码读取文件，失败返回 None"""
    encodings = ["utf-8", "gbk", "latin-1"]
    for enc in encodings:
        try:
            return path.read_text(encoding=enc)
        except (UnicodeDecodeError, ValueError):
            continue
    return None