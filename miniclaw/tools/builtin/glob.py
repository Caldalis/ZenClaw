"""
文件模式匹配工具 — 按 glob 模式查找文件

支持通配符模式如 **/*.py, src/**/*.ts 等。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from miniclaw.tools.base import Tool

MAX_RESULTS = 1000


class GlobTool(Tool):
    """文件模式匹配 — 按 glob 模式查找文件"""

    @property
    def name(self) -> str:
        return "glob"

    @property
    def description(self) -> str:
        return (
            "按 glob 模式查找文件。"
            "支持 **/*.py, src/**/*.ts 等模式。"
            "返回匹配文件的路径列表（相对于工作目录），最多 1000 个结果。"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "glob 模式，如 '**/*.py', 'src/**/*.ts'",
                },
                "path": {
                    "type": "string",
                    "description": "搜索起始目录（默认当前工作目录）",
                    "default": ".",
                },
            },
            "required": ["pattern"],
        }

    async def execute(self, **kwargs: Any) -> str:
        pattern = kwargs.get("pattern", "")
        dir_path = kwargs.get("path", ".")

        if not pattern:
            return "错误：请提供 glob 模式"

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

        # 执行 glob 搜索
        matches = sorted(target.glob(pattern))
        # 过滤出文件（排除目录）
        file_matches = [m for m in matches if m.is_file()]

        if not file_matches:
            return f"没有匹配 '{pattern}' 的文件"

        # 截断结果
        if len(file_matches) > MAX_RESULTS:
            file_matches = file_matches[:MAX_RESULTS]
            truncated = f"\n... (共 {len(matches)} 个文件，显示前 {MAX_RESULTS} 个)"
        else:
            truncated = ""

        # 输出相对路径
        lines = [str(m.relative_to(base_dir)) for m in file_matches]
        return "\n".join(lines) + truncated