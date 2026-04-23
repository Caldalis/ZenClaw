"""
文件内容搜索工具 — 正则表达式搜索文件内容

支持递归搜索、glob 文件过滤、大小写忽略等。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from miniclaw.tools.base import Tool

DEFAULT_MAX_RESULTS = 100


class GrepTool(Tool):
    """文件内容搜索 — 在文件中搜索匹配正则表达式的内容"""

    @property
    def name(self) -> str:
        return "grep"

    @property
    def description(self) -> str:
        return (
            "在文件内容中搜索正则表达式匹配。"
            "返回匹配行及行号。支持 glob 参数过滤文件类型（如 '*.py'），"
            "ignore_case 参数忽略大小写。递归搜索子目录。"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "正则表达式搜索模式",
                },
                "path": {
                    "type": "string",
                    "description": "搜索起始目录或文件（默认当前工作目录）",
                    "default": ".",
                },
                "glob": {
                    "type": "string",
                    "description": "文件名过滤模式，如 '*.py', '*.{ts,tsx}'",
                },
                "ignore_case": {
                    "type": "boolean",
                    "description": "是否忽略大小写，默认 false",
                    "default": False,
                },
                "max_results": {
                    "type": "integer",
                    "description": "最大返回匹配行数，默认 100",
                    "default": DEFAULT_MAX_RESULTS,
                },
            },
            "required": ["pattern"],
        }

    async def execute(self, **kwargs: Any) -> str:
        pattern = kwargs.get("pattern", "")
        dir_path = kwargs.get("path", ".")
        glob_filter = kwargs.get("glob")
        ignore_case = kwargs.get("ignore_case", False)
        max_results = kwargs.get("max_results", DEFAULT_MAX_RESULTS)

        if not pattern:
            return "错误：请提供搜索模式"

        # 安全检查
        base_dir = Path.cwd().resolve()
        target = Path(dir_path).resolve()

        try:
            target.relative_to(base_dir)
        except ValueError:
            return f"错误：安全限制，不允许访问当前目录之外的路径。"

        if not target.exists():
            return f"路径不存在: {dir_path}"

        # 编译正则表达式
        flags = re.IGNORECASE if ignore_case else 0
        try:
            regex = re.compile(pattern, flags)
        except re.error as e:
            return f"错误：正则表达式无效 - {e}"

        # 搜索文件
        files_to_search = _collect_files(target, glob_filter)
        if not files_to_search:
            return "没有可搜索的文件"

        matches = []
        for file_path in files_to_search:
            file_matches = _search_file(file_path, regex)
            matches.extend(file_matches)
            if len(matches) >= max_results:
                break

        # 截断结果
        total_matches = len(matches)
        matches = matches[:max_results]

        if not matches:
            return f"没有匹配 '{pattern}' 的内容"

        # 格式化输出
        lines = []
        for rel_path, line_num, line_text in matches:
            lines.append(f"{rel_path}:{line_num}: {line_text}")

        result = "\n".join(lines)
        if total_matches > max_results:
            result += f"\n... (共 {total_matches} 处匹配，显示前 {max_results} 处)"

        return result


def _collect_files(target: Path, glob_filter: str | None) -> list[Path]:
    """收集要搜索的文件列表"""
    if target.is_file():
        return [target]

    # 递归收集目录下的文件
    if glob_filter:
        files = sorted(target.glob(f"**/{glob_filter}"))
    else:
        # 搜索所有文本文件（排除常见二进制和隐藏目录）
        files = []
        for f in target.rglob("*"):
            if f.is_file() and not _is_binary_path(f):
                files.append(f)
        files.sort()

    # 过滤掉二进制文件
    return [f for f in files if f.is_file() and not _looks_binary(f)]


def _is_binary_path(path: Path) -> bool:
    """判断路径是否在隐藏或不应搜索的目录中"""
    hidden_dirs = {".git", "__pycache__", "node_modules", ".venv", "venv", ".idea"}
    for part in path.parts:
        if part in hidden_dirs:
            return True
    return False


def _looks_binary(path: Path) -> bool:
    """简单判断文件是否看起来是二进制文件"""
    binary_exts = {
        ".pyc", ".so", ".dll", ".exe", ".bin", ".dat",
        ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico",
        ".zip", ".tar", ".gz", ".rar", ".7z",
        ".pdf", ".doc", ".docx", ".xls", ".xlsx",
        ".mp3", ".mp4", ".wav", ".avi", ".mkv",
    }
    return path.suffix.lower() in binary_exts


def _search_file(file_path: Path, regex: re.Pattern) -> list[tuple[str, int, str]]:
    """在单个文件中搜索匹配"""
    # 读取文件（尝试多种编码）
    content = _read_with_encoding(file_path)
    if content is None:
        return []

    matches = []
    # 计算基础路径用于相对路径显示
    try:
        base = Path.cwd().resolve()
        rel = str(file_path.relative_to(base))
    except ValueError:
        rel = str(file_path)

    for line_num, line in enumerate(content.splitlines(), 1):
        if regex.search(line):
            # 截断过长行
            display_line = line[:300] if len(line) > 300 else line
            matches.append((rel, line_num, display_line))

    return matches


def _read_with_encoding(path: Path) -> str | None:
    encodings = ["utf-8", "gbk", "latin-1"]
    for enc in encodings:
        try:
            return path.read_text(encoding=enc)
        except (UnicodeDecodeError, ValueError):
            continue
    return None