"""
Worktree 感知的工具 — 文件操作和命令执行限定在 worktree 边界内

核心安全原则:
  1. 所有文件操作必须在 worktree 目录内
  2. 禁止访问 worktree 外的文件
  3. 路径规范化防止路径穿越攻击

工具集:
  - read_file: 读取文件内容（带行号标注、偏移/限制）
  - write_file: 创建或覆盖文件
  - edit_file: 精确查找替换编辑
  - ls: 列出目录内容
  - glob: 按 glob 模式查找文件
  - grep: 在文件内容中搜索正则表达式
  - terminal: 执行终端命令
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

from miniclaw.tools.base import Tool
from miniclaw.utils.logging import get_logger

logger = get_logger(__name__)

# --- Shared Constants ---
MAX_DEPTH_LS = 4
MAX_GLOB_RESULTS = 1000
MAX_GREP_RESULTS = 100
BINARY_EXTENSIONS = {
    ".pyc", ".so", ".dll", ".exe", ".bin", ".dat",
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico",
    ".zip", ".tar", ".gz", ".rar", ".7z",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx",
    ".mp3", ".mp4", ".wav", ".avi", ".mkv",
}
HIDDEN_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", ".idea"}


# --- Security ---
class PathSecurityError(Exception):
    """路径安全错误"""
    pass


def validate_path(
    target_path: Path,
    allowed_root: Path,
) -> Path:
    """验证路径安全性

    确保目标路径在允许的根目录内，防止路径穿越攻击。

    Args:
        target_path: 目标路径
        allowed_root: 允许的根目录

    Returns:
        解析后的绝对路径

    Raises:
        PathSecurityError: 路径不在允许范围内
    """
    if not target_path.is_absolute():
        target_path = allowed_root / target_path

    try:
        resolved = target_path.resolve()
    except Exception as e:
        raise PathSecurityError(f"路径解析失败: {target_path}") from e

    try:
        resolved.relative_to(allowed_root.resolve())
    except ValueError:
        raise PathSecurityError(
            f"路径不在工作目录内: {resolved} (allowed: {allowed_root})"
        )

    return resolved


def _read_with_encoding(path: Path) -> str | None:
    """尝试多种编码读取文件"""
    encodings = ["utf-8", "gbk", "latin-1"]
    for enc in encodings:
        try:
            return path.read_text(encoding=enc)
        except (UnicodeDecodeError, ValueError):
            continue
    return None


# --- Isolated Tools ---


class IsolatedReadFileTool(Tool):
    """隔离的文件读取工具 — 带行号标注、偏移/限制"""

    def __init__(self, worktree_root: Path | None = None):
        self._worktree_root = worktree_root

    def set_worktree_root(self, root: Path) -> None:
        self._worktree_root = root

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return (
            "读取文件内容，输出带行号标注。"
            "支持 offset/limit 参数读取指定行范围。"
            "路径必须在当前工作目录内。"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "文件路径（相对于工作目录或绝对路径）",
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
        if self._worktree_root is None:
            return "错误: 工作目录未设置"

        file_path = Path(kwargs.get("path", ""))
        offset = kwargs.get("offset", 1)
        limit = kwargs.get("limit", 2000)

        try:
            safe_path = validate_path(file_path, self._worktree_root)

            if not safe_path.exists():
                return f"文件不存在: {file_path}"

            if not safe_path.is_file():
                return f"不是文件: {file_path}"

            content = _read_with_encoding(safe_path)
            if content is None:
                return "文件编码无法识别（可能为二进制文件）"

            lines = content.splitlines()
            total_lines = len(lines)

            start = max(offset - 1, 0)
            end = min(start + limit, total_lines)
            selected = lines[start:end]

            numbered = []
            for i, line in enumerate(selected, start=start + 1):
                numbered.append(f"{i}\t{line}")

            result = "\n".join(numbered)
            if end < total_lines:
                result += f"\n... (共 {total_lines} 行，显示 {start + 1}-{end} 行)"

            return result

        except PathSecurityError as e:
            logger.warning("路径安全违规: %s", e)
            return f"安全错误: {e}"
        except Exception as e:
            logger.error("读取文件失败: %s", e)
            return f"读取错误: {e}"


class IsolatedWriteFileTool(Tool):
    """隔离的文件写入工具 — 创建或覆盖文件"""

    def __init__(self, worktree_root: Path | None = None):
        self._worktree_root = worktree_root

    def set_worktree_root(self, root: Path) -> None:
        self._worktree_root = root

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return (
            "写入文件内容。如果文件不存在则创建，如果存在则覆盖。"
            "自动创建所需的父目录。路径必须在当前工作目录内。"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "文件路径（相对于工作目录或绝对路径）",
                },
                "content": {
                    "type": "string",
                    "description": "要写入的内容",
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
        if self._worktree_root is None:
            return "错误: 工作目录未设置"

        file_path = Path(kwargs.get("path", ""))
        content = kwargs.get("content", "")
        create_dirs = kwargs.get("create_dirs", True)

        try:
            safe_path = validate_path(file_path, self._worktree_root)

            if create_dirs:
                safe_path.parent.mkdir(parents=True, exist_ok=True)

            # 异步写入
            await _async_write(safe_path, content)

            line_count = len(content.splitlines())
            return f"文件已写入: {file_path} ({line_count} 行)"

        except PathSecurityError as e:
            logger.warning("路径安全违规: %s", e)
            return f"安全错误: {e}"
        except Exception as e:
            logger.error("写入文件失败: %s", e)
            return f"写入错误: {e}"


class IsolatedEditFileTool(Tool):
    """隔离的文件编辑工具 — 精确查找替换"""

    def __init__(self, worktree_root: Path | None = None):
        self._worktree_root = worktree_root

    def set_worktree_root(self, root: Path) -> None:
        self._worktree_root = root

    @property
    def name(self) -> str:
        return "edit_file"

    @property
    def description(self) -> str:
        return (
            "通过查找替换精确编辑文件。"
            "old_string 必须精确匹配文件内容（包括缩进）。"
            "默认只替换第一个匹配；使用 replace_all=true 替换所有。"
            "路径必须在当前工作目录内。"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "文件路径",
                },
                "old_string": {
                    "type": "string",
                    "description": "要替换的原始文本（必须精确匹配）",
                },
                "new_string": {
                    "type": "string",
                    "description": "替换后的新文本",
                },
                "replace_all": {
                    "type": "boolean",
                    "description": "是否替换所有匹配，默认 false",
                    "default": False,
                },
            },
            "required": ["path", "old_string", "new_string"],
        }

    async def execute(self, **kwargs: Any) -> str:
        if self._worktree_root is None:
            return "错误: 工作目录未设置"

        file_path = Path(kwargs.get("path", ""))
        old_string = kwargs.get("old_string", "")
        new_string = kwargs.get("new_string", "")
        replace_all = kwargs.get("replace_all", False)

        if not old_string:
            return "错误: old_string 不能为空"

        try:
            safe_path = validate_path(file_path, self._worktree_root)

            if not safe_path.exists():
                return f"文件不存在: {file_path}"

            if not safe_path.is_file():
                return f"不是文件: {file_path}"

            content = _read_with_encoding(safe_path)
            if content is None:
                return "错误: 文件编码无法识别"

            count = content.count(old_string)
            if count == 0:
                return f"错误: 未找到 old_string，文件中没有精确匹配的内容。请确保 old_string 与文件内容精确匹配（包括缩进）。"

            if count > 1 and not replace_all:
                return (
                    f"错误: old_string 在文件中出现 {count} 次，不是唯一的。"
                    f"请提供更多上下文使 old_string 唯一，或使用 replace_all=true。"
                )

            if replace_all:
                new_content = content.replace(old_string, new_string)
                replaced_count = count
            else:
                new_content = content.replace(old_string, new_string, 1)
                replaced_count = 1

            await _async_write(safe_path, new_content)
            return f"已编辑: {file_path} (替换了 {replaced_count} 处)"

        except PathSecurityError as e:
            logger.warning("路径安全违规: %s", e)
            return f"安全错误: {e}"
        except Exception as e:
            logger.error("编辑文件失败: %s", e)
            return f"编辑错误: {e}"


class IsolatedLsTool(Tool):
    """隔离的目录列表工具"""

    def __init__(self, worktree_root: Path | None = None):
        self._worktree_root = worktree_root

    def set_worktree_root(self, root: Path) -> None:
        self._worktree_root = root

    @property
    def name(self) -> str:
        return "ls"

    @property
    def description(self) -> str:
        return (
            "列出目录内容。D=目录, F=文件。"
            "recursive=true 时递归列出子目录，最大深度 4 层。"
            "路径必须在当前工作目录内。"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "目录路径（默认工作目录根）",
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
        if self._worktree_root is None:
            return "错误: 工作目录未设置"

        dir_path = Path(kwargs.get("path", "."))
        recursive = kwargs.get("recursive", False)

        try:
            safe_path = validate_path(dir_path, self._worktree_root)

            if not safe_path.exists():
                return f"目录不存在: {dir_path}"

            if not safe_path.is_dir():
                return f"不是目录: {dir_path}"

            if recursive:
                return _list_recursive(safe_path, MAX_DEPTH_LS)
            else:
                return _list_flat(safe_path)

        except PathSecurityError as e:
            logger.warning("路径安全违规: %s", e)
            return f"安全错误: {e}"
        except Exception as e:
            logger.error("列出目录失败: %s", e)
            return f"错误: {e}"


class IsolatedGlobTool(Tool):
    """隔离的文件模式匹配工具"""

    def __init__(self, worktree_root: Path | None = None):
        self._worktree_root = worktree_root

    def set_worktree_root(self, root: Path) -> None:
        self._worktree_root = root

    @property
    def name(self) -> str:
        return "glob"

    @property
    def description(self) -> str:
        return (
            "按 glob 模式查找文件（如 **/*.py）。"
            "返回匹配文件路径列表，最多 1000 个。"
            "路径必须在当前工作目录内。"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "glob 模式",
                },
                "path": {
                    "type": "string",
                    "description": "搜索起始目录（默认工作目录根）",
                    "default": ".",
                },
            },
            "required": ["pattern"],
        }

    async def execute(self, **kwargs: Any) -> str:
        if self._worktree_root is None:
            return "错误: 工作目录未设置"

        pattern = kwargs.get("pattern", "")
        dir_path = Path(kwargs.get("path", "."))

        if not pattern:
            return "错误: 请提供 glob 模式"

        try:
            safe_path = validate_path(dir_path, self._worktree_root)

            if not safe_path.exists():
                return f"目录不存在: {dir_path}"

            if not safe_path.is_dir():
                return f"不是目录: {dir_path}"

            matches = sorted(safe_path.glob(pattern))
            file_matches = [m for m in matches if m.is_file()]

            if not file_matches:
                return f"没有匹配 '{pattern}' 的文件"

            if len(file_matches) > MAX_GLOB_RESULTS:
                file_matches = file_matches[:MAX_GLOB_RESULTS]
                truncated = f"\n... (显示前 {MAX_GLOB_RESULTS} 个)"
            else:
                truncated = ""

            lines = [str(m.relative_to(self._worktree_root)) for m in file_matches]
            return "\n".join(lines) + truncated

        except PathSecurityError as e:
            logger.warning("路径安全违规: %s", e)
            return f"安全错误: {e}"
        except Exception as e:
            logger.error("glob 搜索失败: %s", e)
            return f"错误: {e}"


class IsolatedGrepTool(Tool):
    """隔离的文件内容搜索工具"""

    def __init__(self, worktree_root: Path | None = None):
        self._worktree_root = worktree_root

    def set_worktree_root(self, root: Path) -> None:
        self._worktree_root = root

    @property
    def name(self) -> str:
        return "grep"

    @property
    def description(self) -> str:
        return (
            "在文件内容中搜索正则表达式。"
            "返回匹配行及行号。支持 glob 文件过滤和大小写忽略。"
            "路径必须在当前工作目录内。"
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
                    "description": "搜索起始目录或文件（默认工作目录根）",
                    "default": ".",
                },
                "glob": {
                    "type": "string",
                    "description": "文件名过滤模式，如 '*.py'",
                },
                "ignore_case": {
                    "type": "boolean",
                    "description": "是否忽略大小写，默认 false",
                    "default": False,
                },
                "max_results": {
                    "type": "integer",
                    "description": "最大返回匹配行数，默认 100",
                    "default": MAX_GREP_RESULTS,
                },
            },
            "required": ["pattern"],
        }

    async def execute(self, **kwargs: Any) -> str:
        if self._worktree_root is None:
            return "错误: 工作目录未设置"

        pattern = kwargs.get("pattern", "")
        dir_path = Path(kwargs.get("path", "."))
        glob_filter = kwargs.get("glob")
        ignore_case = kwargs.get("ignore_case", False)
        max_results = kwargs.get("max_results", MAX_GREP_RESULTS)

        if not pattern:
            return "错误: 请提供搜索模式"

        try:
            safe_path = validate_path(dir_path, self._worktree_root)

            if not safe_path.exists():
                return f"路径不存在: {dir_path}"

            flags = re.IGNORECASE if ignore_case else 0
            try:
                regex = re.compile(pattern, flags)
            except re.error as e:
                return f"错误: 正则表达式无效 - {e}"

            files = _collect_isolated_files(safe_path, glob_filter)
            if not files:
                return "没有可搜索的文件"

            matches = []
            for f in files:
                file_matches = _search_isolated_file(f, regex, self._worktree_root)
                matches.extend(file_matches)
                if len(matches) >= max_results:
                    break

            total_matches = len(matches)
            matches = matches[:max_results]

            if not matches:
                return f"没有匹配 '{pattern}' 的内容"

            lines = []
            for rel_path, line_num, line_text in matches:
                lines.append(f"{rel_path}:{line_num}: {line_text}")

            result = "\n".join(lines)
            if total_matches > max_results:
                result += f"\n... (共 {total_matches} 处匹配，显示前 {max_results} 处)"

            return result

        except PathSecurityError as e:
            logger.warning("路径安全违规: %s", e)
            return f"安全错误: {e}"
        except Exception as e:
            logger.error("grep 搜索失败: %s", e)
            return f"错误: {e}"


class IsolatedTerminalTool(Tool):
    """隔离的终端命令工具 — 在 worktree 目录内执行"""

    def __init__(self, worktree_root: Path | None = None):
        self._worktree_root = worktree_root
        self._timeout = 60000

    def set_worktree_root(self, root: Path) -> None:
        self._worktree_root = root

    @property
    def name(self) -> str:
        return "terminal"

    @property
    def description(self) -> str:
        return "执行终端命令。命令在当前工作目录内执行。请谨慎使用。"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "要执行的命令",
                },
                "timeout": {
                    "type": "integer",
                    "description": "超时时间（毫秒），默认 60000",
                },
            },
            "required": ["command"],
        }

    async def execute(self, **kwargs: Any) -> str:
        if self._worktree_root is None:
            return "错误: 工作目录未设置"

        command = kwargs.get("command", "")
        timeout = kwargs.get("timeout", self._timeout)

        if not command:
            return "错误: 命令为空"

        if self._is_dangerous_command(command):
            return "错误: 该命令被禁止执行（安全原因）"

        try:
            result = await self._run_command(command, timeout)

            output = f"命令: {command}\n"
            output += f"退出码: {result['exit_code']}\n"

            if result["stdout"]:
                output += f"输出:\n{result['stdout']}\n"

            if result["stderr"]:
                output += f"错误:\n{result['stderr']}\n"

            return output

        except asyncio.TimeoutError:
            return f"错误: 命令执行超时 ({timeout}ms)"

        except Exception as e:
            logger.error("执行命令失败: %s", e)
            return f"执行错误: {e}"

    def _is_dangerous_command(self, command: str) -> bool:
        """检查是否为危险命令"""
        dangerous_patterns = [
            "rm -rf /", "rm -rf ~", "rm -rf *",
            ":(){ :|:& };:", "mkfs", "dd if=",
            "> /dev/sd", "chmod -R 777 /",
            "curl | bash", "wget | bash",
        ]
        command_lower = command.lower()
        return any(p.lower() in command_lower for p in dangerous_patterns)

    async def _run_command(self, command: str, timeout: int) -> dict[str, Any]:
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self._worktree_root),
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout / 1000,
            )
            return {
                "exit_code": process.returncode or 0,
                "stdout": stdout.decode("utf-8", errors="replace"),
                "stderr": stderr.decode("utf-8", errors="replace"),
            }
        except asyncio.TimeoutError:
            process.kill()
            raise


# --- Helper functions (non-async, used by ls/glob/grep) ---


def _list_flat(target: Path) -> str:
    entries = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    lines = []
    for entry in entries:
        if entry.is_dir():
            lines.append(f"D  {entry.name}/")
        else:
            lines.append(f"F  {entry.name}")
    return "\n".join(lines) if lines else "(空目录)"


def _list_recursive(target: Path, max_depth: int, current_depth: int = 0) -> str:
    if current_depth > max_depth:
        return "(深度限制，不再展开)"
    entries = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    lines = []
    indent = "  " * current_depth
    for entry in entries:
        if entry.is_dir():
            lines.append(f"{indent}D  {entry.name}/")
            sub = _list_recursive(entry, max_depth, current_depth + 1)
            if sub != "(空目录)":
                lines.append(sub)
        else:
            lines.append(f"{indent}F  {entry.name}")
    return "\n".join(lines) if lines else "(空目录)"


def _collect_isolated_files(target: Path, glob_filter: str | None) -> list[Path]:
    """收集要搜索的文件列表"""
    if target.is_file():
        return [target]

    if glob_filter:
        files = sorted(target.glob(f"**/{glob_filter}"))
    else:
        files = []
        for f in target.rglob("*"):
            if f.is_file() and not _is_hidden_path(f):
                files.append(f)
        files.sort()

    return [f for f in files if f.is_file() and f.suffix.lower() not in BINARY_EXTENSIONS]


def _is_hidden_path(path: Path) -> bool:
    for part in path.parts:
        if part in HIDDEN_DIRS:
            return True
    return False


def _search_isolated_file(
    file_path: Path,
    regex: re.Pattern,
    worktree_root: Path,
) -> list[tuple[str, int, str]]:
    content = _read_with_encoding(file_path)
    if content is None:
        return []

    try:
        rel = str(file_path.relative_to(worktree_root))
    except ValueError:
        rel = str(file_path)

    matches = []
    for line_num, line in enumerate(content.splitlines(), 1):
        if regex.search(line):
            display_line = line[:300] if len(line) > 300 else line
            matches.append((rel, line_num, display_line))
    return matches


async def _async_write(path: Path, content: str) -> None:
    """异步写入文件"""
    loop = asyncio.get_event_loop()

    def _write():
        path.write_text(content, encoding="utf-8")

    await loop.run_in_executor(None, _write)


# --- Tool Set ---


class IsolatedToolSet:
    """隔离工具集 — 为 Subagent 提供受限的文件操作 + 终端工具"""

    def __init__(self, worktree_root: Path):
        self._worktree_root = worktree_root

        self._read_file = IsolatedReadFileTool(worktree_root)
        self._write_file = IsolatedWriteFileTool(worktree_root)
        self._edit_file = IsolatedEditFileTool(worktree_root)
        self._ls = IsolatedLsTool(worktree_root)
        self._glob = IsolatedGlobTool(worktree_root)
        self._grep = IsolatedGrepTool(worktree_root)
        self._terminal = IsolatedTerminalTool(worktree_root)

    @property
    def read_file(self) -> IsolatedReadFileTool:
        return self._read_file

    @property
    def write_file(self) -> IsolatedWriteFileTool:
        return self._write_file

    @property
    def edit_file(self) -> IsolatedEditFileTool:
        return self._edit_file

    @property
    def ls(self) -> IsolatedLsTool:
        return self._ls

    @property
    def glob(self) -> IsolatedGlobTool:
        return self._glob

    @property
    def grep(self) -> IsolatedGrepTool:
        return self._grep

    @property
    def terminal(self) -> IsolatedTerminalTool:
        return self._terminal

    def get_tools(self) -> list[Tool]:
        """获取所有工具"""
        return [
            self._read_file, self._write_file, self._edit_file,
            self._ls, self._glob, self._grep, self._terminal,
        ]

    def get_tool_schemas(self) -> list[dict]:
        """获取工具 schemas"""
        return [tool.to_tool_schema() for tool in self.get_tools()]

    def set_worktree_root(self, root: Path) -> None:
        """更新 worktree 根目录"""
        self._worktree_root = root
        self._read_file.set_worktree_root(root)
        self._write_file.set_worktree_root(root)
        self._edit_file.set_worktree_root(root)
        self._ls.set_worktree_root(root)
        self._glob.set_worktree_root(root)
        self._grep.set_worktree_root(root)
        self._terminal.set_worktree_root(root)


# 导出
__all__ = [
    "PathSecurityError",
    "validate_path",
    "IsolatedReadFileTool",
    "IsolatedWriteFileTool",
    "IsolatedEditFileTool",
    "IsolatedLsTool",
    "IsolatedGlobTool",
    "IsolatedGrepTool",
    "IsolatedTerminalTool",
    "IsolatedToolSet",
]