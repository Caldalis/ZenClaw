"""
Worktree 感知的工具 — 文件操作和命令执行限定在 worktree 边界内

核心安全原则:
  1. 所有文件操作必须在 worktree 目录内
  2. 禁止访问 worktree 外的文件
  3. 路径规范化防止路径穿越攻击
"""

from __future__ import annotations

import asyncio
import os
import subprocess
from pathlib import Path
from typing import Any

from miniclaw.tools.base import Tool
from miniclaw.utils.logging import get_logger

logger = get_logger(__name__)


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
    # 解析为绝对路径
    if not target_path.is_absolute():
        target_path = allowed_root / target_path

    # 规范化路径（解析 .. 和 符号链接）
    try:
        resolved = target_path.resolve()
    except Exception as e:
        raise PathSecurityError(f"路径解析失败: {target_path}") from e

    # 检查是否在允许的根目录内
    try:
        resolved.relative_to(allowed_root.resolve())
    except ValueError:
        raise PathSecurityError(
            f"路径不在工作目录内: {resolved} (allowed: {allowed_root})"
        )

    return resolved


class IsolatedFileReaderTool(Tool):
    """隔离的文件读取工具 — 限定在 worktree 目录内"""

    def __init__(self, worktree_root: Path | None = None):
        self._worktree_root = worktree_root

    def set_worktree_root(self, root: Path) -> None:
        """设置 worktree 根目录"""
        self._worktree_root = root

    @property
    def name(self) -> str:
        return "file_reader"

    @property
    def description(self) -> str:
        return "读取文件内容。支持文本文件和二进制文件。路径必须在当前工作目录内。"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "文件路径（相对于工作目录或绝对路径）",
                },
                "encoding": {
                    "type": "string",
                    "description": "文件编码，默认 utf-8",
                    "default": "utf-8",
                },
                "start_line": {
                    "type": "integer",
                    "description": "起始行号（可选，从 1 开始）",
                },
                "end_line": {
                    "type": "integer",
                    "description": "结束行号（可选）",
                },
            },
            "required": ["path"],
        }

    async def execute(self, **kwargs: Any) -> str:
        if self._worktree_root is None:
            return "错误: 工作目录未设置"

        file_path = Path(kwargs.get("path", ""))
        encoding = kwargs.get("encoding", "utf-8")
        start_line = kwargs.get("start_line")
        end_line = kwargs.get("end_line")

        try:
            # 验证路径安全性
            safe_path = validate_path(file_path, self._worktree_root)

            if not safe_path.exists():
                return f"文件不存在: {file_path}"

            if not safe_path.is_file():
                return f"不是文件: {file_path}"

            # 异步读取文件
            content = await self._read_file(safe_path, encoding, start_line, end_line)

            return content

        except PathSecurityError as e:
            logger.warning("路径安全违规: %s", e)
            return f"安全错误: {e}"

        except Exception as e:
            logger.error("读取文件失败: %s", e)
            return f"读取错误: {e}"

    async def _read_file(
        self,
        path: Path,
        encoding: str,
        start_line: int | None,
        end_line: int | None,
    ) -> str:
        """异步读取文件"""
        loop = asyncio.get_event_loop()

        def _read():
            with open(path, "r", encoding=encoding, errors="replace") as f:
                if start_line is None and end_line is None:
                    return f.read()

                lines = f.readlines()
                start = (start_line or 1) - 1
                end = end_line or len(lines)
                return "".join(lines[start:end])

        return await loop.run_in_executor(None, _read)


class IsolatedFileWriterTool(Tool):
    """隔离的文件写入工具 — 限定在 worktree 目录内"""

    def __init__(self, worktree_root: Path | None = None):
        self._worktree_root = worktree_root

    def set_worktree_root(self, root: Path) -> None:
        """设置 worktree 根目录"""
        self._worktree_root = root

    @property
    def name(self) -> str:
        return "file_writer"

    @property
    def description(self) -> str:
        return "写入文件内容。可以创建新文件或覆盖现有文件。路径必须在当前工作目录内。"

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
                "mode": {
                    "type": "string",
                    "enum": ["write", "append"],
                    "description": "写入模式: write=覆盖, append=追加",
                    "default": "write",
                },
                "encoding": {
                    "type": "string",
                    "description": "文件编码，默认 utf-8",
                    "default": "utf-8",
                },
            },
            "required": ["path", "content"],
        }

    async def execute(self, **kwargs: Any) -> str:
        if self._worktree_root is None:
            return "错误: 工作目录未设置"

        file_path = Path(kwargs.get("path", ""))
        content = kwargs.get("content", "")
        mode = kwargs.get("mode", "write")
        encoding = kwargs.get("encoding", "utf-8")

        try:
            # 验证路径安全性
            safe_path = validate_path(file_path, self._worktree_root)

            # 确保父目录存在
            safe_path.parent.mkdir(parents=True, exist_ok=True)

            # 异步写入文件
            await self._write_file(safe_path, content, mode, encoding)

            return f"文件已写入: {file_path} ({len(content)} 字符)"

        except PathSecurityError as e:
            logger.warning("路径安全违规: %s", e)
            return f"安全错误: {e}"

        except Exception as e:
            logger.error("写入文件失败: %s", e)
            return f"写入错误: {e}"

    async def _write_file(
        self,
        path: Path,
        content: str,
        mode: str,
        encoding: str,
    ) -> None:
        """异步写入文件"""
        loop = asyncio.get_event_loop()

        def _write():
            write_mode = "a" if mode == "append" else "w"
            with open(path, write_mode, encoding=encoding) as f:
                f.write(content)

        await loop.run_in_executor(None, _write)


class IsolatedTerminalTool(Tool):
    """隔离的终端命令工具 — 在 worktree 目录内执行"""

    def __init__(self, worktree_root: Path | None = None):
        self._worktree_root = worktree_root
        self._timeout = 60000  # 默认 60 秒超时

    def set_worktree_root(self, root: Path) -> None:
        """设置 worktree 根目录"""
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

        # 安全检查：禁止危险命令
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
            "rm -rf /",
            "rm -rf ~",
            "rm -rf *",
            ":(){ :|:& };:",
            "mkfs",
            "dd if=",
            "> /dev/sd",
            "chmod -R 777 /",
            "curl | bash",
            "wget | bash",
        ]

        command_lower = command.lower()
        for pattern in dangerous_patterns:
            if pattern.lower() in command_lower:
                return True

        return False

    async def _run_command(
        self,
        command: str,
        timeout: int,
    ) -> dict[str, Any]:
        """执行命令"""
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


class IsolatedToolSet:
    """隔离工具集 — 为 Subagent 提供受限的工具集"""

    def __init__(self, worktree_root: Path):
        self._worktree_root = worktree_root

        # 创建隔离工具实例
        self._file_reader = IsolatedFileReaderTool(worktree_root)
        self._file_writer = IsolatedFileWriterTool(worktree_root)
        self._terminal = IsolatedTerminalTool(worktree_root)

    @property
    def file_reader(self) -> IsolatedFileReaderTool:
        return self._file_reader

    @property
    def file_writer(self) -> IsolatedFileWriterTool:
        return self._file_writer

    @property
    def terminal(self) -> IsolatedTerminalTool:
        return self._terminal

    def get_tools(self) -> list[Tool]:
        """获取所有工具"""
        return [self._file_reader, self._file_writer, self._terminal]

    def get_tool_schemas(self) -> list[dict]:
        """获取工具 schemas"""
        return [tool.to_tool_schema() for tool in self.get_tools()]

    def set_worktree_root(self, root: Path) -> None:
        """更新 worktree 根目录"""
        self._worktree_root = root
        self._file_reader.set_worktree_root(root)
        self._file_writer.set_worktree_root(root)
        self._terminal.set_worktree_root(root)


# 导出
__all__ = [
    "PathSecurityError",
    "validate_path",
    "IsolatedFileReaderTool",
    "IsolatedFileWriterTool",
    "IsolatedTerminalTool",
    "IsolatedToolSet",
]