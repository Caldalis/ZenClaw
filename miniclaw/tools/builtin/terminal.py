"""
终端命令执行技能 — 内置技能
在限定目录下执行轻量级终端命令，带有安全防护。
"""

from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path
from typing import Any

from miniclaw.tools.base import Tool


class TerminalCommandSkill(Tool):
    """终端命令执行 — 执行系统命令并返回结果"""

    @property
    def name(self) -> str:
        return "terminal_command"

    @property
    def description(self) -> str:
        return "在当前目录下执行轻量级的终端命令（支持 Windows 和 Linux）。包含危险命令拦截机制和 30 秒超时限制。"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "要执行的终端命令（例如：dir, ls -l, pip install xx）。在 Windows 上 ls 会自动映射为 dir。",
                },
                "timeout": {
                    "type": "integer",
                    "description": "超时时间（秒），默认 30",
                    "default": 30,
                },
            },
            "required": ["command"],
        }

    async def execute(self, **kwargs: Any) -> str:
        command = kwargs.get("command", "")
        timeout = kwargs.get("timeout", 30)

        if not command:
            return "错误：请提供要执行的命令"

        # 危险命令黑名单 (包括 Linux 和 Windows 系统高危命令)
        blacklist = [
            "rm -rf /",
            "mkfs",
            "sudo",
            "reboot",
            "shutdown",
            "init 0",
            "init 6",
            "del /f /s /q",
            "rmdir /s /q",
            "format",
            "diskpart",
            "powercfg",
            "bcdedit",
        ]

        # 简单的安全拦截
        cmd_lower = command.lower()
        for bad in blacklist:
            if bad in cmd_lower:
                return f"错误: 触发安全拦截，不允许执行包含 '{bad}' 的危险命令。"

        # Windows 跨平台命令适配
        command = self._adapt_for_windows(command)

        base_dir = Path.cwd().resolve()

        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=str(base_dir),
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            output = result.stdout if result.returncode == 0 else result.stderr
            if not output.strip():
                output = f"命令已成功执行，退出码: {result.returncode}，无输出内容。"

            return output
        except subprocess.TimeoutExpired:
            return f"错误: 命令执行超时 (超过 {timeout} 秒)"
        except Exception as e:
            return f"错误: 命令执行异常 - {str(e)}"

    def _adapt_for_windows(self, command: str) -> str:
        """将 Unix 命令适配为 Windows 等价命令

        只替换命令的第一词（命令名），不替换参数。
        例如: ls -la → dir -la，但保留参数不变。
        """
        if platform.system() != "Windows":
            return command

        # 常见的 Unix → Windows 命令映射
        unix_to_win = {
            "ls": "dir",
            "cat": "type",
            "mv": "move",
            "cp": "copy",
            "rm": "del",
            "mkdir": "md",
            "rmdir": "rd",
            "pwd": "cd",
            "touch": "type nul >",
            "grep": "findstr",
            "which": "where",
        }

        # 解析命令的第一个词
        parts = command.strip().split(None, 1)
        if not parts:
            return command

        first_word = parts[0]
        rest = parts[1] if len(parts) > 1 else ""

        # 只替换独立的命令名（不在管道、引号等上下文中）
        if first_word in unix_to_win:
            adapted = unix_to_win[first_word]
            if rest:
                return f"{adapted} {rest}"
            return adapted

        return command