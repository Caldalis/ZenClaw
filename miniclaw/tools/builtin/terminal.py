"""
终端命令执行技能 — 内置技能
在限定目录下执行轻量级终端命令，带有安全防护。
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from miniclaw.tools.base import Skill


class TerminalCommandSkill(Skill):
    """终端命令执行 — 执行系统命令并返回结果"""

    @property
    def name(self) -> str:
        return "terminal_command"

    @property
    def description(self) -> str:
        return "在当前目录下执行轻量级的终端命令（支持 Windows 和 Linux）。包含危险命令拦截机制和 10 秒超时限制。"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "要执行的终端命令（例如：dir, ls -l, pip install xx）",
                },
            },
            "required": ["command"],
        }

    async def execute(self, **kwargs: Any) -> str:
        command = kwargs.get("command", "")

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

        base_dir = Path.cwd().resolve()

        try:
            # cwd 确保命令默认在当前目录执行，shell=True 允许使用管道等特性
            result = subprocess.run(
                command,
                shell=True,
                cwd=str(base_dir),
                capture_output=True,
                text=True,
                timeout=10,  # 防止 AI 写出死循环代码卡死程序
            )

            output = result.stdout if result.returncode == 0 else result.stderr
            if not output.strip():
                output = f"命令已成功执行，退出码: {result.returncode}，无输出内容。"

            return output
        except subprocess.TimeoutExpired:
            return "错误: 命令执行超时 (超过 10 秒)"
        except Exception as e:
            return f"错误: 命令执行异常 - {str(e)}"
