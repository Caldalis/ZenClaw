"""
CLI 终端通道

通过命令行终端与用户交互:
  - 读取 stdin 获取用户输入
  - 将 Agent 事件流输出到 stdout
  - 使用 Rich 库美化输出（可选）

这是最基本的通道，不需要网络连接。

实现方式:
  CLI Channel 不通过 WebSocket 连接 Gateway，
  而是直接在进程内调用 Agent（通过 asyncio.Queue 模拟消息传递）。
  这样启动更简单，适合开发和测试。
"""

from __future__ import annotations

import asyncio
import sys

from typing import Any
from miniclaw.sessions.manager import SessionManager
from miniclaw.types.enums import EventType, Role
from miniclaw.types.messages import Message
from miniclaw.utils.logging import get_logger
from datetime import datetime
from .base import Channel

logger = get_logger(__name__)


class CLIChannel(Channel):
    """CLI 终端通道"""

    def __init__(self, agent: Any, session_manager: SessionManager):
        self._agent = agent
        self._session_mgr = session_manager
        self._running = False


    @property
    def name(self) -> str:
        return "cli"

    async def start(self) -> None:
        """启动 CLI 交互循环"""
        self._running = True

        # 尝试使用 Rich 美化输出
        try:
            from rich.console import Console
            from rich.markdown import Markdown
            from rich.panel import Panel
            console = Console()
            use_rich = True
        except ImportError:
            console = None
            use_rich = False

        # 创建或恢复会话
        sessions = await self._session_mgr.list_sessions()
        if sessions:
            session = sessions[0]  # 使用最近的会话
            self._session_id = session.id
            if use_rich:
                console.print(f"[dim]恢复会话: {session.id[:8]}... ({session.message_count} 条消息)[/dim]")
            else:
                print(f"恢复会话: {session.id[:8]}... ({session.message_count} 条消息)")
        else:
            session = await self._session_mgr.create_session(title="CLI 会话")
            self._session_id = session.id

        # 欢迎消息
        if use_rich:
            console.print(Panel.fit(
                "[bold cyan]ZenClaw[/bold cyan] — AI 助手\n"
                "输入消息开始对话，输入 [bold]/quit[/bold] 退出，[bold]/new[/bold] 新建会话",
                border_style="cyan",
            ))
        else:
            print("=" * 50)
            print("ZenClaw — AI 助手")
            print("输入消息开始对话，输入 /quit 退出，/new 新建会话")
            print("=" * 50)

        # 主循环
        while self._running:
            try:
                # 读取用户输入
                if use_rich:
                    user_input = console.input("[bold green]> [/bold green]")
                else:
                    user_input = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: input("> ")
                    )

                user_input = user_input.strip()
                if not user_input:
                    continue

                # 处理命令
                if user_input.lower() in ("/quit", "/exit", "/q"):
                    if use_rich:
                        console.print("[dim]再见！[/dim]")
                    else:
                        print("再见！")
                    break

                if user_input.lower() == "/new":
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
                    session = await self._session_mgr.create_session(title=f"CLI 会话 {current_time}")
                    self._session_id = session.id
                    if use_rich:
                        console.print(f"[dim]新会话: {session.id[:8]}[/dim]")
                    else:
                        print(f"新会话: {session.id[:8]}")
                    continue

                if user_input.lower() == "/sessions":
                    sessions = await self._session_mgr.list_sessions()
                    for s in sessions:
                        marker = " ←" if s.id == self._session_id else ""
                        print(f"  {s.id[:8]}  标题：{s.title}{marker}")
                    continue

                # 发送消息给 Agent
                user_msg = Message(role=Role.USER, content=user_input)

                if use_rich:
                    console.print("[bold blue]AI> [/bold blue]", end="")
                else:
                    print("AI> ", end="", flush=True)

                full_text = ""
                async for event in self._agent.process_message(user_msg, self._session_id):
                    if event.event_type == EventType.TEXT_DELTA:
                        text = event.data.get("text", "")
                        print(text, end="", flush=True)
                        full_text += text

                    elif event.event_type == EventType.TOOL_CALL_START:
                        tool_name = event.data.get("name", "")
                        if use_rich:
                            console.print(f"\n[dim]  🔧 调用工具: {tool_name}[/dim]", end="")
                        else:
                            print(f"\n  [调用工具: {tool_name}]", end="", flush=True)

                    elif event.event_type == EventType.TOOL_CALL_RESULT:
                        result = event.data.get("result", "")
                        tool_name = event.data.get("name", "")
                        if use_rich:
                            console.print(f"\n[dim]  ✅ {tool_name} 结果: {result[:100]}[/dim]")
                        else:
                            print(f"\n  [{tool_name} 结果: {result[:100]}]")

                    elif event.event_type == EventType.ERROR:
                        error_msg = event.data.get("message", "未知错误")
                        if use_rich:
                            console.print(f"\n[bold red]  ❌ 错误: {error_msg}[/bold red]")
                        else:
                            print(f"\n  [错误: {error_msg}]")

                    elif event.event_type == EventType.THINKING:
                        pass  # CLI 不展示 thinking 状态

                    elif event.event_type == EventType.DONE:
                        print()  # 换行

                print()  # 额外空行

            except (EOFError, KeyboardInterrupt):
                print("\n再见！")
                break
            except Exception as e:
                logger.error("CLI 处理错误: %s", e, exc_info=True)
                print(f"\n[错误: {e}]")

    async def stop(self) -> None:
        self._running = False
