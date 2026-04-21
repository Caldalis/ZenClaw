"""
消息路由器

职责:
  1. 解析客户端发来的 JSON 消息
  2. 根据消息类型路由到相应处理器
  3. 调用 Agent 处理用户消息
  4. 将 Agent 事件流转发给客户端

这是 Gateway 的核心逻辑，连接了协议层和业务层。
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Callable, Awaitable

from miniclaw.sessions.manager import SessionManager
from miniclaw.types.enums import EventType, Role
from miniclaw.types.events import Event
from miniclaw.types.messages import Message
from miniclaw.utils.logging import get_logger

from .auth import TokenAuth
from .protocol import (
    make_auth_error, make_auth_ok, make_error, make_event_response,
    make_pong, make_session_created, make_sessions_response,
)

logger = get_logger(__name__)

# 发送回调类型: 将 JSON dict 发送给客户端
SendCallback = Callable[[dict[str, Any]], Awaitable[None]]

# Agent 协议 — 任何有 process_message 方法的对象都可作为 Agent
AgentProtocol = Any  # 兼容 Agent 和 MasterAgent


class MessageRouter:
    """消息路由器 — 处理客户端请求"""

    def __init__(self, agent: AgentProtocol, session_manager: SessionManager, auth: TokenAuth):
        self._agent = agent
        self._session_mgr = session_manager
        self._auth = auth

    async def handle_message(
        self,
        raw_message: str,
        send: SendCallback,
        is_authenticated: bool = False,
    ) -> bool:
        """处理一条客户端消息

        Args:
            raw_message: 原始 JSON 字符串
            send: 发送回调函数
            is_authenticated: 当前连接是否已认证

        Returns:
            认证后的状态 (True = 已认证)
        """
        try:
            data = json.loads(raw_message)
        except json.JSONDecodeError:
            await send(make_error("无效的 JSON 格式"))
            return is_authenticated

        msg_type = data.get("type", "")

        # --- 认证 ---
        if msg_type == "auth":
            token = data.get("token", "")
            if self._auth.verify(token):
                await send(make_auth_ok())
                return True
            else:
                await send(make_auth_error("认证失败: Token 无效"))
                return False

        # --- 心跳 ---
        if msg_type == "ping":
            await send(make_pong())
            return is_authenticated

        # --- 需要认证的操作 ---
        if self._auth.enabled and not is_authenticated:
            await send(make_auth_error("请先认证"))
            return False

        # --- 发送消息 ---
        if msg_type == "message":
            content = data.get("content", "")
            session_id = data.get("session_id")

            if not content:
                await send(make_error("消息内容不能为空"))
                return is_authenticated

            # 获取或创建会话
            session = await self._session_mgr.get_or_create_session(session_id)
            session_id = session.id

            if not data.get("session_id"):
                await send(make_session_created(session_id))

            # 创建用户消息
            user_msg = Message(role=Role.USER, content=content, session_id=session_id)

            # 调用 Agent 并转发事件
            async for event in self._agent.process_message(user_msg, session_id):
                await send(make_event_response(event))

            return is_authenticated

        # --- 创建新会话 ---
        if msg_type == "new_session":
            session = await self._session_mgr.create_session()
            await send(make_session_created(session.id))
            return is_authenticated

        # --- 列出会话 ---
        if msg_type == "list_sessions":
            sessions = await self._session_mgr.list_sessions()
            sessions_data = [
                {"id": s.id, "title": s.title, "message_count": s.message_count}
                for s in sessions
            ]
            await send(make_sessions_response(sessions_data))
            return is_authenticated

        # --- 未知类型 ---
        await send(make_error(f"未知消息类型: {msg_type}"))
        return is_authenticated
