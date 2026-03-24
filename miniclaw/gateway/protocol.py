"""
协议定义

定义 Gateway WebSocket 服务器与客户端之间的 JSON 消息协议。
所有通信都通过 JSON 消息进行，格式统一。

协议消息类型:
  客户端 → 服务器:
    {"type": "auth", "token": "..."}          # 认证
    {"type": "message", "content": "...", "session_id": "..."}  # 发送消息
    {"type": "new_session"}                    # 创建新会话
    {"type": "list_sessions"}                  # 列出会话
    {"type": "ping"}                           # 心跳

  服务器 → 客户端:
    {"type": "auth_ok"}                        # 认证成功
    {"type": "auth_error", "message": "..."}   # 认证失败
    {"type": "event", "event_type": "...", ...} # Agent 事件
    {"type": "sessions", "sessions": [...]}     # 会话列表
    {"type": "session_created", "session_id": "..."} # 新会话
    {"type": "pong"}                            # 心跳回复
    {"type": "error", "message": "..."}         # 错误
"""

from __future__ import annotations

from typing import Any

from miniclaw.types.events import Event


def make_auth_request(token: str) -> dict[str, Any]:
    """构建认证请求"""
    return {"type": "auth", "token": token}


def make_message_request(content: str, session_id: str | None = None) -> dict[str, Any]:
    """构建消息请求"""
    msg = {"type": "message", "content": content}
    if session_id:
        msg["session_id"] = session_id
    return msg


def make_new_session_request() -> dict[str, Any]:
    return {"type": "new_session"}


def make_list_sessions_request() -> dict[str, Any]:
    return {"type": "list_sessions"}


def make_ping() -> dict[str, Any]:
    return {"type": "ping"}


# --- 服务器响应 ---

def make_auth_ok() -> dict[str, Any]:
    return {"type": "auth_ok"}


def make_auth_error(message: str) -> dict[str, Any]:
    return {"type": "auth_error", "message": message}


def make_event_response(event: Event) -> dict[str, Any]:
    """将 Agent Event 转为协议消息"""
    return {
        "type": "event",
        "event_type": event.event_type.value,
        "data": event.data,
        "session_id": event.session_id,
        "timestamp": event.timestamp.isoformat(),
    }


def make_sessions_response(sessions: list[dict]) -> dict[str, Any]:
    return {"type": "sessions", "sessions": sessions}


def make_session_created(session_id: str) -> dict[str, Any]:
    return {"type": "session_created", "session_id": session_id}


def make_pong() -> dict[str, Any]:
    return {"type": "pong"}


def make_error(message: str) -> dict[str, Any]:
    return {"type": "error", "message": message}
