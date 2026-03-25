"""
WebSocket 服务器

Gateway 是 MiniClaw 的中央消息枢纽:
  - 接受 WebSocket 客户端连接
  - 处理认证
  - 将消息路由到 Agent
  - 将 Agent 事件流转发给客户端

使用 websockets 库实现异步 WebSocket 服务器。
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import websockets
from websockets.asyncio.server import ServerConnection

from miniclaw.agents.agent import Agent
from miniclaw.config.settings import GatewayConfig
from miniclaw.sessions.manager import SessionManager
from miniclaw.utils.logging import get_logger

from .auth import TokenAuth
from .router import MessageRouter

logger = get_logger(__name__)


class GatewayServer:
    """WebSocket 网关服务器

    每个客户端连接对应一个独立的处理循环:
      1. 接受连接
      2. 认证（如果启用）
      3. 接收消息 → 路由 → 返回结果
      4. 断开连接
    """

    def __init__(
        self,
        config: GatewayConfig,
        agent: Agent,
        session_manager: SessionManager,
    ):
        self._config = config
        self._auth = TokenAuth(config.auth_token)
        self._router = MessageRouter(agent, session_manager, self._auth)
        self._server = None
        self._connections: set[ServerConnection] = set()

    async def start(self) -> None:
        """启动 WebSocket 服务器"""
        self._server = await websockets.asyncio.serve(
            self._handle_connection,
            self._config.host,
            self._config.port,
        )
        auth_status = "启用" if self._auth.enabled else "未启用"
        logger.info(
            "Gateway 已启动: ws://%s:%d (认证: %s)",
            self._config.host, self._config.port, auth_status,
        )

    async def stop(self) -> None:
        """停止服务器"""
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            logger.info("Gateway 已停止")

    async def _handle_connection(self, websocket: ServerConnection) -> None:
        """处理单个客户端连接"""
        remote = websocket.remote_address
        logger.info("新连接: %s", remote)
        self._connections.add(websocket)

        is_authenticated = not self._auth.enabled  # 未启用认证则默认已认证

        async def send(data: dict[str, Any]) -> None:
            """发送 JSON 消息给客户端"""
            try:
                await websocket.send(json.dumps(data, ensure_ascii=False, default=str))
            except websockets.exceptions.ConnectionClosed:
                pass

        try:
            async for raw_message in websocket:
                if isinstance(raw_message, bytes):
                    raw_message = raw_message.decode("utf-8")
                is_authenticated = await self._router.handle_message(
                    raw_message, send, is_authenticated
                )
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as e:
            logger.error("连接处理错误: %s", e, exc_info=True)
        finally:
            self._connections.discard(websocket)
            logger.info("连接断开: %s", remote)

    @property
    def connection_count(self) -> int:
        return len(self._connections)
