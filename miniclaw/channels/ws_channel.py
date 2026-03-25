"""
WebSocket 客户端通道

作为 WebSocket 客户端连接到远程 Gateway 服务器。
适用于将 MiniClaw 集成到其他系统（Web 应用、Bot 等）。

与 CLI Channel 不同:
  - CLI Channel 直接调用 Agent（进程内）
  - WS Channel 通过 WebSocket 连接 Gateway（可以跨进程/跨网络）
"""

from __future__ import annotations

import asyncio
import json
from typing import Callable, Awaitable, Any

import websockets

from miniclaw.gateway.protocol import make_auth_request, make_message_request
from miniclaw.utils.logging import get_logger

from .base import Channel

logger = get_logger(__name__)

# 事件回调类型
EventCallback = Callable[[dict[str, Any]], Awaitable[None]]


class WebSocketChannel(Channel):
    """WebSocket 客户端通道"""

    def __init__(
        self,
        url: str = "ws://localhost:8765",
        auth_token: str = "",
        on_event: EventCallback | None = None,
    ):
        self._url = url
        self._auth_token = auth_token
        self._on_event = on_event
        self._ws = None
        self._running = False
        self._session_id: str | None = None

    @property
    def name(self) -> str:
        return "ws"

    async def start(self) -> None:
        """连接到 Gateway 并开始接收消息"""
        self._running = True
        logger.info("连接到 Gateway: %s", self._url)

        try:
            async with websockets.connect(self._url) as ws:
                self._ws = ws

                # 认证
                if self._auth_token:
                    await ws.send(json.dumps(make_auth_request(self._auth_token)))
                    response = json.loads(await ws.recv())
                    if response.get("type") != "auth_ok":
                        logger.error("认证失败: %s", response.get("message", ""))
                        return
                    logger.info("认证成功")

                # 接收消息循环
                async for raw_message in ws:
                    if not self._running:
                        break
                    try:
                        data = json.loads(raw_message)
                        if self._on_event:
                            await self._on_event(data)
                        # 记录 session_id
                        if data.get("type") == "session_created":
                            self._session_id = data.get("session_id")
                    except json.JSONDecodeError:
                        logger.warning("收到无效 JSON")
                    except Exception as e:
                        logger.error("处理消息错误: %s", e)

        except websockets.exceptions.ConnectionClosed:
            logger.info("连接已关闭")
        except Exception as e:
            logger.error("WebSocket 连接错误: %s", e)
        finally:
            self._ws = None

    async def stop(self) -> None:
        """断开连接"""
        self._running = False
        if self._ws:
            await self._ws.close()

    async def send_message(self, content: str, session_id: str | None = None) -> None:
        """发送消息到 Gateway"""
        if not self._ws:
            raise RuntimeError("未连接到 Gateway")
        sid = session_id or self._session_id
        msg = make_message_request(content, sid)
        await self._ws.send(json.dumps(msg, ensure_ascii=False))

    @property
    def connected(self) -> bool:
        return self._ws is not None
