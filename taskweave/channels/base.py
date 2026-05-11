"""
通道抽象基类
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Channel(ABC):
    """通道抽象基类"""

    @property
    @abstractmethod
    def name(self) -> str:
        """通道名称"""
        ...

    @abstractmethod
    async def start(self) -> None:
        """启动通道 — 开始接收用户输入"""
        ...

    @abstractmethod
    async def stop(self) -> None:
        """停止通道"""
        ...
