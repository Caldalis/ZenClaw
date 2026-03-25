"""
通道注册表

管理所有可用的通道类型。
根据配置启用指定的通道。
"""

from __future__ import annotations

from typing import Type

from miniclaw.utils.logging import get_logger

from .base import Channel

logger = get_logger(__name__)

# 已注册的通道类型
_channel_types: dict[str, Type[Channel]] = {}


def register_channel_type(name: str, channel_class: Type[Channel]) -> None:
    """注册通道类型"""
    _channel_types[name] = channel_class
    logger.debug("注册通道类型: %s", name)


def get_channel_type(name: str) -> Type[Channel] | None:
    """获取通道类型"""
    return _channel_types.get(name)


def list_channel_types() -> list[str]:
    """列出所有已注册的通道类型"""
    return list(_channel_types.keys())
