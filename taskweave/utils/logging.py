"""
日志配置

提供统一的日志工具，在 debug 模式下输出详细信息。
"""

from __future__ import annotations

import logging
import sys


def setup_logging(debug: bool = False) -> None:
    """配置全局日志

    Args:
        debug: 是否启用 DEBUG 级别日志
    """
    level = logging.DEBUG if debug else logging.INFO

    # 使用 Rich
    try:
        from rich.logging import RichHandler
        handler = RichHandler(rich_tracebacks=True, show_path=debug)
    except ImportError:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
        )

    logging.basicConfig(level=level, handlers=[handler], force=True)

    # 降低第三方库日志级别
    for noisy in ("httpx", "httpcore", "websockets", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """获取模块级 logger

    Usage:
        logger = get_logger(__name__)
        logger.info("消息已路由到 Agent")
    """
    return logging.getLogger(f"miniclaw.{name}")
