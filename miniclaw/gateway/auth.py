"""
Token 认证
"""

from __future__ import annotations

import hmac

from miniclaw.utils.logging import get_logger

logger = get_logger(__name__)


class TokenAuth:
    """Token 认证器"""

    def __init__(self, token: str = ""):
        self._token = token
        self._enabled = bool(token)

    @property
    def enabled(self) -> bool:
        """是否启用认证"""
        return self._enabled

    def verify(self, token: str) -> bool:
        """验证 Token

        使用 hmac.compare_digest 防止时序攻击。
        """
        if not self._enabled:
            return True  # 未启用认证时始终通过
        return hmac.compare_digest(self._token, token)
