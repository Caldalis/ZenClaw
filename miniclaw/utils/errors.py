"""
自定义异常
定义了层级化的错误类型，便于上层精确捕获和处理。
"""


class MiniClawError(Exception):
    """MiniClaw 所有异常的基类"""
    pass


class ConfigError(MiniClawError):
    """配置错误 — API Key 缺失、YAML 格式错误等"""
    pass


class ProviderError(MiniClawError):
    """AI 提供商错误 — API 调用失败、模型不可用等"""
    pass


class SkillError(MiniClawError):
    """技能执行错误 — 工具调用失败"""
    pass


class AuthError(MiniClawError):
    """认证错误 — Token 无效"""
    pass


class SessionError(MiniClawError):
    """会话错误 — 会话不存在、已过期等"""
    pass


class MemoryError_(MiniClawError):
    """记忆存储错误 — 数据库操作失败

    注意: 不使用 MemoryError 因为它是 Python 内置异常名。
    """
    pass
